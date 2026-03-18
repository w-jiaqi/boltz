"""Cache trunk representations for evolution head training.

Runs Boltz2 inference on a set of input structures and saves the
intermediate representations needed by the EvolutionModule:
  - s_inputs  (input embedder output)
  - z         (pairformer pair representation)
  - x_pred    (best predicted atom coordinates, selected by iPTM)
  - feats     (token_pad_mask, token_to_rep_atom, mol_type, affinity_token_mask)

These cached .pt files are then used by train_evolution.py.

Prerequisites:
    - A Boltz2 checkpoint (structure + confidence)
    - Input structures in boltz predict format (YAML/FASTA files)
    - The boltz package installed (`pip install -e .`)

Usage:
    python scripts/train/cache_evolution_features.py \\
        --data /path/to/structures/ \\
        --output /path/to/cache/ \\
        --checkpoint /path/to/boltz2.ckpt \\
        --recycling_steps 3 \\
        --sampling_steps 200 \\
        --diffusion_samples 5

    The --data directory should contain YAML or FASTA files in the same
    format accepted by `boltz predict`.

    Each input structure will produce a file {complex_id}.pt in the
    output directory.

Alternative (manual caching):
    If you already have Boltz2 predictions and want to cache from them,
    you can write a simple script:

        model = Boltz2.load_from_checkpoint(...)
        model.eval()
        for batch in dataloader:
            with torch.no_grad():
                out = model(batch, recycling_steps=3, ...)
            torch.save({
                "s_inputs": out["s_inputs"][0].cpu().half(),
                "z": out["z"][0].cpu().half(),
                "x_pred": out["sample_atom_coords"][best_idx].cpu().half(),
                "token_pad_mask": batch["token_pad_mask"][0].cpu(),
                "token_to_rep_atom": batch["token_to_rep_atom"][0].cpu().half(),
                "mol_type": batch["mol_type"][0].cpu(),
                "affinity_token_mask": batch["affinity_token_mask"][0].cpu(),
            }, f"cache/{record_id}.pt")
"""

import argparse
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import BasePredictionWriter

# Boltz imports — these are available after `pip install -e .`
from boltz.data.module.inferencev2 import Boltz2InferenceDataModule
from boltz.main import check_inputs, process_inputs


# ---------------------------------------------------------------------------
# Prediction writer that saves cached features
# ---------------------------------------------------------------------------

FEAT_KEYS = ["token_pad_mask", "token_to_rep_atom", "mol_type", "affinity_token_mask"]


class EvolutionCacheWriter(BasePredictionWriter):
    """Saves trunk representations to .pt files during Boltz2 prediction."""

    def __init__(self, output_dir: str, save_half: bool = True):
        super().__init__(write_interval="batch")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.save_half = save_half

    def write_on_batch_end(
        self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx
    ):
        if prediction.get("exception", False):
            return

        record_id = batch["pdb_id"][0]
        out_path = self.output_dir / f"{record_id}.pt"

        maybe_half = lambda t: t.half() if self.save_half else t  # noqa: E731

        best_idx = 0
        if "iptm" in prediction and prediction["iptm"] is not None:
            best_idx = torch.argsort(prediction["iptm"], descending=True)[0].item()

        coords = prediction["coords"]
        if coords.dim() == 3:
            x_pred = coords[best_idx]
        else:
            x_pred = coords

        cache = {
            "s_inputs": maybe_half(prediction["s_inputs"][0].cpu()),
            "z": maybe_half(prediction["z"][0].cpu()),
            "x_pred": maybe_half(x_pred.cpu()),
        }

        for key in FEAT_KEYS:
            if key in batch:
                val = batch[key][0].cpu()
                if val.is_floating_point() and self.save_half:
                    val = val.half()
                cache[key] = val

        torch.save(cache, out_path)
        print(f"  Cached {record_id} → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@dataclass
class BoltzDiffusionParams:
    sigma_min: float = 0.0004
    sigma_max: float = 160.0
    sigma_data: float = 16.0
    rho: int = 7
    P_mean: float = -1.2
    P_std: float = 1.5
    gamma_0: float = 0.8
    gamma_min: float = 1.0
    noise_scale: float = 1.0
    step_scale: float = 1.0
    coordinate_augmentation: bool = True
    alignment_reverse_diff: bool = True
    synchronize_sigmas: bool = True
    use_inference_model_cache: bool = True


@dataclass
class BoltzPairformerParams:
    num_blocks: int = 48
    num_heads: int = 16
    dropout: float = 0.25
    activation_checkpointing: bool = False
    offload_to_cpu: bool = False


@dataclass
class BoltzMSAParams:
    msa_s: int = 64
    msa_blocks: int = 4
    msa_dropout: float = 0.15
    z_dropout: float = 0.25
    pairwise_head_width: int = 32
    pairwise_num_heads: int = 4
    activation_checkpointing: bool = False
    offload_to_cpu: bool = False


@dataclass
class BoltzSteeringParams:
    fk_steering: bool = False
    num_particles: int = 3
    fk_lambda: float = 4.0
    fk_resampling_interval: int = 3
    physical_guidance_update: bool = False
    num_gd_steps: int = 16
    contact_guidance_update: bool = False


def main():
    parser = argparse.ArgumentParser(
        description="Cache Boltz2 trunk representations for evolution head training."
    )
    parser.add_argument(
        "--data", required=True, help="Path to input structures (YAML/FASTA directory)"
    )
    parser.add_argument(
        "--output", required=True, help="Output directory for cached .pt files"
    )
    parser.add_argument(
        "--checkpoint", required=True, help="Path to Boltz2 checkpoint (.ckpt)"
    )
    parser.add_argument(
        "--cache", default="~/.boltz",
        help="Boltz cache directory for CCD dict etc.",
    )
    parser.add_argument("--recycling_steps", type=int, default=3)
    parser.add_argument("--sampling_steps", type=int, default=200)
    parser.add_argument("--diffusion_samples", type=int, default=5)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--accelerator", default="gpu")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--no_half", action="store_true", help="Save in float32")
    parser.add_argument("--use_msa_server", action="store_true",
                        help="Use MMseqs2 server for MSA generation")
    parser.add_argument("--no_kernels", action="store_true",
                        help="Disable custom CUDA kernels (use pure PyTorch)")
    args = parser.parse_args()

    data_path = Path(args.data)
    out_dir = Path(args.output)
    cache = Path(args.cache).expanduser()
    checkpoint = Path(args.checkpoint)

    out_dir.mkdir(parents=True, exist_ok=True)

    # Load molecule data (Boltz2 uses mols/ directory, not ccd.pkl)
    from boltz.data.mol import load_canonicals
    mol_dir = cache / "mols"
    if not mol_dir.exists():
        print(f"ERROR: Molecule data not found at {mol_dir}")
        print("Run `boltz predict` once to download it, or download manually:")
        print(f"  wget -O {cache}/mols.tar https://huggingface.co/boltz-community/boltz-2/resolve/main/mols.tar")
        print(f"  cd {cache} && tar -xf mols.tar")
        sys.exit(1)
    ccd = load_canonicals(mol_dir)

    # Discover and validate input files
    input_paths = check_inputs(data_path)
    if not input_paths:
        print(f"ERROR: No YAML/FASTA files found in {data_path}")
        sys.exit(1)
    print(f"Found {len(input_paths)} input files")

    # Process inputs (tokenize, compute MSA, etc.)
    processing_dir = out_dir / "_processing"
    ccd_path = cache / "ccd.pkl"
    manifest = process_inputs(
        data=input_paths,
        out_dir=processing_dir,
        ccd_path=ccd_path,
        mol_dir=mol_dir,
        use_msa_server=args.use_msa_server,
        msa_server_url="https://api.colabfold.com",
        msa_pairing_strategy="greedy",
        max_msa_seqs=4096,
        boltz2=True,
    )
    # Build a processed-input-like object for the data module
    processed_targets = processing_dir / "processed" / "targets"
    processed_msa = processing_dir / "processed" / "msa"

    # Load model
    predict_args = {
        "recycling_steps": args.recycling_steps,
        "sampling_steps": args.sampling_steps,
        "diffusion_samples": args.diffusion_samples,
        "max_parallel_samples": 1,
    }

    diffusion_params = BoltzDiffusionParams()
    pairformer_params = BoltzPairformerParams()
    msa_params = BoltzMSAParams()
    steering_params = BoltzSteeringParams()

    from boltz.model.models.boltz2 import Boltz2

    model = Boltz2.load_from_checkpoint(
        str(checkpoint),
        strict=True,
        predict_args=predict_args,
        map_location="cpu",
        diffusion_process_args=asdict(diffusion_params),
        pairformer_args=asdict(pairformer_params),
        msa_args=asdict(msa_params),
        steering_args=asdict(steering_params),
        ema=False,
        use_kernels=not args.no_kernels,
    )
    model.eval()

    # Create data module
    data_module = Boltz2InferenceDataModule(
        manifest=manifest,
        target_dir=processed_targets,
        msa_dir=processed_msa,
        mol_dir=mol_dir,
        num_workers=args.num_workers,
    )

    # Set up trainer with cache writer
    cache_writer = EvolutionCacheWriter(
        output_dir=str(out_dir),
        save_half=not args.no_half,
    )

    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        callbacks=[cache_writer],
        logger=False,
        enable_checkpointing=False,
    )

    print(f"\nCaching representations to {out_dir} ...")
    trainer.predict(model, datamodule=data_module, return_predictions=False)

    n_cached = len(list(out_dir.glob("*.pt")))
    print(f"\nDone. Cached {n_cached} complexes to {out_dir}")


if __name__ == "__main__":
    main()
