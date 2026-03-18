"""Cache trunk representations for evolution head training.

Runs Boltz2 inference on input structures and saves the intermediate
representations needed by the EvolutionModule. Uses the EXACT same
model loading and data pipeline as `boltz predict`.

Usage:
    python scripts/train/cache_evolution_features.py \
        --data /path/to/structures/ \
        --output /path/to/cache/ \
        --checkpoint /path/to/boltz2_conf.ckpt \
        --cache /path/to/boltz_cache \
        --use_msa_server \
        --no_kernels
"""

import argparse
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import BasePredictionWriter

from boltz.data.module.inferencev2 import Boltz2InferenceDataModule
from boltz.data.types import Manifest
from boltz.main import (
    Boltz2DiffusionParams,
    BoltzProcessedInput,
    BoltzSteeringParams,
    MSAModuleArgs,
    PairformerArgsV2,
    check_inputs,
    process_inputs,
)
from boltz.model.models.boltz2 import Boltz2


FEAT_KEYS = ["token_pad_mask", "token_to_rep_atom", "mol_type", "affinity_token_mask"]


class EvolutionCacheWriter(BasePredictionWriter):
    """Saves trunk representations to .pt files during prediction."""

    def __init__(self, output_dir: str, save_half: bool = True, skip_diffusion: bool = False):
        super().__init__(write_interval="batch")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.save_half = save_half
        self.skip_diffusion = skip_diffusion

    def write_on_batch_end(
        self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx
    ):
        if prediction.get("exception", False):
            return

        record_id = batch["record"][0].id
        out_path = self.output_dir / f"{record_id}.pt"

        maybe_half = lambda t: t.half() if self.save_half else t  # noqa: E731

        if self.skip_diffusion:
            # Use ground truth coords from the batch as x_pred
            # coords shape: [B, K, N_atoms, 3] — take first conformer
            gt_coords = batch["coords"]
            if gt_coords.dim() == 4:
                x_pred = gt_coords[0, 0]
            elif gt_coords.dim() == 3:
                x_pred = gt_coords[0]
            else:
                x_pred = gt_coords
        else:
            # Use best predicted coords (selected by iPTM)
            best_idx = 0
            if "iptm" in prediction and prediction["iptm"] is not None:
                best_idx = torch.argsort(prediction["iptm"], descending=True)[0].item()
            coords = prediction["coords"]
            x_pred = coords[best_idx] if coords.dim() == 3 else coords

        # Reconstruct s_inputs via input embedder (cheap, avoids needing
        # s_inputs in predict_step output for compatibility with original boltz)
        with torch.no_grad():
            device = next(pl_module.parameters()).device
            batch_device = {
                k: v.to(device) if torch.is_tensor(v) else v
                for k, v in batch.items()
            }
            s_inputs = pl_module.input_embedder(batch_device)

        cache = {
            "s_inputs": maybe_half(s_inputs[0].cpu()),
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
        print(f"  Cached {record_id} -> {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Cache Boltz2 trunk representations for evolution head training."
    )
    parser.add_argument("--data", required=True, help="Input structures (YAML/FASTA dir or file)")
    parser.add_argument("--output", required=True, help="Output directory for cached .pt files")
    parser.add_argument("--checkpoint", required=True, help="Path to boltz2_conf.ckpt")
    parser.add_argument("--cache", default="~/.boltz", help="Boltz cache directory (CCD/mols)")
    parser.add_argument("--recycling_steps", type=int, default=3)
    parser.add_argument("--sampling_steps", type=int, default=200)
    parser.add_argument("--diffusion_samples", type=int, default=5)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--accelerator", default="gpu")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--no_half", action="store_true", help="Save in float32")
    parser.add_argument("--use_msa_server", action="store_true")
    parser.add_argument("--no_kernels", action="store_true")
    parser.add_argument("--skip_diffusion", action="store_true",
                        help="Skip diffusion sampling and use ground truth coords as x_pred. "
                        "Much faster (~2x) when you have experimental structures.")
    args = parser.parse_args()

    data_path = Path(args.data)
    out_dir = Path(args.output)
    cache = Path(args.cache).expanduser()
    checkpoint = Path(args.checkpoint)

    out_dir.mkdir(parents=True, exist_ok=True)

    mol_dir = cache / "mols"
    if not mol_dir.exists():
        print(f"ERROR: Molecule data not found at {mol_dir}")
        sys.exit(1)

    # ---- Step 1: Process inputs (same as boltz predict) ----
    input_paths = check_inputs(data_path)
    if not input_paths:
        print(f"ERROR: No YAML/FASTA files found in {data_path}")
        sys.exit(1)
    print(f"Found {len(input_paths)} input files")

    ccd_path = cache / "ccd.pkl"
    process_inputs(
        data=input_paths,
        out_dir=out_dir,
        ccd_path=ccd_path,
        mol_dir=mol_dir,
        use_msa_server=args.use_msa_server,
        msa_server_url="https://api.colabfold.com",
        msa_pairing_strategy="greedy",
        max_msa_seqs=4096,
        boltz2=True,
    )

    # Load manifest (same as boltz predict line 1180)
    manifest = Manifest.load(out_dir / "processed" / "manifest.json")
    print(f"Manifest has {len(manifest.records)} records")

    # Build processed input paths (same as boltz predict lines 1190-1208)
    processed_dir = out_dir / "processed"
    processed = BoltzProcessedInput(
        manifest=manifest,
        targets_dir=processed_dir / "structures",
        msa_dir=processed_dir / "msa",
        constraints_dir=(
            (processed_dir / "constraints")
            if (processed_dir / "constraints").exists()
            else None
        ),
        template_dir=(
            (processed_dir / "templates")
            if (processed_dir / "templates").exists()
            else None
        ),
        extra_mols_dir=(
            (processed_dir / "mols") if (processed_dir / "mols").exists() else None
        ),
    )

    # ---- Step 2: Load model (same as boltz predict lines 1228-1326) ----
    diffusion_params = Boltz2DiffusionParams()
    diffusion_params.step_scale = 1.5
    pairformer_args = PairformerArgsV2()
    msa_args = MSAModuleArgs(subsample_msa=False, use_paired_feature=True)
    steering_args = BoltzSteeringParams()
    steering_args.fk_steering = False
    steering_args.physical_guidance_update = False

    predict_args = {
        "recycling_steps": args.recycling_steps,
        "sampling_steps": args.sampling_steps,
        "diffusion_samples": args.diffusion_samples,
        "max_parallel_samples": 1,
        "write_confidence_summary": False,
        "write_full_pae": False,
        "write_full_pde": False,
    }

    load_kwargs = dict(
        strict=True,
        predict_args=predict_args,
        map_location="cpu",
        diffusion_process_args=asdict(diffusion_params),
        ema=False,
        use_kernels=not args.no_kernels,
        pairformer_args=asdict(pairformer_args),
        msa_args=asdict(msa_args),
        steering_args=asdict(steering_args),
    )

    if args.skip_diffusion:
        # Tell the model to run trunk but skip structure prediction.
        # The model still produces s, z, s_inputs and the predict_step
        # still returns them. Diffusion sampling is skipped entirely.
        load_kwargs["skip_run_structure"] = True

    model = Boltz2.load_from_checkpoint(str(checkpoint), **load_kwargs)
    model.eval()

    # ---- Step 3: Create data module (same as boltz predict lines 1271-1282) ----
    data_module = Boltz2InferenceDataModule(
        manifest=processed.manifest,
        target_dir=processed.targets_dir,
        msa_dir=processed.msa_dir,
        mol_dir=mol_dir,
        num_workers=args.num_workers,
        constraints_dir=processed.constraints_dir,
        template_dir=processed.template_dir,
        extra_mols_dir=processed.extra_mols_dir,
    )

    # ---- Step 4: Run prediction with cache writer ----
    cache_writer = EvolutionCacheWriter(
        output_dir=str(out_dir),
        save_half=not args.no_half,
        skip_diffusion=args.skip_diffusion,
    )

    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        callbacks=[cache_writer],
        logger=False,
        enable_checkpointing=False,
        precision="bf16-mixed",
    )

    print(f"\nCaching representations to {out_dir} ...")
    trainer.predict(model, datamodule=data_module, return_predictions=False)

    n_cached = len(list(out_dir.glob("*.pt")))
    print(f"\nDone. Cached {n_cached} complexes to {out_dir}")


if __name__ == "__main__":
    main()
