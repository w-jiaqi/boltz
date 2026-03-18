"""Cache trunk representations for evolution head training.

Runs Boltz2 inference on input structures and saves the intermediate
representations needed by the EvolutionModule. Uses the EXACT same
model loading path as `boltz predict` to avoid version mismatches.

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
from boltz.main import (
    Boltz2DiffusionParams,
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
        x_pred = coords[best_idx] if coords.dim() == 3 else coords

        # Reconstruct s_inputs by running the input embedder on the batch
        # (cheap operation, avoids needing s_inputs in predict_step output)
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
    args = parser.parse_args()

    data_path = Path(args.data)
    out_dir = Path(args.output)
    cache = Path(args.cache).expanduser()
    checkpoint = Path(args.checkpoint)

    out_dir.mkdir(parents=True, exist_ok=True)

    # Validate molecule data exists
    mol_dir = cache / "mols"
    if not mol_dir.exists():
        print(f"ERROR: Molecule data not found at {mol_dir}")
        print("Run `boltz predict` once to download it, or download manually:")
        print(f"  wget -O {cache}/mols.tar https://huggingface.co/boltz-community/boltz-2/resolve/main/mols.tar")
        print(f"  cd {cache} && tar -xf mols.tar")
        sys.exit(1)

    # --- Step 1: Process inputs (same as boltz predict) ---
    input_paths = check_inputs(data_path)
    if not input_paths:
        print(f"ERROR: No YAML/FASTA files found in {data_path}")
        sys.exit(1)
    print(f"Found {len(input_paths)} input files")

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
    processed_targets = processing_dir / "processed" / "targets"
    processed_msa = processing_dir / "processed" / "msa"

    # --- Step 2: Load model (EXACT same way as boltz predict) ---
    predict_args = {
        "recycling_steps": args.recycling_steps,
        "sampling_steps": args.sampling_steps,
        "diffusion_samples": args.diffusion_samples,
        "max_parallel_samples": 1,
        "write_confidence_summary": False,
        "write_full_pae": False,
        "write_full_pde": False,
    }

    diffusion_params = Boltz2DiffusionParams()
    pairformer_args = PairformerArgsV2()
    msa_args = MSAModuleArgs(subsample_msa=False, use_paired_feature=True)
    steering_args = BoltzSteeringParams()
    steering_args.fk_steering = False
    steering_args.physical_guidance_update = False
    steering_args.contact_guidance_update = False

    model = Boltz2.load_from_checkpoint(
        str(checkpoint),
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
    model.eval()

    # --- Step 3: Run prediction with cache writer ---
    data_module = Boltz2InferenceDataModule(
        manifest=manifest,
        target_dir=processed_targets,
        msa_dir=processed_msa,
        mol_dir=mol_dir,
        num_workers=args.num_workers,
    )

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
