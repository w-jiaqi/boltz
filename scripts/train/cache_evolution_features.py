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
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import BasePredictionWriter
from pytorch_lightning.strategies import DDPStrategy

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


def _get_total_seq_len(yaml_path: Path) -> int:
    """Read a YAML input file and return total sequence length across all chains."""
    import yaml
    try:
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        total = 0
        for entry in data.get("sequences", []):
            for entity_type, info in entry.items():
                seq = info.get("sequence", "")
                if seq:
                    total += len(seq)
        return total
    except Exception:
        return None


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
    parser.add_argument("--recycling_steps", type=int, default=10)
    parser.add_argument("--sampling_steps", type=int, default=200)
    parser.add_argument("--diffusion_samples", type=int, default=1)
    parser.add_argument("--devices", type=int, default=-1,
                        help="Number of GPUs/devices. -1 (default) auto-detects all "
                        "visible GPUs (e.g. via CUDA_VISIBLE_DEVICES / SLURM --gres). "
                        "When >1, runs DDP and shards the manifest across ranks.")
    parser.add_argument("--num_nodes", type=int, default=1,
                        help="Number of nodes for multi-node DDP. Default: 1.")
    parser.add_argument("--accelerator", default="gpu")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--no_half", action="store_true", help="Save in float32")
    parser.add_argument("--use_msa_server", action="store_true")
    parser.add_argument("--no_kernels", action="store_true")
    parser.add_argument("--skip_diffusion", action="store_true",
                        help="Skip diffusion sampling and use ground truth coords as x_pred. "
                        "Much faster (~2x) when you have experimental structures.")
    parser.add_argument("--max_total_tokens", type=int, default=None,
                        help="Skip complexes where len(seq_A) + len(seq_B) exceeds this. "
                        "Useful to avoid OOM on long sequences. Default: no limit.")
    parser.add_argument("--max_complexes", type=int, default=None,
                        help="Only cache the first N complexes (after filtering). "
                        "Useful for testing. Default: no limit.")
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

    # ---- Resolve device count and DDP topology ----
    if args.devices < 0:
        if args.accelerator == "gpu":
            n_devices = max(1, torch.cuda.device_count())
        else:
            n_devices = 1
    else:
        n_devices = args.devices
    is_distributed = n_devices > 1 or args.num_nodes > 1

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    node_rank = int(os.environ.get("NODE_RANK", 0))
    is_rank_zero = local_rank == 0 and node_rank == 0

    rank_tag = f"[rank {node_rank}.{local_rank}]" if is_distributed else ""

    # ---- Step 1: Discover, filter, process inputs (rank 0 only under DDP) ----
    # process_inputs hits the MSA server; we only want one rank doing that.
    # Other ranks wait until the manifest is on disk before continuing.
    manifest_path = out_dir / "processed" / "manifest.json"

    if is_rank_zero:
        if manifest_path.exists():
            print(f"Reusing existing manifest at {manifest_path}")
        else:
            input_paths = check_inputs(data_path)
            if not input_paths:
                print(f"ERROR: No YAML/FASTA files found in {data_path}")
                sys.exit(1)
            print(f"Found {len(input_paths)} input files")

            if args.max_total_tokens is not None:
                filtered = []
                skipped = 0
                for p in input_paths:
                    total_len = _get_total_seq_len(p)
                    if total_len is not None and total_len <= args.max_total_tokens:
                        filtered.append(p)
                    else:
                        skipped += 1
                input_paths = filtered
                print(f"  After token filter (<= {args.max_total_tokens}): "
                      f"{len(input_paths)} kept, {skipped} skipped")

            if args.max_complexes is not None and len(input_paths) > args.max_complexes:
                input_paths = input_paths[:args.max_complexes]
                print(f"  Truncated to first {args.max_complexes} complexes")

            if not input_paths:
                print("ERROR: No inputs remain after filtering.")
                sys.exit(1)

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
    else:
        print(f"{rank_tag} waiting for manifest at {manifest_path} ...")
        waited = 0
        while not manifest_path.exists():
            time.sleep(10)
            waited += 10
            if waited > 7200:  # 2 hours
                print(f"{rank_tag} timed out waiting for manifest.")
                sys.exit(1)
        time.sleep(5)  # grace period so the file is fully flushed

    manifest = Manifest.load(manifest_path)
    print(f"{rank_tag} Manifest has {len(manifest.records)} records")

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

    strategy = "auto"
    if is_distributed:
        strategy = DDPStrategy(find_unused_parameters=False)

    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=n_devices,
        num_nodes=args.num_nodes,
        strategy=strategy,
        callbacks=[cache_writer],
        logger=False,
        enable_checkpointing=False,
        precision="bf16-mixed",
        use_distributed_sampler=True,
    )

    print(f"\n{rank_tag} Caching representations to {out_dir} "
          f"(devices={n_devices}, nodes={args.num_nodes}) ...")
    trainer.predict(model, datamodule=data_module, return_predictions=False)

    if trainer.is_global_zero:
        n_cached = len(list(out_dir.glob("*.pt")))
        print(f"\nDone. Cached {n_cached} complexes to {out_dir}")


if __name__ == "__main__":
    main()
