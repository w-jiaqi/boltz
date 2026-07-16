"""Cache trunk representations for evolution head training.

COMMAND 3 of the evolution training pipeline.

Loads the manifest produced by ``build_manifest.py``, runs Boltz2 inference
across all GPUs (DDP-sharded), and writes one ``.pt`` cache file per
complex containing the intermediate representations needed by the
EvolutionModule. Uses the EXACT same model loading and data pipeline as
``boltz predict``.

Prerequisite:
    {output}/processed/manifest.json must already exist (created by
    build_manifest.py). This script does NOT call the MSA server.

Usage:
    python scripts/train/cache_evolution_features.py \\
        --output /path/to/cache/ \\
        --checkpoint /path/to/boltz2_conf.ckpt \\
        --cache /path/to/boltz_cache \\
        --no_kernels
"""

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

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
)
from boltz.model.models.boltz2 import Boltz2


FEAT_KEYS = ["token_pad_mask", "token_to_rep_atom", "mol_type", "affinity_token_mask"]


class EvolutionCacheWriter(BasePredictionWriter):
    """Saves trunk representations to .pt files during prediction.

    Normally saves the trunk's FINAL pair rep ``z`` (output of the last
    pairformer layer). When ``split_capture`` is provided, saves instead the
    pair rep at the INPUT to trunk layer ``split_capture.split_layer`` — the
    representation a TrunkTailModule needs so the tail layers can be re-run and
    fine-tuned at head-training time. ``split_capture`` is a mutable dict written
    by a forward-pre-hook (see ``register_split_hook``); its ``"z"`` entry is the
    most recent capture, which — with batch_size=1 and the trunk running once per
    record — is exactly this record's pre-split z.
    """

    def __init__(self, output_dir: str, save_half: bool = True, skip_diffusion: bool = False,
                 split_capture: dict = None, aux_from: str = None):
        super().__init__(write_interval="batch")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.save_half = save_half
        self.skip_diffusion = skip_diffusion
        self.split_capture = split_capture
        # If set, reuse x_pred from an existing per-record .pt in this dir instead
        # of running diffusion. Lets trunk-tail caching skip the (stochastic,
        # expensive) structure module while keeping the exact same predicted pose
        # as the baseline cache — so only z changes between baseline and tail runs.
        self.aux_from = Path(aux_from) if aux_from else None

    def write_on_batch_end(
        self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx
    ):
        if prediction.get("exception", False):
            return

        record_id = batch["record"][0].id
        out_path = self.output_dir / f"{record_id}.pt"

        maybe_half = lambda t: t.half() if self.save_half else t  # noqa: E731

        if self.aux_from is not None:
            # Reuse the predicted pose from the baseline cache; no diffusion ran.
            aux_path = self.aux_from / f"{record_id}.pt"
            if not aux_path.exists():
                raise FileNotFoundError(
                    f"--aux_from is set but {aux_path} does not exist. Every record "
                    f"being cached must have an x_pred in the baseline cache."
                )
            aux = torch.load(aux_path, map_location="cpu", weights_only=False)
            x_pred = aux["x_pred"]
            del aux
        elif self.skip_diffusion:
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

        if self.split_capture is not None:
            captured = self.split_capture.get("z", None)
            if captured is None:
                raise RuntimeError(
                    f"split_layer capture is empty for {record_id}: the forward "
                    f"pre-hook on the trunk layer never fired. Did the trunk run?"
                )
            z_to_save = captured[0]
        else:
            z_to_save = prediction["z"][0]

        cache = {
            "s_inputs": maybe_half(s_inputs[0].cpu()),
            "z": maybe_half(z_to_save.cpu()),
            "x_pred": maybe_half(x_pred.cpu()),
        }
        if self.split_capture is not None:
            cache["split_layer"] = int(self.split_capture["split_layer"])

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
    parser.add_argument("--output", required=True,
                        help="Cache root. Reads {output}/processed/manifest.json, "
                        "writes {output}/<record_id>.pt files.")
    parser.add_argument("--checkpoint", required=True, help="Path to boltz2_conf.ckpt")
    parser.add_argument("--cache", default="~/.boltz",
                        help="Boltz cache directory (CCD/mols). Default: ~/.boltz")
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
    parser.add_argument("--no_kernels", action="store_true")
    parser.add_argument("--skip_diffusion", action="store_true",
                        help="Skip diffusion sampling and use ground truth coords as x_pred. "
                        "Much faster (~2x) when you have experimental structures.")
    parser.add_argument("--max_complexes", type=int, default=None,
                        help="Only cache the first N records from the manifest. "
                        "Useful for sanity checks without rebuilding the manifest. "
                        "Default: cache everything in the manifest.")
    parser.add_argument("--split_layer", type=int, default=None,
                        help="For trunk-tail fine-tuning: instead of the final trunk "
                        "pair rep, cache the pair rep at the INPUT to trunk pairformer "
                        "layer SPLIT_LAYER (0-indexed). A TrunkTailModule then re-runs "
                        "layers [SPLIT_LAYER, num_blocks) at train time. Boltz-2 has 64 "
                        "blocks, so e.g. 60 keeps the last 4 layers replayable. Default: "
                        "None (cache final z, original behavior).")
    parser.add_argument("--aux_from", default=None,
                        help="Reuse x_pred from the baseline cache at this dir (one "
                        ".pt per record) instead of running diffusion. Implies the "
                        "structure module is skipped: only the trunk runs (to capture "
                        "the split-layer z), so caching is much cheaper AND the pose is "
                        "byte-identical to the baseline. Intended with --split_layer.")
    parser.add_argument("--num_shards", type=int, default=1,
                        help="Split the manifest into this many contiguous shards and "
                        "cache only --shard_index. For array jobs over one big manifest "
                        "(e.g. the odinz cache). Applied after --max_complexes.")
    parser.add_argument("--shard_index", type=int, default=0,
                        help="Which shard (0-indexed) of --num_shards to cache.")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip records whose .pt already exists in --output. Makes "
                        "caching resumable across requeues.")
    args = parser.parse_args()

    out_dir = Path(args.output)
    cache = Path(args.cache).expanduser()
    checkpoint = Path(args.checkpoint)

    out_dir.mkdir(parents=True, exist_ok=True)

    mol_dir = cache / "mols"
    if not mol_dir.exists():
        print(f"ERROR: Molecule data not found at {mol_dir}")
        sys.exit(1)

    manifest_path = out_dir / "processed" / "manifest.json"
    if not manifest_path.exists():
        print(f"ERROR: Manifest not found at {manifest_path}")
        print("Run build_manifest.py first to generate it.")
        sys.exit(1)

    if args.devices < 0:
        if args.accelerator == "gpu":
            n_devices = max(1, torch.cuda.device_count())
        else:
            n_devices = 1
    else:
        n_devices = args.devices
    is_distributed = n_devices > 1 or args.num_nodes > 1

    manifest = Manifest.load(manifest_path)
    print(f"Manifest has {len(manifest.records)} records")

    if args.max_complexes is not None and len(manifest.records) > args.max_complexes:
        manifest = Manifest(manifest.records[:args.max_complexes])
        print(f"Truncated manifest to first {args.max_complexes} records (debug)")

    if args.num_shards > 1:
        if not 0 <= args.shard_index < args.num_shards:
            print(f"ERROR: --shard_index must be in [0, {args.num_shards})")
            sys.exit(1)
        recs = manifest.records[args.shard_index::args.num_shards]
        manifest = Manifest(recs)
        print(f"Shard {args.shard_index}/{args.num_shards}: {len(recs)} records")

    if args.skip_existing:
        before = len(manifest.records)
        recs = [r for r in manifest.records if not (out_dir / f"{r.id}.pt").exists()]
        manifest = Manifest(recs)
        print(f"skip_existing: {before - len(recs)} already cached, {len(recs)} to do")
        if not recs:
            print("Nothing to cache in this shard; exiting.")
            return

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

    if args.skip_diffusion or args.aux_from:
        # Tell the model to run trunk but skip structure prediction.
        # The model still produces s, z, s_inputs and the predict_step
        # still returns them. Diffusion sampling is skipped entirely.
        # With --aux_from, x_pred comes from the baseline cache, so we never
        # need diffusion or ground-truth coords.
        load_kwargs["skip_run_structure"] = True

    model = Boltz2.load_from_checkpoint(str(checkpoint), **load_kwargs)
    model.eval()

    # Trunk-tail split capture: hook the input z of trunk layer `split_layer`.
    split_capture = None
    if args.split_layer is not None:
        pf = model.pairformer_module
        pf = getattr(pf, "_orig_mod", pf)  # unwrap torch.compile if present
        n_blocks = len(pf.layers)
        if not 0 < args.split_layer < n_blocks:
            print(f"ERROR: --split_layer must be in (0, {n_blocks}); got {args.split_layer}")
            sys.exit(1)
        split_capture = {"split_layer": args.split_layer, "z": None}

        def _pre_hook(module, inputs):
            # PairformerLayer.forward(s, z, mask, pair_mask, ...) -> inputs[1] is z.
            # Keep only the latest; with batch_size=1 and the trunk running once
            # per record, the value present at write time is this record's pre-split z.
            split_capture["z"] = inputs[1].detach()

        pf.layers[args.split_layer].register_forward_pre_hook(_pre_hook)
        print(f"Trunk-tail caching: saving z at INPUT to layer {args.split_layer} "
              f"of {n_blocks}; tail = layers [{args.split_layer}, {n_blocks}).")

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

    cache_writer = EvolutionCacheWriter(
        output_dir=str(out_dir),
        save_half=not args.no_half,
        skip_diffusion=args.skip_diffusion,
        split_capture=split_capture,
        aux_from=args.aux_from,
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

    print(f"\nCaching representations to {out_dir} "
          f"(devices={n_devices}, nodes={args.num_nodes}) ...")
    trainer.predict(model, datamodule=data_module, return_predictions=False)

    if trainer.is_global_zero:
        n_cached = len(list(out_dir.glob("*.pt")))
        print(f"\nDone. Cached {n_cached} complexes to {out_dir}")


if __name__ == "__main__":
    main()
