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


# asym_id (per-token chain index) is needed for interface-only pooling in the
# evolution head (cross-chain A<->B pairs). Added so a fresh cache supports it.
FEAT_KEYS = ["token_pad_mask", "token_to_rep_atom", "mol_type", "affinity_token_mask", "asym_id"]


def compute_contact_map(x_pred, atom_to_token, ref_element, atom_pad_mask, big=1.0e4):
    """Residue-residue MINIMUM heavy-atom distance (Angstrom) -> [N_tok, N_tok].

    The cross-chain entries of this matrix are the protein-protein contact map:
    residues i, j are in contact when their closest heavy atoms are within a
    cutoff (CAPRI uses 5 A; the coevolution literature often 8 A). We cache the
    raw distance (not a thresholded mask) so the contact cutoff stays tunable
    at train time without re-caching. Hydrogens (atomic number 1) and padding
    atoms are excluded; empty residue pairs get `big`.

    atom_to_token : [N_atom, N_tok] one-hot
    ref_element   : [N_atom, num_elements] one-hot of atomic number
    atom_pad_mask : [N_atom]
    """
    dev = x_pred.device
    a2t = atom_to_token.to(dev)
    n_atom, n_tok = a2t.shape
    tok = a2t.argmax(-1)                                   # token index per atom
    elem = ref_element.to(dev).argmax(-1)                 # atomic number per atom
    heavy = (elem > 1) & atom_pad_mask.to(dev).bool()     # drop H(=1) + padding(=0)
    if int(heavy.sum()) < 2:
        return x_pred.new_full((n_tok, n_tok), big)
    coords = x_pred[heavy].float()                         # [nh, 3]
    tokh = tok[heavy]                                      # [nh]
    nh = coords.shape[0]
    d = torch.cdist(coords, coords)                        # [nh, nh]
    # min over target atoms grouped by their token -> [nh, n_tok]
    t1 = coords.new_full((nh, n_tok), big)
    t1.scatter_reduce_(1, tokh.view(1, nh).expand(nh, nh), d, reduce="amin", include_self=True)
    # min over source atoms grouped by their token -> [n_tok, n_tok]
    m = coords.new_full((n_tok, n_tok), big)
    m.scatter_reduce_(0, tokh.view(nh, 1).expand(nh, n_tok), t1, reduce="amin", include_self=True)
    return m


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

        # Interface contact map: residue-residue min heavy-atom distance, on the
        # SAME predicted coords used for x_pred. Cross-chain entries are the PPI
        # contact map the evolution head pools over. Stored raw (Angstrom) so the
        # contact cutoff is tunable without re-caching.
        if all(k in batch for k in ("atom_to_token", "ref_element", "atom_pad_mask")):
            cmap = compute_contact_map(
                x_pred.to(device),
                batch["atom_to_token"][0],
                batch["ref_element"][0],
                batch["atom_pad_mask"][0],
            )
            cache["iface_dist"] = maybe_half(cmap.cpu())

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

    if args.skip_diffusion:
        # Tell the model to run trunk but skip structure prediction.
        # The model still produces s, z, s_inputs and the predict_step
        # still returns them. Diffusion sampling is skipped entirely.
        load_kwargs["skip_run_structure"] = True

    model = Boltz2.load_from_checkpoint(str(checkpoint), **load_kwargs)
    model.eval()

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
