"""End-to-end inference: sequences → structure + evolutionary energy.

Runs the full Boltz2 pipeline (trunk + diffusion + confidence) and
applies the trained evolution head on the intermediates in a single pass.

Usage:
    python scripts/predict_evolution.py \
        --data /path/to/inputs/ \
        --out_dir /path/to/output/ \
        --boltz_checkpoint /path/to/boltz2_conf.ckpt \
        --evolution_checkpoint /path/to/evolution_training/checkpoints/last.ckpt \
        --cache /path/to/boltz_cache \
        --use_msa_server --no_kernels

Output per complex:
    output/{complex_id}/
    ├── {complex_id}_model_0.cif     # Predicted structure (from boltz)
    ├── confidence_*.json            # Confidence metrics (from boltz)
    └── evolution_{complex_id}.json  # { "evo_energy": float }
"""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import BasePredictionWriter

from boltz.data.module.inferencev2 import Boltz2InferenceDataModule
from boltz.data.types import Manifest
from boltz.data.write.writer import BoltzWriter
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

from train.train_evolution import EvolutionTrainingModule


FEAT_KEYS = ["token_pad_mask", "token_to_rep_atom", "mol_type", "affinity_token_mask"]


class EvolutionInferenceWriter(BasePredictionWriter):
    """Runs the evolution head and saves evo_energy alongside structures.

    The Boltz2 model produces structures, confidence, s, z, coords.
    This writer loads a trained EvolutionModule, applies it to the
    intermediates, and writes the result as a JSON file.
    """

    def __init__(
        self,
        output_dir: str,
        evolution_module: torch.nn.Module,
        structure_writer: BoltzWriter,
    ):
        super().__init__(write_interval="batch")
        self.output_dir = Path(output_dir)
        self.evolution_module = evolution_module
        self.structure_writer = structure_writer

    def write_on_batch_end(
        self, trainer, pl_module, prediction, batch_indices,
        batch, batch_idx, dataloader_idx
    ):
        # Let the standard BoltzWriter handle structure output
        self.structure_writer.write_on_batch_end(
            trainer, pl_module, prediction, batch_indices,
            batch, batch_idx, dataloader_idx,
        )

        if prediction.get("exception", False):
            return

        record_id = batch["record"][0].id
        device = next(pl_module.parameters()).device

        # Select best diffusion sample by iPTM
        best_idx = 0
        if "iptm" in prediction and prediction["iptm"] is not None:
            best_idx = torch.argsort(prediction["iptm"], descending=True)[0].item()

        coords = prediction["coords"]
        x_pred = coords[best_idx:best_idx + 1] if coords.dim() == 3 else coords.unsqueeze(0)

        # Get s_inputs from the input embedder
        with torch.no_grad():
            batch_device = {
                k: v.to(device) if torch.is_tensor(v) else v
                for k, v in batch.items()
            }
            s_inputs = pl_module.input_embedder(batch_device)

            z = prediction["z"].to(device)

            feats_device = {}
            for k in FEAT_KEYS:
                if k in batch:
                    feats_device[k] = batch[k].to(device)

            evo_out = self.evolution_module(
                s_inputs=s_inputs.float(),
                z=z.float(),
                x_pred=x_pred.to(device).float(),
                feats={k: v.float() if v.is_floating_point() else v
                       for k, v in feats_device.items()},
                multiplicity=1,
            )

        evo_energy = evo_out["evo_energy"].item()

        # Save alongside structure predictions
        struct_dir = self.output_dir / record_id
        struct_dir.mkdir(exist_ok=True)
        result = {"evo_energy": evo_energy}
        with open(struct_dir / f"evolution_{record_id}.json", "w") as f:
            json.dump(result, f, indent=2)

        print(f"  {record_id}: evo_energy = {evo_energy:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end Boltz2 + evolution head inference."
    )
    parser.add_argument("--data", required=True, help="Input YAML/FASTA dir or file")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    parser.add_argument("--boltz_checkpoint", required=True, help="Boltz2 checkpoint")
    parser.add_argument("--evolution_checkpoint", required=True,
                        help="Evolution head checkpoint (from train_evolution.py)")
    parser.add_argument("--cache", default="~/.boltz", help="Boltz cache directory")
    parser.add_argument("--recycling_steps", type=int, default=3)
    parser.add_argument("--sampling_steps", type=int, default=200)
    parser.add_argument("--diffusion_samples", type=int, default=5)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--accelerator", default="gpu")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--use_msa_server", action="store_true")
    parser.add_argument("--no_kernels", action="store_true")
    parser.add_argument("--output_format", default="mmcif", choices=["pdb", "mmcif"])
    args = parser.parse_args()

    data_path = Path(args.data)
    out_dir = Path(args.out_dir).expanduser()
    cache = Path(args.cache).expanduser()

    out_dir.mkdir(parents=True, exist_ok=True)

    mol_dir = cache / "mols"
    if not mol_dir.exists():
        print(f"ERROR: Molecule data not found at {mol_dir}")
        sys.exit(1)

    # ---- Process inputs (same as boltz predict) ----
    input_paths = check_inputs(data_path)
    print(f"Found {len(input_paths)} input files")

    ccd_path = cache / "ccd.pkl"
    process_inputs(
        data=input_paths, out_dir=out_dir,
        ccd_path=ccd_path, mol_dir=mol_dir,
        use_msa_server=args.use_msa_server,
        msa_server_url="https://api.colabfold.com",
        msa_pairing_strategy="greedy",
        max_msa_seqs=4096, boltz2=True,
    )

    manifest = Manifest.load(out_dir / "processed" / "manifest.json")
    processed_dir = out_dir / "processed"
    processed = BoltzProcessedInput(
        manifest=manifest,
        targets_dir=processed_dir / "structures",
        msa_dir=processed_dir / "msa",
        constraints_dir=(processed_dir / "constraints")
        if (processed_dir / "constraints").exists() else None,
        template_dir=(processed_dir / "templates")
        if (processed_dir / "templates").exists() else None,
        extra_mols_dir=(processed_dir / "mols")
        if (processed_dir / "mols").exists() else None,
    )

    # ---- Load Boltz2 model (same as boltz predict) ----
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
        "write_confidence_summary": True,
        "write_full_pae": False,
        "write_full_pde": False,
    }

    boltz_model = Boltz2.load_from_checkpoint(
        args.boltz_checkpoint, strict=True,
        predict_args=predict_args, map_location="cpu",
        diffusion_process_args=asdict(diffusion_params),
        ema=False, use_kernels=not args.no_kernels,
        pairformer_args=asdict(pairformer_args),
        msa_args=asdict(msa_args),
        steering_args=asdict(steering_args),
    )
    boltz_model.eval()

    # ---- Load evolution head from training checkpoint ----
    print(f"Loading evolution head from {args.evolution_checkpoint}")
    evo_lit_module = EvolutionTrainingModule.load_from_checkpoint(
        args.evolution_checkpoint, map_location="cpu",
    )
    evolution_module = evo_lit_module.evolution_module
    evolution_module.eval()
    print(f"  Evolution module: {sum(p.numel() for p in evolution_module.parameters()):,} params")

    # ---- Set up data module ----
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

    # ---- Set up writers ----
    predictions_dir = out_dir / "predictions"
    structure_writer = BoltzWriter(
        data_dir=processed.targets_dir,
        output_dir=predictions_dir,
        output_format=args.output_format,
        boltz2=True,
    )

    evo_writer = EvolutionInferenceWriter(
        output_dir=predictions_dir,
        evolution_module=evolution_module,
        structure_writer=structure_writer,
    )

    # ---- Run end-to-end inference ----
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        callbacks=[evo_writer],
        logger=False,
        enable_checkpointing=False,
        precision="bf16-mixed",
    )

    print(f"\nRunning Boltz2 + evolution inference on {len(manifest.records)} inputs...")
    trainer.predict(boltz_model, datamodule=data_module, return_predictions=False)
    print("\nDone.")


if __name__ == "__main__":
    main()
