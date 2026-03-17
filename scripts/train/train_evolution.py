"""Train the evolution head on cached trunk representations.

This script trains ONLY the EvolutionModule (the evolution head) using
cached trunk outputs. The trunk (pairformer, MSA module, input embedder,
diffusion, confidence) is NOT loaded or run — only the lightweight
evolution head is trained.

Prerequisites:
    1. Cached trunk representations (.pt files) produced by
       cache_evolution_features.py
    2. A pairs CSV defining which complexes to compare and their
       evolutionary distances

Usage:
    python scripts/train/train_evolution.py scripts/train/configs/evolution.yaml

    # With overrides:
    python scripts/train/train_evolution.py scripts/train/configs/evolution.yaml \
        training.lr=1e-3 training.bt_temperature=0.5

    # Debug mode (single device, no wandb, num_workers=0):
    python scripts/train/train_evolution.py scripts/train/configs/evolution.yaml \
        debug=true
"""

import csv
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import omegaconf
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities import rank_zero_only
from torch.utils.data import DataLoader, Dataset

from boltz.model.loss.evolution import (
    bradley_terry_loss,
    evolution_loss,
    margin_ranking_loss,
)
from boltz.model.modules.evolution import EvolutionModule


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

CACHED_FEAT_KEYS = [
    "token_pad_mask",
    "token_to_rep_atom",
    "mol_type",
    "affinity_token_mask",
]


def _load_cached(path: Path) -> dict:
    """Load a cached .pt file and return tensors ready for the evolution module."""
    data = torch.load(path, map_location="cpu", weights_only=False)
    return {
        "s_inputs": data["s_inputs"],
        "z": data["z"],
        "x_pred": data["x_pred"],
        "feats": {k: data[k] for k in CACHED_FEAT_KEYS if k in data},
    }


class PairedEvolutionDataset(Dataset):
    """Dataset that yields pairs of cached representations for BT training.

    Each item returns a dict with 'preferred' and 'dispreferred' cached
    tensors, plus evolutionary distances for both.

    Parameters
    ----------
    cache_dir : str or Path
        Directory containing per-complex .pt files.
    pairs_csv : str or Path
        CSV file with columns:
            preferred      - complex ID (should have lower energy)
            dispreferred   - complex ID (should have higher energy)
            dist_preferred - evolutionary distance of preferred (e.g. 0.0)
            dist_dispreferred - evolutionary distance of dispreferred
    """

    def __init__(self, cache_dir: str, pairs_csv: str):
        self.cache_dir = Path(cache_dir)
        self.pairs = []
        with open(pairs_csv) as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.pairs.append(row)

        missing = set()
        for row in self.pairs:
            for key in ("preferred", "dispreferred"):
                pt = self.cache_dir / f"{row[key]}.pt"
                if not pt.exists():
                    missing.add(str(pt))
        if missing:
            msg = f"Missing {len(missing)} cached files. First 5: {list(missing)[:5]}"
            raise FileNotFoundError(msg)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        row = self.pairs[idx]
        pref = _load_cached(self.cache_dir / f"{row['preferred']}.pt")
        dispref = _load_cached(self.cache_dir / f"{row['dispreferred']}.pt")
        return {
            "preferred": pref,
            "dispreferred": dispref,
            "dist_preferred": float(row["dist_preferred"]),
            "dist_dispreferred": float(row["dist_dispreferred"]),
        }


def _collate_fn(batch):
    """Custom collate for batch_size=1 (no padding needed)."""
    assert len(batch) == 1
    item = batch[0]
    for side in ("preferred", "dispreferred"):
        item[side]["s_inputs"] = item[side]["s_inputs"].unsqueeze(0)
        item[side]["z"] = item[side]["z"].unsqueeze(0)
        if item[side]["x_pred"].dim() == 2:
            item[side]["x_pred"] = item[side]["x_pred"].unsqueeze(0)
        for k in item[side]["feats"]:
            if item[side]["feats"][k].dim() == 1:
                item[side]["feats"][k] = item[side]["feats"][k].unsqueeze(0)
            elif item[side]["feats"][k].dim() == 2:
                item[side]["feats"][k] = item[side]["feats"][k].unsqueeze(0)
    item["dist_preferred"] = torch.tensor(
        [item["dist_preferred"]], dtype=torch.float32
    )
    item["dist_dispreferred"] = torch.tensor(
        [item["dist_dispreferred"]], dtype=torch.float32
    )
    return item


# ---------------------------------------------------------------------------
# Lightning Module
# ---------------------------------------------------------------------------


class EvolutionTrainingModule(pl.LightningModule):
    """Lightning module that trains ONLY the EvolutionModule.

    Each training step:
        1. Forward pass on the preferred complex → E_preferred
        2. Forward pass on the dispreferred complex → E_dispreferred
        3. Compute BT loss + optional margin loss
        4. Backward + optimize

    Parameters
    ----------
    evolution_model_args : dict
        Arguments for EvolutionModule (token_s, token_z, pairformer_args, etc.)
    training_args : dict
        Training hyperparameters (lr, bt_weight, margin_weight, etc.)
    """

    def __init__(
        self,
        evolution_model_args: dict,
        training_args: dict,
    ):
        super().__init__()
        self.save_hyperparameters()

        token_s = evolution_model_args.pop("token_s", 384)
        token_z = evolution_model_args.pop("token_z", 128)
        self.evolution_module = EvolutionModule(
            token_s=token_s,
            token_z=token_z,
            **evolution_model_args,
        )

        self.training_args = training_args

    def forward(self, s_inputs, z, x_pred, feats):
        return self.evolution_module(
            s_inputs=s_inputs.float(),
            z=z.float(),
            x_pred=x_pred.float(),
            feats={k: v.float() if v.is_floating_point() else v for k, v in feats.items()},
            multiplicity=1,
        )

    def _run_pair(self, batch):
        pref = batch["preferred"]
        dispref = batch["dispreferred"]

        device = self.device
        out_pref = self.forward(
            s_inputs=pref["s_inputs"].to(device),
            z=pref["z"].to(device),
            x_pred=pref["x_pred"].to(device),
            feats={k: v.to(device) for k, v in pref["feats"].items()},
        )
        out_dispref = self.forward(
            s_inputs=dispref["s_inputs"].to(device),
            z=dispref["z"].to(device),
            x_pred=dispref["x_pred"].to(device),
            feats={k: v.to(device) for k, v in dispref["feats"].items()},
        )
        return out_pref, out_dispref

    def training_step(self, batch, batch_idx):
        out_pref, out_dispref = self._run_pair(batch)

        loss_dict = evolution_loss(
            energies_preferred=out_pref["evo_energy"],
            energies_dispreferred=out_dispref["evo_energy"],
            distances_preferred=batch["dist_preferred"].to(self.device),
            distances_dispreferred=batch["dist_dispreferred"].to(self.device),
            bt_weight=self.training_args.get("bt_weight", 1.0),
            margin_weight=self.training_args.get("margin_weight", 0.0),
            bt_temperature=self.training_args.get("bt_temperature", 1.0),
            margin_alpha=self.training_args.get("margin_alpha", 1.0),
        )

        self.log("train/loss", loss_dict["loss"], prog_bar=True)
        self.log("train/bt_loss", loss_dict["loss_breakdown"]["bt_loss"])
        self.log("train/margin_loss", loss_dict["loss_breakdown"]["margin_loss"])

        e_pref = out_pref["evo_energy"].detach().mean()
        e_dispref = out_dispref["evo_energy"].detach().mean()
        self.log("train/energy_preferred", e_pref)
        self.log("train/energy_dispreferred", e_dispref)
        self.log("train/energy_gap", e_dispref - e_pref)
        self.log(
            "train/accuracy",
            (e_pref < e_dispref).float(),
        )

        return loss_dict["loss"]

    def validation_step(self, batch, batch_idx):
        out_pref, out_dispref = self._run_pair(batch)

        loss_dict = evolution_loss(
            energies_preferred=out_pref["evo_energy"],
            energies_dispreferred=out_dispref["evo_energy"],
            distances_preferred=batch["dist_preferred"].to(self.device),
            distances_dispreferred=batch["dist_dispreferred"].to(self.device),
            bt_weight=self.training_args.get("bt_weight", 1.0),
            margin_weight=self.training_args.get("margin_weight", 0.0),
            bt_temperature=self.training_args.get("bt_temperature", 1.0),
            margin_alpha=self.training_args.get("margin_alpha", 1.0),
        )

        self.log("val/loss", loss_dict["loss"], prog_bar=True, sync_dist=True)
        self.log("val/bt_loss", loss_dict["loss_breakdown"]["bt_loss"], sync_dist=True)

        e_pref = out_pref["evo_energy"].detach().mean()
        e_dispref = out_dispref["evo_energy"].detach().mean()
        self.log("val/energy_gap", e_dispref - e_pref, sync_dist=True)
        self.log(
            "val/accuracy",
            (e_pref < e_dispref).float(),
            sync_dist=True,
        )
        return loss_dict["loss"]

    def configure_optimizers(self):
        lr = self.training_args.get("lr", 1.8e-3)
        weight_decay = self.training_args.get("weight_decay", 0.0)
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=lr,
            betas=(
                self.training_args.get("adam_beta_1", 0.9),
                self.training_args.get("adam_beta_2", 0.95),
            ),
            eps=self.training_args.get("adam_eps", 1e-8),
            weight_decay=weight_decay,
        )

        scheduler_type = self.training_args.get("lr_scheduler", None)
        if scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.training_args.get("lr_cosine_T_max", 50000),
                eta_min=self.training_args.get("lr_min", 1e-6),
            )
            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
        elif scheduler_type == "linear_warmup_cosine":
            from torch.optim.lr_scheduler import (
                CosineAnnealingLR,
                LinearLR,
                SequentialLR,
            )

            warmup = LinearLR(
                optimizer,
                start_factor=1e-3,
                total_iters=self.training_args.get("lr_warmup_steps", 1000),
            )
            cosine = CosineAnnealingLR(
                optimizer,
                T_max=self.training_args.get("lr_cosine_T_max", 50000),
                eta_min=self.training_args.get("lr_min", 1e-6),
            )
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup, cosine],
                milestones=[self.training_args.get("lr_warmup_steps", 1000)],
            )
            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

        return optimizer


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@dataclass
class TrainEvolutionConfig:
    cache_dir: str
    train_pairs_csv: str
    val_pairs_csv: Optional[str] = None
    output: str = "./output_evolution"
    evolution_model_args: dict = field(default_factory=dict)
    training: dict = field(default_factory=dict)
    trainer: dict = field(default_factory=dict)
    wandb: Optional[dict] = None
    pretrained: Optional[str] = None
    resume: Optional[str] = None
    debug: bool = False
    num_workers: int = 4
    save_top_k: int = 3


def train(raw_config_path: str, args: list[str]) -> None:
    """Run evolution head training."""
    raw_config = omegaconf.OmegaConf.load(raw_config_path)
    if args:
        overrides = omegaconf.OmegaConf.from_dotlist(args)
        raw_config = omegaconf.OmegaConf.merge(raw_config, overrides)

    cfg = omegaconf.OmegaConf.to_container(raw_config, resolve=True)
    cfg = TrainEvolutionConfig(**cfg)

    # ---------- Datasets ----------
    num_workers = 0 if cfg.debug else cfg.num_workers

    train_ds = PairedEvolutionDataset(cfg.cache_dir, cfg.train_pairs_csv)
    train_loader = DataLoader(
        train_ds,
        batch_size=1,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=_collate_fn,
        pin_memory=True,
    )

    val_loader = None
    if cfg.val_pairs_csv:
        val_ds = PairedEvolutionDataset(cfg.cache_dir, cfg.val_pairs_csv)
        val_loader = DataLoader(
            val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=_collate_fn,
            pin_memory=True,
        )

    # ---------- Model ----------
    model = EvolutionTrainingModule(
        evolution_model_args=dict(cfg.evolution_model_args),
        training_args=dict(cfg.training),
    )

    if cfg.pretrained and not cfg.resume:
        print(f"Loading pretrained evolution weights from {cfg.pretrained}")
        ckpt = torch.load(cfg.pretrained, map_location="cpu", weights_only=False)
        state = ckpt.get("state_dict", ckpt)
        model.load_state_dict(state, strict=False)

    # ---------- Callbacks ----------
    callbacks = []
    mc = ModelCheckpoint(
        dirpath=os.path.join(cfg.output, "checkpoints"),
        filename="evolution-{epoch:03d}-{step}-{val/loss:.4f}",
        monitor="val/loss" if val_loader else "train/loss",
        save_top_k=cfg.save_top_k,
        save_last=True,
        mode="min",
        every_n_epochs=1,
    )
    callbacks.append(mc)

    # ---------- Logger ----------
    loggers = []
    wandb_cfg = cfg.wandb if not cfg.debug else None
    if wandb_cfg:
        wdb_logger = WandbLogger(
            name=wandb_cfg.get("name", "evolution"),
            save_dir=cfg.output,
            project=wandb_cfg.get("project", "boltz-evolution"),
            entity=wandb_cfg.get("entity", None),
            log_model=False,
        )
        loggers.append(wdb_logger)

        @rank_zero_only
        def save_config():
            config_out = Path(wdb_logger.experiment.dir) / "evolution_config.yaml"
            omegaconf.OmegaConf.save(raw_config, config_out)
            wdb_logger.experiment.save(str(config_out))

        save_config()

    # ---------- Trainer ----------
    trainer_kwargs = dict(cfg.trainer)
    devices = trainer_kwargs.pop("devices", 1)
    if cfg.debug:
        devices = 1

    strategy = "auto"
    if isinstance(devices, int) and devices > 1:
        strategy = DDPStrategy(find_unused_parameters=False)
    elif isinstance(devices, list) and len(devices) > 1:
        strategy = DDPStrategy(find_unused_parameters=False)

    trainer = pl.Trainer(
        default_root_dir=cfg.output,
        devices=devices,
        strategy=strategy,
        callbacks=callbacks,
        logger=loggers,
        enable_checkpointing=True,
        **trainer_kwargs,
    )

    # ---------- Train ----------
    trainer.fit(
        model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=cfg.resume,
    )


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python train_evolution.py <config.yaml> [overrides...]")
        sys.exit(1)
    train(sys.argv[1], sys.argv[2:])
