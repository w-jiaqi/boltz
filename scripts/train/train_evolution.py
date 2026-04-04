"""Train the evolution head on cached trunk representations.

COMMAND 2 of the evolution training pipeline.

Trains ONLY the EvolutionModule using cached trunk outputs (.pt files).
The trunk is NOT loaded — only the lightweight evolution head is trained.

Prerequisites:
    1. Run prepare_evolution_data.py to create structures/ and pairs CSVs
    2. Run cache_evolution_features.py to create cache/ with .pt files

Usage:
    python scripts/train/train_evolution.py scripts/train/configs/evolution.yaml

    # Override any config value:
    python scripts/train/train_evolution.py scripts/train/configs/evolution.yaml \\
        training.lr=1e-3 debug=true

Data directory layout (produced by prepare_evolution_data.py):
    data_dir/
    ├── cache/                # .pt files from cache_evolution_features.py
    ├── train_pairs.csv       # Bradley-Terry training pairs
    └── val_pairs.csv         # Validation pairs (optional)
"""

import csv
import os
import sys
from dataclasses import dataclass, field
from functools import lru_cache
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

from boltz.model.loss.evolution import evolution_loss
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


class CachedComplexStore:
    """In-memory LRU cache for loaded .pt files.

    Many training pairs reference the same complex (e.g. a native complex
    appears in dozens of pairs). This avoids re-reading from disk.
    """

    def __init__(self, cache_dir: str, max_in_memory: int = 2000):
        self.cache_dir = Path(cache_dir)
        self._load = lru_cache(maxsize=max_in_memory)(self._load_impl)

    def _load_impl(self, complex_id: str) -> dict:
        path = self.cache_dir / f"{complex_id}.pt"
        data = torch.load(path, map_location="cpu", weights_only=False)
        return {
            "s_inputs": data["s_inputs"],
            "z": data["z"],
            "x_pred": data["x_pred"],
            "feats": {k: data[k] for k in CACHED_FEAT_KEYS if k in data},
        }

    def get(self, complex_id: str) -> dict:
        """Return a deep copy so collate/unsqueeze doesn't mutate cached tensors."""
        cached = self._load(complex_id)
        return {
            "s_inputs": cached["s_inputs"].clone(),
            "z": cached["z"].clone(),
            "x_pred": cached["x_pred"].clone(),
            "feats": {k: v.clone() for k, v in cached["feats"].items()},
        }

    def validate(self, complex_ids: set) -> list[str]:
        """Return list of IDs that are missing from cache."""
        return [cid for cid in complex_ids if not (self.cache_dir / f"{cid}.pt").exists()]


class PairedEvolutionDataset(Dataset):
    """Dataset of Bradley-Terry comparison pairs.

    Each item yields one (preferred, dispreferred) pair loaded from
    the shared CachedComplexStore, plus evolutionary distances.
    """

    def __init__(self, store: CachedComplexStore, pairs_csv: str):
        self.store = store
        self.pairs = []
        with open(pairs_csv) as f:
            for row in csv.DictReader(f):
                self.pairs.append(row)

        referenced_ids = set()
        for row in self.pairs:
            referenced_ids.add(row["preferred"])
            referenced_ids.add(row["dispreferred"])
        missing = self.store.validate(referenced_ids)
        if missing:
            n = len(missing)
            examples = missing[:5]
            raise FileNotFoundError(
                f"{n} cached .pt files missing. First 5: {examples}\n"
                f"Run cache_evolution_features.py first."
            )

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        row = self.pairs[idx]
        return {
            "preferred": self.store.get(row["preferred"]),
            "dispreferred": self.store.get(row["dispreferred"]),
            "dist_preferred": float(row["dist_preferred"]),
            "dist_dispreferred": float(row["dist_dispreferred"]),
        }


def _collate_fn(batch):
    """Collate for batch_size=1. Adds batch dimension to all tensors."""
    assert len(batch) == 1
    item = batch[0]
    for side in ("preferred", "dispreferred"):
        for k in ("s_inputs", "z", "x_pred"):
            item[side][k] = item[side][k].unsqueeze(0)
        for k in item[side]["feats"]:
            item[side]["feats"][k] = item[side]["feats"][k].unsqueeze(0)
    item["dist_preferred"] = torch.tensor([item["dist_preferred"]], dtype=torch.float32)
    item["dist_dispreferred"] = torch.tensor([item["dist_dispreferred"]], dtype=torch.float32)
    return item


# ---------------------------------------------------------------------------
# Lightning Module
# ---------------------------------------------------------------------------


class EvolutionTrainingModule(pl.LightningModule):
    """Lightning module that trains the EvolutionModule.

    Each step runs two forward passes (preferred + dispreferred complex)
    and computes Bradley-Terry + optional margin ranking loss.
    """

    def __init__(self, evolution_model_args: dict, training_args: dict):
        super().__init__()
        self.save_hyperparameters()

        token_s = evolution_model_args.pop("token_s", 384)
        token_z = evolution_model_args.pop("token_z", 128)
        self.evolution_module = EvolutionModule(
            token_s=token_s, token_z=token_z, **evolution_model_args,
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
        device = self.device
        results = []
        for side in ("preferred", "dispreferred"):
            d = batch[side]
            results.append(self.forward(
                s_inputs=d["s_inputs"].to(device),
                z=d["z"].to(device),
                x_pred=d["x_pred"].to(device),
                feats={k: v.to(device) for k, v in d["feats"].items()},
            ))
        return results[0], results[1]

    def _compute_loss(self, batch):
        out_pref, out_dispref = self._run_pair(batch)
        return evolution_loss(
            energies_preferred=out_pref["evo_energy"],
            energies_dispreferred=out_dispref["evo_energy"],
            distances_preferred=batch["dist_preferred"].to(self.device),
            distances_dispreferred=batch["dist_dispreferred"].to(self.device),
            bt_weight=self.training_args.get("bt_weight", 1.0),
            margin_weight=self.training_args.get("margin_weight", 0.0),
            bt_temperature=self.training_args.get("bt_temperature", 1.0),
            margin_alpha=self.training_args.get("margin_alpha", 1.0),
        ), out_pref, out_dispref

    def training_step(self, batch, batch_idx):
        loss_dict, out_pref, out_dispref = self._compute_loss(batch)

        self.log("train/loss", loss_dict["loss"], prog_bar=True)
        self.log("train/bt_loss", loss_dict["loss_breakdown"]["bt_loss"])
        self.log("train/margin_loss", loss_dict["loss_breakdown"]["margin_loss"])

        e_pref = out_pref["evo_energy"].detach().mean()
        e_dispref = out_dispref["evo_energy"].detach().mean()
        self.log("train/energy_gap", e_dispref - e_pref)
        self.log("train/accuracy", (e_pref < e_dispref).float())

        return loss_dict["loss"]

    def validation_step(self, batch, batch_idx):
        loss_dict, out_pref, out_dispref = self._compute_loss(batch)

        self.log("val/loss", loss_dict["loss"], prog_bar=True, sync_dist=True)
        self.log("val/bt_loss", loss_dict["loss_breakdown"]["bt_loss"], sync_dist=True)

        e_pref = out_pref["evo_energy"].detach().mean()
        e_dispref = out_dispref["evo_energy"].detach().mean()
        self.log("val/energy_gap", e_dispref - e_pref, sync_dist=True)
        self.log("val/accuracy", (e_pref < e_dispref).float(), sync_dist=True)

        return loss_dict["loss"]

    def configure_optimizers(self):
        lr = self.training_args.get("lr", 1.8e-3)
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=lr,
            betas=(
                self.training_args.get("adam_beta_1", 0.9),
                self.training_args.get("adam_beta_2", 0.95),
            ),
            eps=self.training_args.get("adam_eps", 1e-8),
            weight_decay=self.training_args.get("weight_decay", 0.0),
        )

        sched_type = self.training_args.get("lr_scheduler", None)
        if sched_type == "cosine":
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.training_args.get("lr_cosine_T_max", 50000),
                eta_min=self.training_args.get("lr_min", 1e-6),
            )
            return [optimizer], [{"scheduler": sched, "interval": "step"}]
        elif sched_type == "linear_warmup_cosine":
            from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
            warmup_steps = self.training_args.get("lr_warmup_steps", 1000)
            warmup = LinearLR(optimizer, start_factor=1e-3, total_iters=warmup_steps)
            cosine = CosineAnnealingLR(
                optimizer,
                T_max=self.training_args.get("lr_cosine_T_max", 50000),
                eta_min=self.training_args.get("lr_min", 1e-6),
            )
            sched = SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_steps])
            return [optimizer], [{"scheduler": sched, "interval": "step"}]

        return optimizer


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@dataclass
class TrainEvolutionConfig:
    data_dir: str
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
    max_cached_in_memory: int = 2000


def train(raw_config_path: str, args: list[str]) -> None:
    """Run evolution head training."""
    raw_config = omegaconf.OmegaConf.load(raw_config_path)
    if args:
        overrides = omegaconf.OmegaConf.from_dotlist(args)
        raw_config = omegaconf.OmegaConf.merge(raw_config, overrides)

    cfg = omegaconf.OmegaConf.to_container(raw_config, resolve=True)
    cfg = TrainEvolutionConfig(**cfg)

    data_dir = Path(cfg.data_dir)

    # Auto-discover paths within data_dir
    cache_dir = data_dir / "cache"
    train_csv = data_dir / "train_pairs.csv"
    val_csv = data_dir / "val_pairs.csv"

    if not cache_dir.exists():
        print(f"ERROR: {cache_dir} not found. Run cache_evolution_features.py first.")
        sys.exit(1)
    if not train_csv.exists():
        print(f"ERROR: {train_csv} not found. Run prepare_evolution_data.py first.")
        sys.exit(1)

    # ---------- Datasets ----------
    num_workers = 0 if cfg.debug else cfg.num_workers

    store = CachedComplexStore(
        str(cache_dir),
        max_in_memory=cfg.max_cached_in_memory,
    )

    train_ds = PairedEvolutionDataset(store, str(train_csv))
    train_loader = DataLoader(
        train_ds, batch_size=1, shuffle=True,
        num_workers=num_workers, collate_fn=_collate_fn, pin_memory=True,
    )

    val_loader = None
    if val_csv.exists():
        val_ds = PairedEvolutionDataset(store, str(val_csv))
        val_loader = DataLoader(
            val_ds, batch_size=1, shuffle=False,
            num_workers=num_workers, collate_fn=_collate_fn, pin_memory=True,
        )

    print(f"Data dir:        {data_dir}")
    print(f"Cache dir:       {cache_dir}")
    print(f"Train pairs:     {len(train_ds)}")
    print(f"Val pairs:       {len(val_ds) if val_loader else 0}")
    print(f"Cached in memory: up to {cfg.max_cached_in_memory}")

    # ---------- Model ----------
    model = EvolutionTrainingModule(
        evolution_model_args=dict(cfg.evolution_model_args),
        training_args=dict(cfg.training),
    )

    if cfg.pretrained and not cfg.resume:
        print(f"Loading pretrained weights from {cfg.pretrained}")
        ckpt = torch.load(cfg.pretrained, map_location="cpu", weights_only=False)
        state = ckpt.get("state_dict", ckpt)
        model.load_state_dict(state, strict=False)

    # ---------- Callbacks ----------
    mc = ModelCheckpoint(
        dirpath=os.path.join(cfg.output, "checkpoints"),
        filename="evolution-{epoch:03d}-{step}",
        monitor="val/loss" if val_loader else "train/loss",
        save_top_k=cfg.save_top_k, save_last=True, mode="min",
        every_n_epochs=1,
    )

    # ---------- Logger ----------
    loggers = []
    wandb_cfg = cfg.wandb if not cfg.debug else None
    if wandb_cfg:
        wdb = WandbLogger(
            name=wandb_cfg.get("name", "evolution"),
            save_dir=cfg.output,
            project=wandb_cfg.get("project", "boltz-evolution"),
            entity=wandb_cfg.get("entity", None),
            log_model=False,
        )
        loggers.append(wdb)

        @rank_zero_only
        def _save_cfg():
            p = Path(wdb.experiment.dir) / "evolution_config.yaml"
            omegaconf.OmegaConf.save(raw_config, p)
            wdb.experiment.save(str(p))
        _save_cfg()

    # ---------- Trainer ----------
    trainer_kwargs = dict(cfg.trainer)
    devices = trainer_kwargs.pop("devices", 1)
    if cfg.debug:
        devices = 1

    strategy = "auto"
    if isinstance(devices, (int, list)) and (
        (isinstance(devices, int) and devices > 1) or
        (isinstance(devices, list) and len(devices) > 1)
    ):
        strategy = DDPStrategy(find_unused_parameters=False)

    trainer = pl.Trainer(
        default_root_dir=cfg.output,
        devices=devices, strategy=strategy,
        callbacks=[mc], logger=loggers,
        enable_checkpointing=True,
        **trainer_kwargs,
    )

    # ---------- Train ----------
    trainer.fit(model, train_dataloaders=train_loader,
                val_dataloaders=val_loader, ckpt_path=cfg.resume)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python train_evolution.py <config.yaml> [overrides...]")
        sys.exit(1)
    train(sys.argv[1], sys.argv[2:])
