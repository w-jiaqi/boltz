"""Prepare evolution training data from the HuggingFace evo dataset.

COMMAND 1 of the evolution training pipeline.

Downloads wjiaqi/evo, filters by sequence length, deduplicates complexes,
generates YAML files for Boltz2 caching, and creates train/val pair CSVs.

After this, run cache_evolution_features.py on the structures/ dir to
produce cached .pt files, then train with train_evolution.py.

Usage:
    python scripts/train/prepare_evolution_data.py \
        --output /path/to/evolution_data/ \
        --max_seq_len 300 \
        --val_fraction 0.1

Output:
    evolution_data/
    ├── structures/           # YAML files for Boltz2 (one per unique complex)
    ├── train_pairs.csv       # Bradley-Terry training pairs
    ├── val_pairs.csv         # Validation pairs
    └── stats.txt             # Summary statistics

Next step:
    python cache_evolution_features.py \\
        --data evolution_data/structures/ \\
        --output evolution_data/cache/ \\
        --checkpoint ... --cache ... --use_msa_server --no_kernels
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


def complex_id(uniprot_a: str, uniprot_b: str) -> str:
    """Deterministic complex ID from two UniProt accessions.

    Order matters: (A, B) != (B, A). This captures the directionality
    of the swap — protein A is the "anchor" and B is the "partner."
    """
    return f"{uniprot_a}__{uniprot_b}"


def write_yaml(path: Path, seq_a: str, seq_b: str):
    """Write a two-chain protein complex YAML for boltz predict."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("version: 1\n")
        f.write("sequences:\n")
        f.write("  - protein:\n")
        f.write("      id: A\n")
        f.write(f"      sequence: {seq_a}\n")
        f.write("  - protein:\n")
        f.write("      id: B\n")
        f.write(f"      sequence: {seq_b}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare evolution training data from wjiaqi/evo dataset."
    )
    parser.add_argument("--output", required=True,
                        help="Output directory. Will contain structures/, CSVs, etc.")
    parser.add_argument("--max_seq_len", type=int, default=300,
                        help="Max sequence length per chain. Rows where ANY of the "
                        "4 chains exceeds this are excluded. Default: 300")
    parser.add_argument("--max_total_tokens", type=int, default=None,
                        help="Max sum (seq_A + seq_B) per complex. Pairs whose "
                        "preferred or dispreferred complex exceeds this are "
                        "dropped, and only complexes within budget get YAMLs. "
                        "Set this to match the limit applied downstream so the "
                        "pairs CSV stays consistent with the cache. Default: no limit.")
    parser.add_argument("--val_fraction", type=float, default=0.1,
                        help="Fraction of interaction groups held out for validation. "
                        "Split is by group to prevent protein leakage. Default: 0.1")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for train/val split. Default: 42")
    parser.add_argument("--null_identity", type=float, default=0.0,
                        help="Value for null seq_identity (~5%% of rows where proteins "
                        "are too divergent for MMseqs2 alignment). Default: 0.0")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    structures_dir = out_dir / "structures"
    structures_dir.mkdir(exist_ok=True)

    # ---- Load dataset ----
    print("Loading dataset from HuggingFace (wjiaqi/evo)...")
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: `datasets` package not installed. Run: pip install datasets")
        sys.exit(1)

    ds = load_dataset("wjiaqi/evo", split="train")
    total_rows = len(ds)
    print(f"  Loaded {total_rows} rows")

    # ---- Filter by sequence length ----
    max_len = args.max_seq_len
    print(f"\nFiltering: all 4 chains must be <= {max_len} aa...")

    kept_rows = []
    for row in ds:
        lens = [
            row["protein_A_sp1_len"], row["protein_B_sp1_len"],
            row["protein_A_sp2_len"], row["protein_B_sp2_len"],
        ]
        if all(l <= max_len for l in lens):
            kept_rows.append(row)

    print(f"  Kept {len(kept_rows)} / {total_rows} rows "
          f"({100 * len(kept_rows) / total_rows:.1f}%)")

    if not kept_rows:
        print("ERROR: No rows passed the length filter. Try increasing --max_seq_len.")
        sys.exit(1)

    # ---- Train/val split by interaction_group_id (prevents data leakage) ----
    print(f"\nSplitting by interaction_group_id "
          f"(val_fraction={args.val_fraction})...")

    group_ids = sorted(set(row["interaction_group_id"] for row in kept_rows))
    rng = np.random.default_rng(args.seed)
    rng.shuffle(group_ids)

    n_val = max(1, int(len(group_ids) * args.val_fraction))
    val_groups = set(group_ids[:n_val])
    train_groups = set(group_ids[n_val:])

    train_rows = [r for r in kept_rows if r["interaction_group_id"] in train_groups]
    val_rows = [r for r in kept_rows if r["interaction_group_id"] in val_groups]

    print(f"  Train: {len(train_rows)} rows from {len(train_groups)} groups")
    print(f"  Val:   {len(val_rows)} rows from {len(val_groups)} groups")

    # ---- Extract unique complexes ----
    # Each (uniprot_A, uniprot_B) pair defines one unique complex.
    # A row produces 4 complexes: 2 native + 2 swaps. Many rows share
    # the same native complexes, so deduplication is critical.
    print("\nExtracting unique complexes and building pairs...")

    complexes = {}  # complex_id -> (seq_a, seq_b)

    def register(uniprot_a, seq_a, uniprot_b, seq_b):
        cid = complex_id(uniprot_a, uniprot_b)
        if cid not in complexes:
            complexes[cid] = (seq_a, seq_b)
        return cid

    def build_pairs(rows):
        """From dataset rows, build Bradley-Terry comparison pairs.

        For each row (one conserved interaction across two species):
        - Native 1: (A_sp1, B_sp1) — correct in species 1
        - Native 2: (A_sp2, B_sp2) — correct in species 2
        - Swap 1:   (A_sp1, B_sp2) — cross-species B swap
        - Swap 2:   (A_sp2, B_sp1) — cross-species B swap

        Pairs: native_1 preferred over swap_1,
               native_2 preferred over swap_2.
        Distance = 1 - seq_identity_B (how different the swapped B is).
        """
        pairs = []
        for row in rows:
            sid_b = row["seq_identity_B"]
            if sid_b is None:
                sid_b = args.null_identity
            dist_swap = 1.0 - sid_b

            native_1 = register(
                row["protein_A_sp1_uniprot"], row["protein_A_sp1_seq"],
                row["protein_B_sp1_uniprot"], row["protein_B_sp1_seq"],
            )
            native_2 = register(
                row["protein_A_sp2_uniprot"], row["protein_A_sp2_seq"],
                row["protein_B_sp2_uniprot"], row["protein_B_sp2_seq"],
            )
            swap_1 = register(
                row["protein_A_sp1_uniprot"], row["protein_A_sp1_seq"],
                row["protein_B_sp2_uniprot"], row["protein_B_sp2_seq"],
            )
            swap_2 = register(
                row["protein_A_sp2_uniprot"], row["protein_A_sp2_seq"],
                row["protein_B_sp1_uniprot"], row["protein_B_sp1_seq"],
            )

            pairs.append({
                "preferred": native_1,
                "dispreferred": swap_1,
                "dist_preferred": 0.0,
                "dist_dispreferred": round(dist_swap, 4),
            })
            pairs.append({
                "preferred": native_2,
                "dispreferred": swap_2,
                "dist_preferred": 0.0,
                "dist_dispreferred": round(dist_swap, 4),
            })
        return pairs

    train_pairs = build_pairs(train_rows)
    val_pairs = build_pairs(val_rows)

    print(f"  Unique complexes: {len(complexes)}")
    print(f"  Training pairs:   {len(train_pairs)}")
    print(f"  Validation pairs: {len(val_pairs)}")

    # ---- Apply token-sum filter (must match downstream cache filter) ----
    if args.max_total_tokens is not None:
        valid_cids = {
            cid for cid, (sa, sb) in complexes.items()
            if len(sa) + len(sb) <= args.max_total_tokens
        }
        n_cx_before = len(complexes)
        n_train_before = len(train_pairs)
        n_val_before = len(val_pairs)
        complexes = {cid: c for cid, c in complexes.items() if cid in valid_cids}
        train_pairs = [
            p for p in train_pairs
            if p["preferred"] in valid_cids and p["dispreferred"] in valid_cids
        ]
        val_pairs = [
            p for p in val_pairs
            if p["preferred"] in valid_cids and p["dispreferred"] in valid_cids
        ]
        print(f"\nApplied max_total_tokens={args.max_total_tokens} filter:")
        print(f"  Complexes:        {n_cx_before} -> {len(complexes)}")
        print(f"  Training pairs:   {n_train_before} -> {len(train_pairs)}")
        print(f"  Validation pairs: {n_val_before} -> {len(val_pairs)}")

    if not train_pairs:
        print("ERROR: No training pairs survive filtering.")
        sys.exit(1)

    # ---- Write YAML files ----
    print(f"\nWriting {len(complexes)} YAML files to {structures_dir}/ ...")
    for i, (cid, (seq_a, seq_b)) in enumerate(complexes.items()):
        write_yaml(structures_dir / f"{cid}.yaml", seq_a, seq_b)
        if (i + 1) % 5000 == 0:
            print(f"  {i + 1} / {len(complexes)}")
    print(f"  Done.")

    # ---- Write pairs CSVs ----
    fieldnames = ["preferred", "dispreferred", "dist_preferred", "dist_dispreferred"]

    def write_csv(path, pairs):
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(pairs)

    train_csv = out_dir / "train_pairs.csv"
    val_csv = out_dir / "val_pairs.csv"
    write_csv(train_csv, train_pairs)
    write_csv(val_csv, val_pairs)
    print(f"\n  {train_csv}  ({len(train_pairs)} pairs)")
    print(f"  {val_csv}  ({len(val_pairs)} pairs)")

    # ---- Write statistics ----
    all_lens = [len(s1) + len(s2) for s1, s2 in complexes.values()]
    stats_lines = [
        f"Source: wjiaqi/evo (HuggingFace)",
        f"Total rows in dataset: {total_rows}",
        f"Max sequence length filter: {max_len}",
        f"Rows after filter: {len(kept_rows)} ({100 * len(kept_rows) / total_rows:.1f}%)",
        f"Interaction groups (train): {len(train_groups)}",
        f"Interaction groups (val): {len(val_groups)}",
        f"Training pairs: {len(train_pairs)}",
        f"Validation pairs: {len(val_pairs)}",
        f"Unique complexes: {len(complexes)}",
        f"Total tokens per complex: min={min(all_lens)}, max={max(all_lens)}, "
        f"mean={np.mean(all_lens):.0f}, median={np.median(all_lens):.0f}",
    ]
    stats_str = "\n".join(stats_lines)
    (out_dir / "stats.txt").write_text(stats_str + "\n")

    print(f"\n{'=' * 60}")
    print(stats_str)
    print(f"{'=' * 60}")

    print(f"\nNext steps:")
    print(f"  1. Cache features (GPU):")
    print(f"     python cache_evolution_features.py \\")
    print(f"         --data {structures_dir} \\")
    print(f"         --output {out_dir / 'cache'} \\")
    print(f"         --checkpoint <boltz2_conf.ckpt> --cache <boltz_cache> \\")
    print(f"         --use_msa_server --no_kernels")
    print(f"")
    print(f"  2. Train (GPU):")
    print(f"     python train_evolution.py <config.yaml>")
    print(f"     with data_dir: {out_dir}")


if __name__ == "__main__":
    main()
