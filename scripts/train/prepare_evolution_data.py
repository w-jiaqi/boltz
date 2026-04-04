"""Prepare evolution training data from the HuggingFace evo dataset.

Downloads the wjiaqi/evo dataset, filters by sequence length, deduplicates
complexes, generates YAML files for Boltz2 caching, creates train/val
pairs CSVs, and prints statistics.

Usage:
    python scripts/train/prepare_evolution_data.py \
        --output /path/to/prepared_data/ \
        --max_seq_len 300 \
        --val_fraction 0.1

Output structure:
    /path/to/prepared_data/
    ├── structures/           # YAML files for each unique complex
    │   ├── P04637__Q00987.yaml
    │   ├── P04637__P23804.yaml   (swap)
    │   └── ...
    ├── train_pairs.csv       # Training pairs
    ├── val_pairs.csv         # Validation pairs
    └── stats.txt             # Summary statistics
"""

import argparse
import csv
import hashlib
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def complex_id(uniprot_a: str, uniprot_b: str) -> str:
    """Deterministic complex ID from two UniProt accessions."""
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
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--max_seq_len", type=int, default=300,
                        help="Max sequence length per chain. Rows where any chain "
                        "exceeds this are excluded. Default: 300")
    parser.add_argument("--val_fraction", type=float, default=0.1,
                        help="Fraction of interaction groups for validation. Default: 0.1")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split")
    parser.add_argument("--null_identity", type=float, default=0.0,
                        help="Value to use for null seq_identity. Default: 0.0 (maximally diverged)")
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
    print(f"  Loaded {len(ds)} rows")

    # ---- Filter by sequence length ----
    max_len = args.max_seq_len
    print(f"\nFiltering rows where all 4 chains <= {max_len} aa...")

    kept_rows = []
    for row in ds:
        lens = [
            row["protein_A_sp1_len"],
            row["protein_B_sp1_len"],
            row["protein_A_sp2_len"],
            row["protein_B_sp2_len"],
        ]
        if all(l <= max_len for l in lens):
            kept_rows.append(row)

    print(f"  Kept {len(kept_rows)} / {len(ds)} rows ({100*len(kept_rows)/len(ds):.1f}%)")

    if not kept_rows:
        print("ERROR: No rows passed the length filter. Try increasing --max_seq_len.")
        sys.exit(1)

    # ---- Train/val split by interaction_group_id ----
    print(f"\nSplitting by interaction_group_id (val_fraction={args.val_fraction})...")

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

    # ---- Extract unique complexes and generate YAMLs ----
    print("\nExtracting unique complexes...")

    # Map: complex_id -> (seq_a, seq_b)
    complexes = {}

    def register_complex(uniprot_a, seq_a, uniprot_b, seq_b):
        cid = complex_id(uniprot_a, uniprot_b)
        if cid not in complexes:
            complexes[cid] = (seq_a, seq_b)
        return cid

    def process_rows(rows):
        """Extract pairs from rows. Returns list of pair dicts."""
        pairs = []
        for row in rows:
            # Handle null sequence identities
            sid_a = row["seq_identity_A"]
            sid_b = row["seq_identity_B"]
            if sid_a is None:
                sid_a = args.null_identity
            if sid_b is None:
                sid_b = args.null_identity

            # Register all 4 complexes
            native_1 = register_complex(
                row["protein_A_sp1_uniprot"], row["protein_A_sp1_seq"],
                row["protein_B_sp1_uniprot"], row["protein_B_sp1_seq"],
            )
            native_2 = register_complex(
                row["protein_A_sp2_uniprot"], row["protein_A_sp2_seq"],
                row["protein_B_sp2_uniprot"], row["protein_B_sp2_seq"],
            )
            swap_1 = register_complex(
                row["protein_A_sp1_uniprot"], row["protein_A_sp1_seq"],
                row["protein_B_sp2_uniprot"], row["protein_B_sp2_seq"],
            )
            swap_2 = register_complex(
                row["protein_A_sp2_uniprot"], row["protein_A_sp2_seq"],
                row["protein_B_sp1_uniprot"], row["protein_B_sp1_seq"],
            )

            # Distance for the swap = how different the swapped B chain is
            dist_swap = 1.0 - sid_b

            # Pair 1: native_1 preferred over swap_1 (A_sp1 kept, B swapped)
            pairs.append({
                "preferred": native_1,
                "dispreferred": swap_1,
                "dist_preferred": 0.0,
                "dist_dispreferred": dist_swap,
            })
            # Pair 2: native_2 preferred over swap_2 (A_sp2 kept, B swapped)
            pairs.append({
                "preferred": native_2,
                "dispreferred": swap_2,
                "dist_preferred": 0.0,
                "dist_dispreferred": dist_swap,
            })

        return pairs

    train_pairs = process_rows(train_rows)
    val_pairs = process_rows(val_rows)

    print(f"  Unique complexes: {len(complexes)}")
    print(f"  Training pairs:   {len(train_pairs)}")
    print(f"  Validation pairs: {len(val_pairs)}")

    # ---- Write YAML files ----
    print(f"\nWriting {len(complexes)} YAML files to {structures_dir}/...")

    for cid, (seq_a, seq_b) in complexes.items():
        write_yaml(structures_dir / f"{cid}.yaml", seq_a, seq_b)

    print(f"  Done.")

    # ---- Write pairs CSVs ----
    def write_pairs_csv(path, pairs):
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "preferred", "dispreferred", "dist_preferred", "dist_dispreferred"
            ])
            writer.writeheader()
            writer.writerows(pairs)

    train_csv = out_dir / "train_pairs.csv"
    write_pairs_csv(train_csv, train_pairs)
    print(f"  Wrote {train_csv} ({len(train_pairs)} pairs)")

    val_csv = out_dir / "val_pairs.csv"
    write_pairs_csv(val_csv, val_pairs)
    print(f"  Wrote {val_csv} ({len(val_pairs)} pairs)")

    # ---- Statistics ----
    stats = []
    stats.append(f"Dataset: wjiaqi/evo")
    stats.append(f"Total rows: {len(ds)}")
    stats.append(f"Max sequence length filter: {max_len}")
    stats.append(f"Rows after filter: {len(kept_rows)} ({100*len(kept_rows)/len(ds):.1f}%)")
    stats.append(f"Interaction groups (train): {len(train_groups)}")
    stats.append(f"Interaction groups (val): {len(val_groups)}")
    stats.append(f"Training pairs: {len(train_pairs)}")
    stats.append(f"Validation pairs: {len(val_pairs)}")
    stats.append(f"Unique complexes to cache: {len(complexes)}")

    # Sequence length distribution of kept complexes
    all_lens = []
    for seq_a, seq_b in complexes.values():
        all_lens.append(len(seq_a) + len(seq_b))
    stats.append(f"Total tokens per complex: min={min(all_lens)}, max={max(all_lens)}, "
                 f"mean={np.mean(all_lens):.0f}, median={np.median(all_lens):.0f}")

    stats_str = "\n".join(stats)
    print(f"\n{'='*50}")
    print(stats_str)
    print(f"{'='*50}")

    with open(out_dir / "stats.txt", "w") as f:
        f.write(stats_str + "\n")

    print(f"\nNext steps:")
    print(f"  1. Cache:  python cache_evolution_features.py --data {structures_dir} --output <cache_dir> ...")
    print(f"  2. Train:  python train_evolution.py <config.yaml>")
    print(f"     Set cache_dir=<cache_dir>, train_pairs_csv={train_csv}, val_pairs_csv={val_csv}")


if __name__ == "__main__":
    main()
