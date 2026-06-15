"""Prepare evolution training data from the HuggingFace evo-final dataset.

COMMAND 1 of the evolution training pipeline.

Downloads wjiaqi/evo-final (subset `swap_complexes`), filters by sequence
length, deduplicates complexes, generates YAML files for Boltz2 caching, and
creates train/val/test pair CSVs.

After this, run cache_evolution_features.py on the structures/ dir to
produce cached .pt files, then train with train_evolution.py.

Dataset layout (wjiaqi/evo-final, subset `swap_complexes`)
----------------------------------------------------------
Three pre-built splits: ``train`` / ``validation`` / ``test`` (already
separated by ``interaction_group_id`` so there is no protein leakage — we
use them as-is rather than re-splitting). Each row describes one conserved
interaction observed in two species and carries, among others:
    protein_{A,B}_sp{1,2}_{uniprot,seq,len}
    seq_identity_A, seq_identity_B   (identity of the orthologous chain
                                      between the two species)
    divergence_mya                   (phylogenetic distance between the two
                                      species, in millions of years)

Pair construction (4 preference pairs per row)
----------------------------------------------
The two native complexes are (A_sp1, B_sp1) and (A_sp2, B_sp2). Mixing the
chains across species yields the two swap complexes (A_sp1, B_sp2) and
(A_sp2, B_sp1). From these 4 complexes we emit 4 native-vs-swap pairs — two
that swap chain B and two that swap chain A — mirroring the AB/BA crossovers
in the dataset's own `preference_pairs` subset (which has exactly 4x the rows
of `swap_complexes`).

Distance label
--------------
The dispreferred (swap) complex is given an evolutionary-distance label that
is a weighted blend of two normalized signals. We DO NOT bake the weights in
here — instead we emit the two normalized components per pair and let
train_evolution.py combine them with config-tunable weights:

    div_norm = log1p(divergence_mya) / log1p(D_max)          in [0, 1]
    seq_norm = minmax(1 - seq_identity_<swapped chain>)       in [0, 1]

    dist_dispreferred = w_div * div_norm + w_seq * seq_norm   (computed at
                                                               train time)
    dist_preferred    = 0.0                                   (native anchor)

Normalization constants (D_max, the seq min/max) are fit on the TRAIN split
only and reused for val/test so there is no leakage. They are recorded in
stats.txt and norm_params.json for reproducibility.

Usage:
    python scripts/train/prepare_evolution_data.py \
        --output /path/to/evolution_data/ \
        --max_seq_len 300 \
        --max_total_tokens 400

Output:
    evolution_data/
    ├── structures/           # YAML files for Boltz2 (one per unique complex)
    ├── train_pairs.csv       # preference pairs (train split)
    ├── val_pairs.csv         # preference pairs (validation split)
    ├── test_pairs.csv        # preference pairs (test split)
    ├── norm_params.json      # distance-normalization constants
    └── stats.txt             # Summary statistics

Pair CSV columns:
    preferred, dispreferred, swapped_chain, div_norm, seq_norm

Next step:
    python cache_evolution_features.py \\
        --data evolution_data/structures/ \\
        --output evolution_data/cache/ \\
        --checkpoint ... --cache ... --use_msa_server --no_kernels
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

DATASET = "wjiaqi/evo-final"
SUBSET = "swap_complexes"
# HuggingFace split name -> output CSV stem.
SPLITS = {"train": "train_pairs", "validation": "val_pairs", "test": "test_pairs"}


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
        description=f"Prepare evolution training data from {DATASET} ({SUBSET})."
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
    parser.add_argument("--null_identity", type=float, default=1.0,
                        help="Identity value to assume when seq_identity_A/B is "
                        "missing (treated as identical => no sequence signal). "
                        "The new dataset has no nulls; this is a safety net. "
                        "Default: 1.0")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    structures_dir = out_dir / "structures"
    structures_dir.mkdir(exist_ok=True)

    # ---- Load dataset (all three pre-built splits) ----
    print(f"Loading dataset from HuggingFace ({DATASET}, subset={SUBSET})...")
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: `datasets` package not installed. Run: pip install datasets")
        sys.exit(1)

    ds = load_dataset(DATASET, SUBSET)
    missing_splits = [s for s in SPLITS if s not in ds]
    if missing_splits:
        print(f"ERROR: dataset is missing expected splits {missing_splits}. "
              f"Available: {list(ds.keys())}")
        sys.exit(1)

    total_rows = {s: len(ds[s]) for s in SPLITS}
    print(f"  Loaded splits: " + ", ".join(f"{s}={n}" for s, n in total_rows.items()))

    # ---- Filter by sequence length (per split) ----
    max_len = args.max_seq_len
    print(f"\nFiltering: all 4 chains must be <= {max_len} aa...")

    def length_ok(row):
        lens = [
            row["protein_A_sp1_len"], row["protein_B_sp1_len"],
            row["protein_A_sp2_len"], row["protein_B_sp2_len"],
        ]
        return all(l <= max_len for l in lens)

    kept = {}
    for split in SPLITS:
        kept[split] = [r for r in ds[split] if length_ok(r)]
        n, tot = len(kept[split]), total_rows[split]
        print(f"  {split:11s}: kept {n} / {tot} ({100 * n / tot:.1f}%)")

    if not kept["train"]:
        print("ERROR: No train rows passed the length filter. "
              "Try increasing --max_seq_len.")
        sys.exit(1)

    # ---- Fit distance-normalization constants on the TRAIN split only ----
    # div_norm = log1p(divergence_mya) / log1p(D_max)
    # seq_norm = (x - seq_min) / (seq_max - seq_min), x = 1 - seq_identity_<chain>
    def _seqid(v):
        return args.null_identity if v is None else v

    train_div = [r["divergence_mya"] for r in kept["train"]
                 if r["divergence_mya"] is not None]
    d_max = max(train_div) if train_div else 1.0
    log_d_max = math.log1p(d_max) if d_max > 0 else 1.0

    train_seq_dist = []
    for r in kept["train"]:
        train_seq_dist.append(1.0 - _seqid(r["seq_identity_A"]))
        train_seq_dist.append(1.0 - _seqid(r["seq_identity_B"]))
    seq_min = min(train_seq_dist)
    seq_max = max(train_seq_dist)
    seq_span = (seq_max - seq_min) if seq_max > seq_min else 1.0

    norm_params = {
        "div": {"transform": "log1p_minmax", "d_max": float(d_max),
                "log1p_d_max": float(log_d_max)},
        "seq": {"transform": "minmax_of_1_minus_identity",
                "seq_dist_min": float(seq_min), "seq_dist_max": float(seq_max)},
        "fit_on": "train",
    }
    (out_dir / "norm_params.json").write_text(json.dumps(norm_params, indent=2) + "\n")
    print(f"\nDistance normalization (fit on train):")
    print(f"  divergence: log1p / log1p({d_max:.1f})")
    print(f"  seq (1-identity): minmax over [{seq_min:.4f}, {seq_max:.4f}]")

    def div_norm(divergence):
        if divergence is None or log_d_max <= 0:
            return 0.0
        return round(math.log1p(divergence) / log_d_max, 6)

    def seq_norm(identity):
        x = 1.0 - _seqid(identity)
        return round(min(max((x - seq_min) / seq_span, 0.0), 1.0), 6)

    # ---- Build complexes + preference pairs per split ----
    print("\nExtracting unique complexes and building pairs...")

    complexes = {}  # complex_id -> (seq_a, seq_b)

    def register(uniprot_a, seq_a, uniprot_b, seq_b):
        cid = complex_id(uniprot_a, uniprot_b)
        if cid not in complexes:
            complexes[cid] = (seq_a, seq_b)
        return cid

    def build_pairs(rows):
        """Build 4 native-vs-swap preference pairs per row.

        Native complexes : nat1=(A_sp1,B_sp1), nat2=(A_sp2,B_sp2)
        Swap complexes   : mix1=(A_sp1,B_sp2), mix2=(A_sp2,B_sp1)

        Pairs (preferred=native, dispreferred=swap):
          nat1 vs mix1  -> chain B swapped  (seq_identity_B)
          nat2 vs mix2  -> chain B swapped  (seq_identity_B)
          nat1 vs mix2  -> chain A swapped  (seq_identity_A)
          nat2 vs mix1  -> chain A swapped  (seq_identity_A)

        divergence_mya is per-row (species-pair level) and shared by all 4.
        """
        pairs = []
        for row in rows:
            dn = div_norm(row["divergence_mya"])
            sn_a = seq_norm(row["seq_identity_A"])
            sn_b = seq_norm(row["seq_identity_B"])

            nat1 = register(
                row["protein_A_sp1_uniprot"], row["protein_A_sp1_seq"],
                row["protein_B_sp1_uniprot"], row["protein_B_sp1_seq"],
            )
            nat2 = register(
                row["protein_A_sp2_uniprot"], row["protein_A_sp2_seq"],
                row["protein_B_sp2_uniprot"], row["protein_B_sp2_seq"],
            )
            mix1 = register(
                row["protein_A_sp1_uniprot"], row["protein_A_sp1_seq"],
                row["protein_B_sp2_uniprot"], row["protein_B_sp2_seq"],
            )
            mix2 = register(
                row["protein_A_sp2_uniprot"], row["protein_A_sp2_seq"],
                row["protein_B_sp1_uniprot"], row["protein_B_sp1_seq"],
            )

            # Chain-B swaps (anchor A fixed): label uses seq_identity_B.
            pairs.append({"preferred": nat1, "dispreferred": mix1,
                          "swapped_chain": "B", "div_norm": dn, "seq_norm": sn_b})
            pairs.append({"preferred": nat2, "dispreferred": mix2,
                          "swapped_chain": "B", "div_norm": dn, "seq_norm": sn_b})
            # Chain-A swaps (anchor B fixed): label uses seq_identity_A.
            pairs.append({"preferred": nat1, "dispreferred": mix2,
                          "swapped_chain": "A", "div_norm": dn, "seq_norm": sn_a})
            pairs.append({"preferred": nat2, "dispreferred": mix1,
                          "swapped_chain": "A", "div_norm": dn, "seq_norm": sn_a})
        return pairs

    pairs_by_split = {s: build_pairs(kept[s]) for s in SPLITS}

    print(f"  Unique complexes: {len(complexes)}")
    for s in SPLITS:
        print(f"  {s:11s} pairs: {len(pairs_by_split[s])}")

    # ---- Apply token-sum filter (must match downstream cache filter) ----
    if args.max_total_tokens is not None:
        valid_cids = {
            cid for cid, (sa, sb) in complexes.items()
            if len(sa) + len(sb) <= args.max_total_tokens
        }
        n_cx_before = len(complexes)
        complexes = {cid: c for cid, c in complexes.items() if cid in valid_cids}
        print(f"\nApplied max_total_tokens={args.max_total_tokens} filter:")
        print(f"  Complexes: {n_cx_before} -> {len(complexes)}")
        for s in SPLITS:
            before = len(pairs_by_split[s])
            pairs_by_split[s] = [
                p for p in pairs_by_split[s]
                if p["preferred"] in valid_cids and p["dispreferred"] in valid_cids
            ]
            print(f"  {s:11s} pairs: {before} -> {len(pairs_by_split[s])}")

    if not pairs_by_split["train"]:
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
    fieldnames = ["preferred", "dispreferred", "swapped_chain", "div_norm", "seq_norm"]

    def write_csv(path, pairs):
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(pairs)

    print()
    for s, stem in SPLITS.items():
        path = out_dir / f"{stem}.csv"
        write_csv(path, pairs_by_split[s])
        print(f"  {path}  ({len(pairs_by_split[s])} pairs)")

    # ---- Write statistics ----
    all_lens = [len(s1) + len(s2) for s1, s2 in complexes.values()]
    stats_lines = [
        f"Source: {DATASET} (subset {SUBSET}, HuggingFace)",
        f"Total rows: " + ", ".join(f"{s}={n}" for s, n in total_rows.items()),
        f"Max sequence length filter: {max_len}",
        f"Rows after length filter: "
        + ", ".join(f"{s}={len(kept[s])}" for s in SPLITS),
        f"Unique complexes: {len(complexes)}",
        f"Pairs: " + ", ".join(f"{s}={len(pairs_by_split[s])}" for s in SPLITS),
        f"Distance norm: div=log1p/log1p({d_max:.1f}), "
        f"seq=minmax[{seq_min:.4f},{seq_max:.4f}] (fit on train)",
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
    print(f"     tune training.dist_div_weight / training.dist_seq_weight")


if __name__ == "__main__":
    main()
