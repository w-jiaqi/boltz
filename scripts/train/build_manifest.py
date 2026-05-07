"""Build the Boltz processed manifest from input YAML/FASTA structures.

COMMAND 2 of the evolution training pipeline.

Discovers input files, runs MSA preprocessing via the colabfold MSA server
(parallelized across worker processes), and writes the
``processed/manifest.json`` plus the ``records/``, ``structures/``, ``msa/``
directories that downstream caching needs.

CPU-only. No GPU is required or used.

Usage:
    python scripts/train/build_manifest.py \\
        --data /path/to/structures/ \\
        --output /path/to/cache/ \\
        --cache /path/to/boltz_cache \\
        --use_msa_server \\
        --preprocessing_threads 32

Output:
    cache/processed/manifest.json
    cache/processed/{records,structures,msa,...}/

Next step:
    python cache_evolution_features.py \\
        --data <unused, kept for legacy> \\
        --output /path/to/cache/ \\
        --checkpoint <boltz2_conf.ckpt> --cache <boltz_cache> \\
        --no_kernels
"""

import argparse
import sys
from pathlib import Path

from boltz.main import check_inputs, process_inputs


def _get_total_seq_len(yaml_path: Path):
    """Return total sequence length across all chains, or None on parse error."""
    import yaml
    try:
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        total = 0
        for entry in data.get("sequences", []):
            for _entity_type, info in entry.items():
                seq = info.get("sequence", "")
                if seq:
                    total += len(seq)
        return total
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Build Boltz processed manifest (CPU-only MSA preprocessing)."
    )
    parser.add_argument("--data", required=True,
                        help="Input structures (YAML/FASTA dir or single file).")
    parser.add_argument("--output", required=True,
                        help="Output dir; manifest goes to {output}/processed/manifest.json.")
    parser.add_argument("--cache", default="~/.boltz",
                        help="Boltz cache directory (CCD/mols). Default: ~/.boltz")
    parser.add_argument("--use_msa_server", action="store_true",
                        help="Generate MSAs via the colabfold server. "
                        "Required unless YAMLs already reference local MSAs.")
    parser.add_argument("--preprocessing_threads", type=int, default=32,
                        help="Worker processes for parallel MSA preprocessing. "
                        "Default: 32. The colabfold server may rate-limit "
                        "(HTTP 429) past ~32 concurrent requests.")
    parser.add_argument("--max_total_tokens", type=int, default=None,
                        help="Skip complexes where len(seq_A)+len(seq_B) exceeds this. "
                        "Useful to avoid OOM downstream. Default: no limit.")
    parser.add_argument("--max_complexes", type=int, default=None,
                        help="Only process the first N complexes (after filtering). "
                        "Useful for testing. Default: no limit.")
    args = parser.parse_args()

    data_path = Path(args.data)
    out_dir = Path(args.output)
    cache = Path(args.cache).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    mol_dir = cache / "mols"
    if not mol_dir.exists():
        print(f"ERROR: Molecule data not found at {mol_dir}")
        sys.exit(1)

    manifest_path = out_dir / "processed" / "manifest.json"
    if manifest_path.exists():
        print(f"Manifest already exists at {manifest_path}.")
        print("Delete it (and the processed/ dir) to rebuild from scratch.")
        return

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
        preprocessing_threads=args.preprocessing_threads,
    )

    if not manifest_path.exists():
        print(f"ERROR: process_inputs finished but {manifest_path} was not written.")
        sys.exit(1)

    print(f"\nDone. Manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
