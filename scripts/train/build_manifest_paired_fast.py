"""Fast paired-MSA assembly: search unique chains ONCE, pair by taxonomy.

COMMAND 2.7 of the evolution pipeline -- a drop-in replacement for the slow
2.6 paired path.

Why 2.6 is slow: it runs colabfold_search a SECOND time at *complex*
granularity, re-scanning each chain once per complex it appears in. With
~5.1k unique chains but ~68k complexes that is ~8-27x redundant DB search.

Pairing does not need a second search. AlphaFold-Multimer / ColabFold pair
by *taxonomy*: search each unique chain once (retaining each hit's taxid),
then for every complex intersect its two chains' hits by species. That is a
pure-CPU join -- O(hits) per complex, trivially parallel.

This script only does the JOIN (the ``pair`` subcommand). The single search
+ taxid table is produced by the bash wrapper (2.7_build_manifest_fast.sh)
via one ``mmseqs search`` + ``mmseqs convertalis``. Unpaired depth is reused
from the existing per-chain a3m (a3m_unpaired/), so only the *paired block*
is rebuilt here. The output CSVs are byte-compatible with what 2.6's
merge_csv produced (key = pair index for paired rows, -1 for unpaired), so
the downstream ``build`` step of build_manifest_local_paired.py is reused
unchanged.

Usage:
    python build_manifest_paired_fast.py pair \
        --hits        hits.tsv          # query target taxid qstart qend qaln taln
        --seqs_fasta  seqs.fasta        # unique chains (>seq_id)
        --map_in      map.json          # from build_manifest_local_paired.py extract
        --unpaired_dir a3m_unpaired     # existing <seq_id>.a3m (unpaired depth)
        --csv_out     csv/
"""
import argparse
import json
import re
from pathlib import Path

MAX_PAIRED_SEQS = 8192
MAX_MSA_SEQS = 16384

# UniRef headers carry taxonomy as "... TaxID=9606 RepID=...". We read taxid
# from the target header (theader) rather than mmseqs `taxid`, because the local
# uniref30 DB has no taxonomy mapping built.
_TAXID_RE = re.compile(r"TaxID=(\d+)")


def read_fasta(path):
    seqs, sid, buf = {}, None, []
    for line in Path(path).read_text().splitlines():
        if line.startswith(">"):
            if sid is not None:
                seqs[sid] = "".join(buf)
            sid = line[1:].split()[0]; buf = []
        elif line.strip():
            buf.append(line.strip())
    if sid is not None:
        seqs[sid] = "".join(buf)
    return seqs


def a3m_row(qaln, taln, qstart, qend, qlen):
    """Reconstruct an a3m row (target aligned to full query) from mmseqs aln.

    Match columns (query has a residue) -> upper/'-'; insertions (gap in
    query) -> lowercase; pad leading/trailing query gaps to full length.
    """
    core = []
    for qc, tc in zip(qaln, taln):
        if qc == "-":
            if tc != "-":
                core.append(tc.lower())
        else:
            core.append(tc.upper() if tc != "-" else "-")
    return "-" * (qstart - 1) + "".join(core) + "-" * (qlen - qend)


def parse_hits(hits_path, qlens):
    """query seq_id -> {taxid: a3m_row} keeping the first (best) hit per taxid.

    Expected columns (mmseqs convertalis --format-output):
        query, target, qstart, qend, qaln, taln, theader
    taxid is parsed from theader (TaxID=...).
    """
    by_chain = {}
    with open(hits_path) as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 7:
                continue
            q, t, qs, qe, qa, ta = p[0], p[1], p[2], p[3], p[4], p[5]
            header = "\t".join(p[6:])  # theader (rejoin defensively)
            m = _TAXID_RE.search(header)
            if not m:
                continue
            tax = int(m.group(1))
            if tax <= 0:
                continue
            qlen = qlens.get(q)
            if qlen is None:
                continue
            d = by_chain.setdefault(q, {})
            if tax in d:
                continue  # first hit per taxid is the best (mmseqs orders by score)
            d[tax] = a3m_row(qa, ta, int(qs), int(qe), qlen)
    return by_chain


def unpaired_rows(a3m_path, limit):
    """Sequence rows (every other line) from an existing unpaired a3m, minus query."""
    if not a3m_path.exists():
        return []
    lines = a3m_path.read_text().strip().splitlines()
    rows = lines[1::2]
    return rows[1: 1 + limit]  # drop query (row 0)


def write_entity_csv(path, paired, unpaired):
    # Strip NUL bytes / surrounding whitespace: a stray '\0' in a sequence makes
    # pandas read the cell as NaN downstream (-> 'float has no strip' crash that
    # silently drops the complex). Drop any row that is empty after cleaning.
    def clean(s):
        return s.replace("\x00", "").strip()
    keys = list(range(len(paired))) + [-1] * len(unpaired)
    seqs = [clean(s) for s in (paired + unpaired)]
    rows = [(k, s) for k, s in zip(keys, seqs) if s]
    path.write_text("\n".join(["key,sequence"] + [f"{k},{s}" for k, s in rows]) + "\n")


def cmd_pair(args):
    seqs = read_fasta(args.seqs_fasta)
    qlens = {sid: len(s) for sid, s in seqs.items()}
    hits = parse_hits(args.hits, qlens)
    meta = json.loads(Path(args.map_in).read_text())
    complex_meta = meta["complex_meta"]
    csv_dir = Path(args.csv_out); csv_dir.mkdir(parents=True, exist_ok=True)
    unpaired_dir = Path(args.unpaired_dir)

    written, no_pair = 0, 0
    for cid, m in complex_meta.items():
        sids = m["seq_ids"]
        if len(sids) != 2:
            continue  # only 2-chain complexes here
        a, b = sids
        ha, hb = hits.get(a, {}), hits.get(b, {})
        shared = [t for t in ha.keys() if t in hb]  # taxids present in BOTH chains
        if not shared:
            no_pair += 1
        # pair 0 = the query self-row; pairs 1..K = shared taxa
        rows = {a: [seqs[a]], b: [seqs[b]]}
        for t in shared[: MAX_PAIRED_SEQS - 1]:
            rows[a].append(ha[t]); rows[b].append(hb[t])
        for idx, sid in enumerate(sids):
            paired = rows[sid]
            up = unpaired_rows(unpaired_dir / f"{sid}.a3m", MAX_MSA_SEQS - len(paired))
            write_entity_csv(csv_dir / f"{cid}_{idx}.csv", paired, up)
            written += 1
    print(f"Wrote {written} per-(complex,entity) CSVs to {csv_dir}")
    print(f"  complexes with zero shared taxa (unpaired-only): {no_pair}/{len(complex_meta)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="phase", required=True)
    pp = sub.add_parser("pair")
    pp.add_argument("--hits", required=True)
    pp.add_argument("--seqs_fasta", required=True)
    pp.add_argument("--map_in", required=True)
    pp.add_argument("--unpaired_dir", required=True)
    pp.add_argument("--csv_out", required=True)
    pp.set_defaults(func=cmd_pair)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
