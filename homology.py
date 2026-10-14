"""
Homology-based label transfer, and blending it with the neural predictions.

  python -m homology transfer --branch mf          # build + query DIAMOND
  python -m homology blend --branch mf --run mf_seq_only_deepwalk_s42

WHY THIS MATTERS MOST.

Every method above you in the MSNGO table uses homology transfer in some form:
SPROF-GO's entire contribution is homology-based label diffusion (Fmax 0.7598
on its own, with no network at all); DeepGOPlus blends a DiamondScore predictor
with its CNN; MSNGO propagates labels over a DIAMOND-built homology similarity
network. Your pipeline has no homology component anywhere, which means you are
leaving the single strongest and cheapest signal in this problem unused.

The transfer itself is simple: for a query protein q, run DIAMOND against the
training set, and score GO term t as the bitscore-weighted fraction of hits
that carry t:

    S(q, t) = sum_h bitscore(h) * 1[t in labels(h)] / sum_h bitscore(h)

Then blend with the network's probabilities, with the weight chosen on
validation only:

    final = w * neural + (1 - w) * homology

Requires DIAMOND (fast, ~seconds for this dataset size):
    conda install -c bioconda diamond
    # or: wget https://github.com/bbuchfink/diamond/releases/latest/download/diamond-linux64.tar.gz
"""
import argparse
import os
import shutil
import subprocess
import sys

import numpy as np
from tqdm import tqdm


def write_fasta(path, ids, seqs):
    with open(path, "w") as f:
        for i, s in zip(ids, seqs):
            f.write(f">{i}\n{s}\n")


def have_diamond():
    return shutil.which("diamond") is not None


def tsv_has_identity(path, min_cols=4):
    """
    Does an existing hits file carry the percent-identity column?

    Needed because run_diamond is skipped when the .tsv already exists, so a
    file written before pident was added would be silently reused -- giving
    NaN identities and an empty stratification table. Detect and re-run
    instead of trusting the filename.
    """
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return False
    with open(path) as f:
        for line in f:
            if line.strip():
                return len(line.rstrip("\n").split("\t")) >= min_cols
    return False


def run_diamond(train_fasta, query_fasta, out_tsv, threads=8,
                max_target_seqs=250, evalue=0.001):
    db = train_fasta + ".dmnd"
    if not os.path.exists(db):
        print("building DIAMOND database...")
        subprocess.run(["diamond", "makedb", "--in", train_fasta,
                        "-d", db, "--quiet"], check=True)
    print(f"querying {os.path.basename(query_fasta)} ...")
    subprocess.run([
        "diamond", "blastp", "-d", db, "-q", query_fasta, "-o", out_tsv,
        "--outfmt", "6", "qseqid", "sseqid", "bitscore", "pident",
        "--max-target-seqs", str(max_target_seqs),
        "--evalue", str(evalue), "--threads", str(threads),
        "--quiet", "--very-sensitive",
    ], check=True)
    return out_tsv


def parse_hits(tsv, exclude_self=True, with_identity=False):
    """query id -> list of (train id, bitscore[, percent identity])."""
    hits = {}
    with open(tsv) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            q, s, b = parts[0], parts[1], float(parts[2])
            if exclude_self and q == s:
                continue          # a val/test protein must not transfer to itself
            if with_identity:
                pid = float(parts[3]) if len(parts) > 3 else float("nan")
                hits.setdefault(q, []).append((s, b, pid))
            else:
                hits.setdefault(q, []).append((s, b))
    return hits


def max_identity_per_query(tsv, query_ids):
    """
    Highest percent-identity hit against the training set, per query.

    THIS IS THE NUMBER THAT DECIDES WHETHER YOUR BENCHMARK IS HONEST.
    A temporal split (train before date X, test after) does NOT remove
    sequence redundancy: a "new" protein can be 99% identical to a training
    protein (same gene, different species; a paralog; a re-annotated entry).
    DIAMOND then transfers labels trivially. This is a known and documented
    problem with time-gated evaluation, not a quirk of this dataset.
    """
    if not tsv_has_identity(tsv):
        raise ValueError(
            f"{tsv} has no percent-identity column. Delete it and re-run\n"
            "  python -m homology transfer --branch <branch>\n"
            "to regenerate with DIAMOND's pident field."
        )
    hits = parse_hits(tsv, exclude_self=True, with_identity=True)
    out = {}
    for q in query_ids:
        h = hits.get(q)
        if not h:
            out[q] = 0.0
            continue
        vals = [x[2] for x in h if not np.isnan(x[2])]
        out[q] = max(vals) if vals else 0.0
    return out


def transfer_labels(query_ids, hits, train_id_to_row, y_train, top_k=None):
    """
    Bitscore-weighted label transfer.

    Returns [n_query, n_terms] float32 in [0, 1]. Queries with no hit get all
    zeros, which is the correct abstention -- the blend then falls back
    entirely to the neural prediction for those proteins.
    """
    n_terms = y_train.shape[1]
    out = np.zeros((len(query_ids), n_terms), dtype=np.float32)
    no_hit = 0

    for i, q in enumerate(tqdm(query_ids, desc="transfer")):
        h = hits.get(q)
        if not h:
            no_hit += 1
            continue
        h.sort(key=lambda t: -t[1])
        if top_k:
            h = h[:top_k]
        rows, scores = [], []
        for sid, b in h:
            r = train_id_to_row.get(sid)
            if r is not None:
                rows.append(r)
                scores.append(b)
        if not rows:
            no_hit += 1
            continue
        w = np.asarray(scores, dtype=np.float32)
        w /= w.sum()
        out[i] = (w[:, None] * y_train[rows].astype(np.float32)).sum(axis=0)

    print(f"{no_hit}/{len(query_ids)} queries had no usable DIAMOND hit "
          f"({100 * no_hit / max(len(query_ids), 1):.1f}%) -> zero row, "
          "blend falls back to the network for those")
    return out


# ------------------------------------------------------------------ blend --
def grid_search_blend(y_true_val, neural_val, homo_val, go_terms, ontology,
                      ic, weights=None):
    """Pick the blend weight on VALIDATION only, then report it."""
    from evaluate import evaluate_cafa
    if weights is None:
        weights = np.arange(0.0, 1.01, 0.05)
    best_w, best_f = 1.0, -1.0
    for w in weights:
        blend = w * neural_val + (1 - w) * homo_val
        r = evaluate_cafa(y_true_val, blend, go_terms, ontology, ic,
                          verbose=False)
        flag = ""
        if r["Fmax"] > best_f:
            best_f, best_w, flag = r["Fmax"], w, "  <-- best"
        print(f"  w={w:.2f} (neural) : val Fmax={r['Fmax']:.4f}{flag}")
    print(f"\nchosen w={best_w:.2f}, val Fmax={best_f:.4f}")
    return best_w, best_f


def main():
    from config import get_config
    from data import load_all_splits, get_labels, load_ic
    from evaluate import evaluate_cafa, load_ontology

    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["transfer", "blend"])
    ap.add_argument("--branch", default="mf")
    ap.add_argument("--run", default=None,
                    help="run tag under work/runs/ for blend")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--top-k", type=int, default=None,
                    help="use only the top-k hits per query")
    args = ap.parse_args()

    cfg = get_config(branch=args.branch)
    train, val, test, go_terms = load_all_splits(cfg)
    y_tr = get_labels(train, go_terms)

    homo_dir = os.path.join(cfg.paths.work_root, "homology")
    os.makedirs(homo_dir, exist_ok=True)

    if args.command == "transfer":
        if not have_diamond():
            sys.exit("DIAMOND not found on PATH.\n"
                     "  conda install -c bioconda diamond\n"
                     "then re-run.")
        tr_fa = os.path.join(homo_dir, f"{args.branch}_train.fasta")
        if not os.path.exists(tr_fa):
            write_fasta(tr_fa, train["ID"], train["sequence"])

        train_id_to_row = {p: i for i, p in enumerate(train["ID"])}
        for split, df in [("val", val), ("test", test)]:
            q_fa = os.path.join(homo_dir, f"{args.branch}_{split}.fasta")
            if not os.path.exists(q_fa):
                write_fasta(q_fa, df["ID"], df["sequence"])
            tsv = os.path.join(homo_dir, f"{args.branch}_{split}_hits.tsv")
            if not tsv_has_identity(tsv):
                if os.path.exists(tsv):
                    print(f"{os.path.basename(tsv)} predates the percent-identity "
                          "column; regenerating")
                    os.remove(tsv)
                run_diamond(tr_fa, q_fa, tsv, args.threads)
            hits = parse_hits(tsv)
            scores = transfer_labels(df["ID"].tolist(), hits,
                                     train_id_to_row, y_tr, args.top_k)
            out = os.path.join(homo_dir, f"{args.branch}_{split}_homology.npz")
            np.savez_compressed(out, scores=scores,
                                ids=df["ID"].values, go_terms=go_terms)
            print(f"saved {out}\n")
        return

    # ---- blend ----
    if not args.run:
        sys.exit("--run <tag> required for blend")
    run_dir = os.path.join(cfg.paths.work_root, "runs", args.run)
    ontology = load_ontology(cfg.paths.ontology, cfg.namespace)
    ic, go_to_ic = load_ic(cfg, go_terms)

    results = {}
    blend_w = None
    for split in ("val", "test"):
        pred = np.load(os.path.join(run_dir, f"pred_{split}.npz"),
                       allow_pickle=True)
        neural = pred["y_score"]
        y_true = pred["y_true"]
        go_valid = pred["go_terms"]
        ic_valid = pred["ic"]

        homo_all = np.load(os.path.join(homo_dir,
                                        f"{args.branch}_{split}_homology.npz"),
                           allow_pickle=True)
        # align homology columns to the evaluated (IC-valid) term subset
        col = {g: i for i, g in enumerate([str(x) for x in homo_all["go_terms"]])}
        idx = [col[str(g)] for g in go_valid]
        homo = homo_all["scores"][:, idx]

        if split == "val":
            print("=== blend weight search (validation only) ===")
            blend_w, _ = grid_search_blend(y_true, neural, homo, go_valid,
                                           ontology, ic_valid)
            print()

        blended = blend_w * neural + (1 - blend_w) * homo
        for name, arr in [("neural only", neural), ("homology only", homo),
                          (f"blend w={blend_w:.2f}", blended)]:
            r = evaluate_cafa(y_true, arr, go_valid, ontology, ic_valid,
                              verbose=False)
            print(f"{split:5s} {name:<18} Fmax={r['Fmax']:.4f} "
                  f"wFmax={r['wFmax']:.4f} AuPRC={r['AuPRC']:.4f} "
                  f"Smin={r['Smin']:.4f}")
            results[f"{split}_{name}"] = r
        print()

    out = os.path.join(run_dir, "blended_results.json")
    import json
    with open(out, "w") as f:
        json.dump({"blend_weight": float(blend_w),
                   "results": {k: {kk: float(vv) for kk, vv in v.items()}
                               for k, v in results.items()}}, f, indent=2)
    print(f"saved {out}")


if __name__ == "__main__":
    main()