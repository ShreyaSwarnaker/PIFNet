"""
Build the multi-species PPI graph and complete it.

Run:  python -m ppi_graph --branch mf

Two substantive corrections vs the notebook:

1. COMPLETION SEMANTICS. The notebook's function was called
   "triadic closure" but the loop body only ever added edge (a, c) where c is a
   *sequence* neighbour of a. The graph edge (a, b) acted purely as a trigger.
   That means the added edges carried no relational information at all -- the
   result was a sequence-kNN graph overlaid on the PPI graph, which is exactly
   the circularity a reviewer will attack ("your network branch is just
   re-encoded sequence").

   The real closure is: if a--b is a measured interaction and c is
   sequence-similar to a, then c plausibly also interacts with *b*. So we add
   (c, b). This uses the observed edge as evidence and is not reducible to
   sequence similarity. `completion_mode="knn"` reproduces the old behaviour
   for ablation.

2. COMPLEXITY. The original looped over all 4.36M edges x 15 candidates doing
   LIL-matrix element lookups (~65M Python-level sparse ops, hours of runtime),
   and re-derived the same candidate set once per incident edge, so a degree-500
   hub redid identical work 500 times. Here the candidate edges are generated
   once per node and assembled as COO arrays.
"""
import argparse
import multiprocessing as mp
import os
from collections import defaultdict

import numpy as np
import scipy.sparse as sp
from tqdm import tqdm

_CPU = os.cpu_count() or 4


# ----------------------------------------------------------------- STRING ---
def load_uniprot_to_string(cfg, protein_ids):
    """UniProt accession -> list of STRING ids, restricted to our proteins."""
    wanted = {p.split("-")[0] for p in protein_ids}
    mapping = defaultdict(list)

    for taxid in cfg.species:
        path = cfg.paths.string_aliases(taxid)
        if not os.path.exists(path):
            print(f"aliases {taxid}: not found, skipping")
            continue
        with open(path) as f:
            next(f)
            for line in tqdm(f, desc=f"aliases {taxid}"):
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 3:
                    continue
                string_id, alias, source = parts[0], parts[1], parts[2]
                if source in ("UniProt_AC", "Ensembl_UniProt"):
                    acc = alias.split("-")[0]
                    if acc in wanted:
                        mapping[acc].append(string_id)

    covered = sum(1 for p in wanted if p in mapping)
    print(f"STRING alias coverage: {covered}/{len(wanted)} UniProt accessions")
    return mapping


def build_string_adj(cfg, protein_ids, mapping):
    uni2idx = {}
    for i, pid in enumerate(protein_ids):
        uni2idx.setdefault(pid.split("-")[0], i)

    # STRING id -> row index, kept separate per species so that identical
    # accessions in different organisms cannot collide.
    by_species = defaultdict(dict)
    for acc, idx in uni2idx.items():
        for sid in mapping.get(acc, []):
            by_species[sid.split(".")[0]][sid] = idx

    rows, cols, vals = [], [], []
    for taxid in cfg.species:
        local = by_species.get(taxid, {})
        print(f"species {taxid}: {len(local)} mapped proteins")
        if not local:
            continue
        links_path = cfg.paths.string_links(taxid)
        if not os.path.exists(links_path):
            print(f"links {taxid}: not found, skipping")
            continue
        with open(links_path) as f:
            next(f)
            for line in tqdm(f, desc=f"links {taxid}"):
                p1, p2, score = line.split()
                score = int(score)
                if score < cfg.graph.string_threshold:
                    continue
                i, j = local.get(p1), local.get(p2)
                if i is not None and j is not None and i != j:
                    rows.append(i); cols.append(j); vals.append(score / 1000.0)

    n = len(protein_ids)
    adj = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    return adj.maximum(adj.T)


# --------------------------------------------------- BioGRID / IntAct -------
def build_biogrid_adj(cfg, protein_ids):
    import pandas as pd
    uni2idx = {p.split("-")[0]: i for i, p in enumerate(protein_ids)}
    cols = ["SWISS-PROT Accessions Interactor A",
            "SWISS-PROT Accessions Interactor B"]

    rows, colsi = [], []
    # chunked + vectorized: the notebook read the whole ~2GB file then used
    # df.iterrows() over millions of rows.
    for chunk in tqdm(pd.read_csv(cfg.paths.biogrid, sep="\t", usecols=cols,
                                  chunksize=200_000, low_memory=False),
                      desc="BioGRID"):
        a = chunk[cols[0]].astype(str).str.split("|").str[0].map(uni2idx)
        b = chunk[cols[1]].astype(str).str.split("|").str[0].map(uni2idx)
        ok = a.notna() & b.notna()
        rows.extend(a[ok].astype(int).tolist())
        colsi.extend(b[ok].astype(int).tolist())

    n = len(protein_ids)
    adj = sp.csr_matrix((np.ones(len(rows)), (rows, colsi)), shape=(n, n))
    return adj.maximum(adj.T)


def build_intact_adj(cfg, protein_ids):
    uni2idx = {p.split("-")[0]: i for i, p in enumerate(protein_ids)}
    rows, cols = [], []
    with open(cfg.paths.intact) as f:
        for line in tqdm(f, desc="IntAct"):
            if line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            p1 = parts[0].split(":")[-1].split("-")[0]
            p2 = parts[1].split(":")[-1].split("-")[0]
            i, j = uni2idx.get(p1), uni2idx.get(p2)
            if i is not None and j is not None and i != j:
                rows.append(i); cols.append(j)

    n = len(protein_ids)
    adj = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    return adj.maximum(adj.T)


# ------------------------------------------------------------- ProtVec ------
def load_protvec(path):
    model = {}
    with open(path) as f:
        for line in f:
            line = line.strip().replace('"', "")
            if not line:
                continue
            parts = line.split(",") if "," in line else line.split()
            try:
                vec = np.asarray(parts[1:], dtype=float)
            except ValueError:
                continue
            if vec.size:
                model[parts[0]] = vec
    print(f"ProtVec: {len(model)} k-mers")
    return model


def protvec_embed(seq, model, k=3):
    if not isinstance(seq, str) or len(seq) < k:
        return None
    vecs = [model[seq[i:i + k]] for i in range(len(seq) - k + 1)
            if seq[i:i + k] in model]
    return np.mean(vecs, axis=0) if vecs else None


_PV_MODEL = None


def _pv_init(model):
    # Runs once per worker process (not once per protein), so the k-mer
    # dict is pickled n_jobs times total, not n_proteins times.
    global _PV_MODEL
    _PV_MODEL = model


def _pv_one(seq):
    return protvec_embed(seq, _PV_MODEL)


def protvec_matrix(protein_ids, seq_dict, model, n_jobs=None):
    """
    Parallelized over proteins with multiprocessing (not threads: the k-mer
    dict lookups here are pure-Python and hold the GIL, so threads would not
    give real parallelism the way they do for the numpy-heavy evaluate_cafa).
    Falls back to sequential for small inputs, where pool startup overhead
    would dominate.
    """
    dim = len(next(iter(model.values())))
    seqs = [seq_dict.get(pid, "") for pid in protein_ids]
    n_jobs = n_jobs or min(_CPU, 16)

    if n_jobs > 1 and len(seqs) > 2000:
        with mp.Pool(n_jobs, initializer=_pv_init, initargs=(model,)) as pool:
            results = list(tqdm(
                pool.imap(_pv_one, seqs, chunksize=max(1, len(seqs) // (n_jobs * 8))),
                total=len(seqs), desc=f"ProtVec ({n_jobs} workers)"))
    else:
        results = [protvec_embed(s, model)
                  for s in tqdm(seqs, desc="ProtVec")]

    X = np.zeros((len(protein_ids), dim))
    missing = 0
    for i, v in enumerate(results):
        if v is None:
            missing += 1
        else:
            X[i] = v
    if missing:
        print(f"warning: {missing} proteins had no k-mer hits (zero vectors)")
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)


# ---------------------------------------------------------- completion ------
_CG = {}


def _cg_init(idx, sim, boundaries, dst, deg_factor, mode, cap, alpha):
    # One copy per worker process, set once via Pool's initializer -- not
    # re-pickled per chunk. On Linux (fork start method, the default) these
    # arrays are inherited via copy-on-write without even being pickled.
    global _CG
    _CG = dict(idx=idx, sim=sim, boundaries=boundaries, dst=dst,
              deg_factor=deg_factor, mode=mode, cap=cap, alpha=alpha)


def _cg_chunk(node_range):
    idx, sim = _CG["idx"], _CG["sim"]
    boundaries, dst = _CG["boundaries"], _CG["dst"]
    deg_factor, mode = _CG["deg_factor"], _CG["mode"]
    cap, alpha = _CG["cap"], _CG["alpha"]

    new_r, new_c, new_v = [], [], []
    for a in node_range:
        lo, hi = boundaries[a], boundaries[a + 1]
        if lo == hi:
            continue
        partners = dst[lo:hi]
        cand, cand_sim = idx[a], sim[a]

        if mode == "triadic":
            if len(partners) > cap:
                keep = np.argsort(-deg_factor[partners])[:cap]
                partners = partners[keep]
            cc = np.repeat(cand, len(partners))
            bb = np.tile(partners, len(cand))
            ss = np.repeat(cand_sim, len(partners))
            w = ss * (deg_factor[a] * deg_factor[bb] * deg_factor[cc]) ** (alpha / 1.5)
            good = cc != bb
            new_r.extend(cc[good].tolist())
            new_c.extend(bb[good].tolist())
            new_v.extend(w[good].tolist())
        else:
            b_rep = partners[0]
            w = cand_sim * (deg_factor[a] * deg_factor[b_rep]
                            * deg_factor[cand]) ** (alpha / 1.5)
            new_r.extend([a] * len(cand))
            new_c.extend(cand.tolist())
            new_v.extend(w.tolist())
    return new_r, new_c, new_v


def complete_graph(adj, X_norm, cfg, n_jobs=None):
    """
    Sequence-guided graph completion with a preferential-attachment weight.

    triadic: (a,b) observed and c ~ a  =>  propose (c,b)
    knn:     (a,b) observed and c ~ a  =>  propose (a,c)   [original behaviour]

    Parallelized over node ranges with multiprocessing (each node's inner
    numpy ops are cheap, so the per-node Python loop overhead -- not the
    array math -- was the bottleneck; multiprocessing avoids the GIL that a
    thread pool would still be bound by here).
    """
    from sklearn.neighbors import NearestNeighbors

    n = adj.shape[0]
    k = cfg.graph.knn_k
    alpha = cfg.graph.alpha
    mode = cfg.graph.completion_mode
    n_jobs = n_jobs or min(_CPU, 16)

    print(f"kNN over ProtVec (k={k})")
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine",
                          n_jobs=n_jobs).fit(X_norm)
    dist, idx = nn.kneighbors(X_norm)
    sim, idx = 1.0 - dist[:, 1:], idx[:, 1:]        # drop self-match

    degree = np.asarray(adj.sum(axis=1)).ravel()
    deg_factor = (degree + 1.0) ** alpha
    adj_coo = adj.tocoo()

    order = np.argsort(adj_coo.row, kind="stable")
    src = adj_coo.row[order]
    dst = adj_coo.col[order]
    boundaries = np.searchsorted(src, np.arange(n + 1))

    cap = cfg.graph.max_new_edges_per_node
    new_r, new_c, new_v = [], [], []

    if n_jobs > 1 and n > 2000:
        chunk_size = max(1, n // (n_jobs * 4))
        chunks = [range(i, min(i + chunk_size, n))
                 for i in range(0, n, chunk_size)]
        with mp.Pool(n_jobs, initializer=_cg_init,
                    initargs=(idx, sim, boundaries, dst, deg_factor, mode,
                              cap, alpha)) as pool:
            for r, c, v in tqdm(pool.imap(_cg_chunk, chunks),
                                total=len(chunks),
                                desc=f"completion ({mode}, {n_jobs} workers)"):
                new_r.extend(r); new_c.extend(c); new_v.extend(v)
    else:
        for a in tqdm(range(n), desc=f"completion ({mode})"):
            lo, hi = boundaries[a], boundaries[a + 1]
            if lo == hi:
                continue
            partners = dst[lo:hi]
            cand, cand_sim = idx[a], sim[a]
            if mode == "triadic":
                if len(partners) > cap:
                    keep = np.argsort(-deg_factor[partners])[:cap]
                    partners = partners[keep]
                cc = np.repeat(cand, len(partners))
                bb = np.tile(partners, len(cand))
                ss = np.repeat(cand_sim, len(partners))
                w = ss * (deg_factor[a] * deg_factor[bb] * deg_factor[cc]) ** (alpha / 1.5)
                good = cc != bb
                new_r.extend(cc[good]); new_c.extend(bb[good]); new_v.extend(w[good])
            else:
                b_rep = partners[0]
                w = cand_sim * (deg_factor[a] * deg_factor[b_rep]
                                * deg_factor[cand]) ** (alpha / 1.5)
                new_r.extend([a] * len(cand)); new_c.extend(cand); new_v.extend(w)

    proposed = sp.csr_matrix((new_v, (new_r, new_c)), shape=(n, n))
    proposed = proposed.maximum(proposed.T)
    # Rescale (not clip) into [0, 1] so proposals stay comparable to STRING
    # confidences while preserving their ordering -- clipping created mass
    # ties at 1.0 and made the budget cut arbitrary.
    if proposed.nnz:
        proposed.data = proposed.data / proposed.data.max()

    # Never overwrite a measured edge: keep observed weight where it exists.
    existing = adj.copy()
    existing.data = np.ones_like(existing.data)
    proposed = proposed - proposed.multiply(existing)
    proposed.eliminate_zeros()

    # Global budget: keep the highest-confidence proposals only, so completion
    # supplements rather than swamps the measured interactome.
    budget = int(cfg.graph.new_edge_budget * adj.nnz)
    if proposed.nnz > budget > 0:
        cutoff = np.partition(proposed.data, -budget)[-budget]
        proposed.data[proposed.data < cutoff] = 0.0
        proposed.eliminate_zeros()
        proposed = proposed.maximum(proposed.T)   # keep symmetry after pruning
        print(f"pruned proposals to weight >= {cutoff:.4f} "
              f"({proposed.nnz} edges, budget {budget})")

    completed = (adj + proposed).tocsr()
    print(f"edges {adj.nnz} -> {completed.nnz} (+{completed.nnz - adj.nnz})")
    return completed


def graph_stats(adj, name):
    deg = np.asarray(adj.sum(axis=1)).ravel()
    nz = int((deg > 0).sum())
    print(f"[{name}] nodes with >=1 edge: {nz}/{adj.shape[0]} "
          f"({100 * nz / adj.shape[0]:.1f}%) | edges: {adj.nnz}")
    return nz


def main():
    from config import get_config
    from data import load_all_splits, build_seq_dict

    ap = argparse.ArgumentParser()
    ap.add_argument("--branch", default="mf")
    ap.add_argument("--completion", default="triadic",
                    choices=["triadic", "knn", "none"])
    args = ap.parse_args()

    cfg = get_config(branch=args.branch)
    cfg.graph.completion_mode = args.completion

    train, val, test, _ = load_all_splits(cfg)
    seq_dict = build_seq_dict(train, val, test)
    known_ids = train["ID"].tolist() + val["ID"].tolist()

    mapping = load_uniprot_to_string(cfg, known_ids)
    adj = build_string_adj(cfg, known_ids, mapping)
    graph_stats(adj, "STRING")

    for builder, label in [(build_biogrid_adj, "BioGRID"),
                           (build_intact_adj, "IntAct")]:
        try:
            extra = builder(cfg, known_ids)
            graph_stats(extra, label)
            adj = adj.maximum(extra)
        except FileNotFoundError:
            print(f"{label} file not found, skipping")
    graph_stats(adj, "union")
    sp.save_npz(cfg.paths.work("graph", f"adj_raw_{args.branch}.npz"), adj)

    if args.completion != "none":
        pv = load_protvec(cfg.paths.protvec)
        X = protvec_matrix(known_ids, seq_dict, pv)
        np.save(cfg.paths.work("graph", f"protvec_known_{args.branch}.npy"), X)
        adj = complete_graph(adj, X, cfg)
        graph_stats(adj, f"completed:{args.completion}")

    out = cfg.paths.work("graph", f"adj_{args.completion}_{args.branch}.npz")
    sp.save_npz(out, adj)
    np.save(cfg.paths.work("graph", f"known_ids_{args.branch}.npy"),
            np.array(known_ids))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
