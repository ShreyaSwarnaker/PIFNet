"""
Node embeddings (DeepWalk / SVD) + inductive projection for unseen proteins.

Run:  python -m ppi_embed --branch mf --method deepwalk

THE IMPORTANT FIX IS HERE.

The notebook projected a test protein as a similarity-weighted average over
*all* ~70,773 reference proteins:

    test_emb = (sim_mtx @ train_emb) / sim_mtx.sum(axis=1)

Mean-pooled ProtVec vectors of real proteins are highly correlated -- typical
pairwise cosine sits around 0.6-0.9 and is almost never near zero. With 70k
references at roughly equal weight, that expression collapses to (approximately)
the global mean embedding, identical for every test protein. The network branch
would then carry almost no per-protein signal at test time, while validation
proteins -- which are real graph nodes with real DeepWalk vectors -- keep theirs.

That is a sufficient explanation for the validation-to-test gap on its own, and
it means the reported gap may be measuring this artifact rather than the
difficulty of cold start.

Fix: restrict to the top-k most similar references and use a softmax with a
temperature, so weight concentrates on genuine homologues.
Diagnostic: `projection_collapse_report` prints how distinguishable the
projected vectors actually are. Run it before trusting any PPI-branch result.
"""
import argparse

import numpy as np
import scipy.sparse as sp
from tqdm import tqdm


def deepwalk_embeddings(adj, cfg, seed=42):
    import random
    import networkx as nx
    from gensim.models import Word2Vec

    random.seed(seed)
    np.random.seed(seed)

    G = nx.from_scipy_sparse_array(adj)
    nodes = list(G.nodes())
    neighbours = {v: list(G.neighbors(v)) for v in nodes}

    walks = []
    for _ in tqdm(range(cfg.embed.walk_number), desc="walks"):
        random.shuffle(nodes)
        for v in nodes:
            walk = [v]
            while len(walk) < cfg.embed.walk_length:
                nbrs = neighbours[walk[-1]]
                if not nbrs:
                    break
                walk.append(random.choice(nbrs))
            walks.append([str(x) for x in walk])

    print(f"training Word2Vec on {len(walks)} walks "
          f"(epochs={cfg.embed.epochs})")
    w2v = Word2Vec(
        walks, vector_size=cfg.embed.dim, window=cfg.embed.window,
        min_count=0, sg=1, workers=cfg.embed.workers,
        epochs=cfg.embed.epochs, seed=seed,
    )

    n = adj.shape[0]
    emb = np.zeros((n, cfg.embed.dim), dtype=np.float32)
    isolated = 0
    for i in range(n):
        key = str(i)
        if key in w2v.wv:
            emb[i] = w2v.wv[key]
        else:
            isolated += 1
    if isolated:
        # Isolated nodes produce length-1 walks, so their vectors are
        # essentially untrained noise. Zeroing is honest; LayerNorm downstream
        # then gives them a constant, which the gate can learn to ignore.
        print(f"note: {isolated} nodes never appeared in a walk -> zero vector")
    return emb


def svd_embeddings(adj, cfg):
    from sklearn.decomposition import TruncatedSVD
    n_comp = min(cfg.embed.dim, adj.shape[0] - 1)
    if n_comp < cfg.embed.dim:
        print(f"warning: only {adj.shape[0]} nodes, reducing SVD dim "
              f"{cfg.embed.dim} -> {n_comp}")
    svd = TruncatedSVD(n_components=n_comp, n_iter=10, random_state=42)
    emb = svd.fit_transform(adj).astype(np.float32)
    if n_comp < cfg.embed.dim:
        emb = np.pad(emb, ((0, 0), (0, cfg.embed.dim - n_comp)))
    return emb


def project_inductive(query_X, ref_X, ref_emb, top_k=30, temperature=0.1,
                      batch_size=512):
    """
    Map proteins with no graph node into the embedding space using only their
    most similar in-graph references.

    query_X / ref_X : L2-normalised ProtVec matrices
    """
    out = np.zeros((query_X.shape[0], ref_emb.shape[1]), dtype=np.float32)
    for i in tqdm(range(0, query_X.shape[0], batch_size), desc="projection"):
        q = query_X[i:i + batch_size]
        sims = q @ ref_X.T                                   # [b, n_ref]
        k = min(top_k, sims.shape[1])
        top = np.argpartition(-sims, k - 1, axis=1)[:, :k]
        top_sims = np.take_along_axis(sims, top, axis=1)

        w = np.exp((top_sims - top_sims.max(axis=1, keepdims=True)) / temperature)
        w /= w.sum(axis=1, keepdims=True) + 1e-12
        out[i:i + batch_size] = np.einsum("bk,bkd->bd", w, ref_emb[top])
    return out


def projection_fidelity(X_known, emb, cfg, n_holdout=1000, seed=0):
    """
    The honest test of the inductive projection.

    Hide a sample of proteins that DO have real graph embeddings, project them
    as if they were unseen, and measure how well the projection recovers their
    true vector. Reports cosine to the true embedding and top-1 retrieval
    accuracy against a random-vector baseline. If retrieval is near chance, the
    network branch cannot be carrying protein-specific information at test time
    and any cold-start claim is unsupported.
    """
    rng = np.random.default_rng(seed)
    n = X_known.shape[0]
    hold = rng.choice(n, min(n_holdout, n // 2), replace=False)
    keep = np.setdiff1d(np.arange(n), hold)

    proj = project_inductive(X_known[hold], X_known[keep], emb[keep],
                             top_k=cfg.embed.project_top_k,
                             temperature=cfg.embed.project_temperature)
    true = emb[hold]

    def _norm(a):
        return a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)

    cos = (_norm(proj) * _norm(true)).sum(axis=1)
    sim_all = _norm(proj) @ _norm(true).T
    top1 = float((sim_all.argmax(axis=1) == np.arange(len(hold))).mean())

    rand = rng.normal(size=proj.shape)
    cos_rand = (_norm(rand) * _norm(true)).sum(axis=1)

    print("\n--- projection fidelity (held-out in-graph proteins) ---")
    print(f"cosine to true embedding : {cos.mean():.4f} "
          f"(random baseline {cos_rand.mean():.4f})")
    print(f"top-1 retrieval accuracy : {top1:.4f} "
          f"(chance {1 / len(hold):.5f})")
    if cos.mean() < 0.2:
        print("WEAK: projected vectors barely resemble the real ones. The "
              "network branch is close to noise for cold-start proteins.")
    return {"cosine": float(cos.mean()), "top1": top1}


def projection_collapse_report(projected, reference):
    """Is the projection actually protein-specific, or near-constant?"""
    mu = projected.mean(axis=0, keepdims=True)
    spread = np.linalg.norm(projected - mu, axis=1).mean()
    scale = np.linalg.norm(reference - reference.mean(axis=0, keepdims=True),
                           axis=1).mean()
    ratio = spread / (scale + 1e-12)
    print("\n--- projection diagnostic ---")
    print(f"mean distance from centroid, projected : {spread:.4f}")
    print(f"mean distance from centroid, in-graph  : {scale:.4f}")
    print(f"ratio                                   : {ratio:.3f}")
    # Averaging always shrinks the norm, so this ratio is expected to sit
    # below 1 even for a good projection. What matters is that it is not
    # ~0, which is what an all-reference weighted mean produces.
    if ratio < 0.05:
        print("COLLAPSED: projected vectors are nearly identical to each "
              "other. The PPI branch carries almost no test-time signal. "
              "Lower project_top_k / project_temperature, or treat "
              "cold-start proteins explicitly as PPI-missing.")
    else:
        print("OK: projected vectors retain per-protein variation. "
              "Confirm with projection_fidelity().")
    return ratio


def neighbourhood_preservation(emb, adj, k=10, sample=5000, seed=42):
    """Fraction of k nearest embedding neighbours that are true graph edges."""
    from sklearn.neighbors import NearestNeighbors
    rng = np.random.default_rng(seed)
    n = emb.shape[0]
    deg = np.asarray(adj.sum(axis=1)).ravel()
    pool = np.where(deg > 0)[0]          # isolated nodes have no edges to hit
    idx = rng.choice(pool, min(sample, len(pool)), replace=False)

    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(emb)
    _, knn = nn.kneighbors(emb[idx])

    score = 0.0
    for i, nbrs in enumerate(knn):
        true_nbrs = adj[idx[i]].indices
        score += np.intersect1d(nbrs[1:], true_nbrs).size / k
    score /= len(idx)
    print(f"neighbourhood preservation @{k}: {score:.4f} "
          f"(n={len(idx)}, non-isolated only)")
    return score


def main():
    import torch
    from config import get_config
    from data import load_all_splits, build_seq_dict
    from ppi_graph import load_protvec, protvec_matrix

    ap = argparse.ArgumentParser()
    ap.add_argument("--branch", default="mf")
    ap.add_argument("--completion", default="triadic")
    ap.add_argument("--method", default="deepwalk", choices=["deepwalk", "svd"])
    ap.add_argument("--compare", action="store_true",
                    help="also fit SVD and report preservation for both")
    args = ap.parse_args()

    cfg = get_config(branch=args.branch)
    train, val, test, _ = load_all_splits(cfg)
    seq_dict = build_seq_dict(train, val, test)
    known_ids = train["ID"].tolist() + val["ID"].tolist()

    adj = sp.load_npz(cfg.paths.work("graph",
                                     f"adj_{args.completion}_{args.branch}.npz"))

    emb = (deepwalk_embeddings(adj, cfg) if args.method == "deepwalk"
           else svd_embeddings(adj, cfg))
    print(f"{args.method} embeddings: {emb.shape}")
    neighbourhood_preservation(emb, adj)

    if args.compare:
        other = svd_embeddings(adj, cfg) if args.method == "deepwalk" \
            else deepwalk_embeddings(adj, cfg)
        print("comparison embedding:")
        neighbourhood_preservation(other, adj)

    pv = load_protvec(cfg.paths.protvec)
    X_known = protvec_matrix(known_ids, seq_dict, pv)
    X_test = protvec_matrix(test["ID"].tolist(), seq_dict, pv)

    test_emb = project_inductive(
        X_test, X_known, emb,
        top_k=cfg.embed.project_top_k,
        temperature=cfg.embed.project_temperature,
    )
    projection_collapse_report(test_emb, emb)
    projection_fidelity(X_known, emb, cfg)

    n_train = len(train)
    out = {
        "train": {pid: torch.tensor(v) for pid, v in
                  zip(train["ID"], emb[:n_train])},
        "val": {pid: torch.tensor(v) for pid, v in
                zip(val["ID"], emb[n_train:])},
        "test": {pid: torch.tensor(v) for pid, v in
                 zip(test["ID"], test_emb)},
    }
    for split, d in out.items():
        path = cfg.paths.work("embed",
                              f"{split}_{args.method}_{args.completion}_{args.branch}.pt")
        torch.save(d, path)
        print(f"saved {path}")


if __name__ == "__main__":
    main()
