"""
Ontology loading, DAG propagation, and CAFA metrics.

CORRECTED PROPAGATION. Version 10 computed, for each term i:

    y_prop[:, i] = max over descendants of i          # i's own score discarded

so any term with at least one descendant lost its direct prediction entirely.
A confidently predicted parent whose children all scored low was pushed to that
low value. Version 22 fixed this; that fix is kept here, which means numbers
from the v10 notebook are NOT comparable to anything produced by this file.
Re-score old saved probabilities through `evaluate_cafa` before comparing runs.

Smin is reported in mean per-protein IC units. Papers normalise this
differently, so do not compare the absolute value across publications.

MEMORY NOTE (this revision). `_threshold_stats` used to materialise five full
n_proteins x n_terms arrays at once (~440 MB at MSNGO scale: 6863 x 6443).
With n_jobs=4 that is ~1.8 GB of transient RSS per training job, and with
several jobs on one node the box goes to swap and everything looks frozen.
The statistics are now accumulated in protein blocks, so peak memory is
O(block x n_terms) -- about 40 MB per worker. The returned numbers are
unchanged apart from float32 summation order, which does not move Fmax at
this precision.
"""

import os
import numpy as np
from tqdm import tqdm

# np.trapz was removed in NumPy 2.0
_trapz = getattr(np, "trapezoid", None) or np.trapz

# Proteins per block in the threshold sweep. Trades a little Python loop
# overhead for a large drop in peak memory.
_STAT_BLOCK = int(os.environ.get("PF_EVAL_BLOCK", "1024"))


# ---------------------------------------------------------------- ontology --
def get_ancestors(ontology, term):
    seen, queue = set(), [term]

    while queue:
        t = queue.pop(0)

        if t in seen or t not in ontology:
            continue

        seen.add(t)
        queue.extend(ontology[t]["parents"])

    seen.discard(term)
    return list(seen)


def load_ontology(obo_file, namespace=None):
    ontology, term, valid = {}, {}, False

    with open(obo_file) as f:
        for line in f:
            line = line.strip()

            if line == "[Term]":
                if "id" in term:
                    ontology[term["id"]] = term

                term, valid = {"parents": [], "alt_ids": []}, True

            elif line == "[Typedef]":
                valid = False

            elif valid and ": " in line:
                key, _, rest = line.partition(": ")

                if key == "id":
                    term["id"] = rest

                elif key == "alt_id":
                    term["alt_ids"].append(rest)

                elif key == "namespace":
                    if namespace and rest != namespace:
                        term, valid = {}, False
                    else:
                        term["namespace"] = rest

                elif key == "is_a":
                    term["parents"].append(rest.split(" ! ")[0])
                    

                elif key == "relationship": ###newly added
                    rel, _, target = rest.partition(" ")
                    if rel == "part_of":
                        term["parents"].append(target.split(" ! ")[0])

                elif key == "is_obsolete" and rest == "true":
                    term, valid = {}, False

        if "id" in term:
            ontology[term["id"]] = term

    for k in list(ontology):
        ontology[k]["ancestors"] = get_ancestors(ontology, k)

    for k in list(ontology):
        for alt in ontology[k].get("alt_ids", []):
            ontology.setdefault(alt, ontology[k])

    print(f"ontology: {len(ontology)} terms ({namespace})", flush=True)
    return ontology


# ---------------------------------------------------------- propagation setup --
#
# The old implementation constructed a dense n_terms x n_terms ancestor mask
# and then repeatedly indexed it for every term.
#
# For propagation we do not need the complete transitive ancestor matrix.
# Direct child -> parent edges are sufficient if terms are processed from
# descendants toward ancestors.
#
# The result is mathematically identical:
#
#     parent = max(parent, child)
#
# performed in descendant-first order propagates a score through the entire DAG.
#
# This structure is cached because ontology and go_terms do not change between
# epochs.


def _build_propagation_structure(go_terms, ontology):
    go_map = {g: i for i, g in enumerate(go_terms)}
    n = len(go_terms)

    # direct child -> parent edges
    children = [[] for _ in range(n)]

    for child_i, go in enumerate(go_terms):
        if go not in ontology:
            continue

        for parent in ontology[go]["parents"]:
            parent_i = go_map.get(parent)

            if parent_i is not None and parent_i != child_i:
                children[child_i].append(parent_i)

    # Calculate a descendant-first topological order.
    #
    # If child -> parent is an edge, children must be processed before
    # parents. Kahn's algorithm is used here rather than relying on the
    # number of ancestors as a proxy for depth.
    indegree = np.zeros(n, dtype=np.int32)

    for child_i in range(n):
        for parent_i in children[child_i]:
            indegree[parent_i] += 1

    queue = [i for i in range(n) if indegree[i] == 0]
    order = []

    head = 0
    while head < len(queue):
        child_i = queue[head]
        head += 1

        order.append(child_i)

        for parent_i in children[child_i]:
            indegree[parent_i] -= 1

            if indegree[parent_i] == 0:
                queue.append(parent_i)

    # GO should be a DAG. If something unexpected prevents a complete
    # topological ordering, fall back to the old ancestor-count ordering.
    if len(order) != n:
        print(
            "warning: ontology did not produce a complete topological order; "
            "using ancestor-count ordering",
            flush=True
        )

        # This is only a safety fallback.
        depth = np.zeros(n, dtype=np.int32)

        for i, go in enumerate(go_terms):
            if go in ontology:
                depth[i] = len(ontology[go].get("ancestors", []))

        order = np.argsort(-depth).tolist()

    # Convert lists to arrays for slightly cheaper repeated iteration.
    children = [
        np.asarray(parents, dtype=np.int32)
        if parents else np.empty(0, dtype=np.int32)
        for parents in children
    ]

    return children, np.asarray(order, dtype=np.int32)


# Cache by ontology object + exact GO-term sequence.
#
# The key uses id(ontology), so the cache ALSO keeps a reference to the
# ontology object itself. Without that reference the ontology could be
# garbage-collected and a different dict could later be allocated at the same
# address, silently returning a propagation structure built from a different
# ontology.
_propagation_cache = {}


def clear_propagation_cache():
    _propagation_cache.clear()


def _get_propagation_structure(go_terms, ontology):
    key = (id(ontology), tuple(go_terms))

    cached = _propagation_cache.get(key)

    if cached is None:
        structure = _build_propagation_structure(go_terms, ontology)
        # keep `ontology` alive so id() cannot be recycled under us
        cached = (structure, ontology)
        _propagation_cache[key] = cached

    return cached[0]


def build_ancestor_mask(go_terms, ontology):
    """
    Retained for compatibility with any external code that uses this function.

    This is NOT used by propagate_scores/evaluate_cafa anymore.

    It still returns the same dense ancestor mask as the original implementation.
    """

    go_map = {g: i for i, g in enumerate(go_terms)}
    n = len(go_terms)

    mask = np.zeros((n, n), dtype=bool)

    for i, go in enumerate(go_terms):
        if go not in ontology:
            continue

        for anc in ontology[go]["ancestors"]:
            j = go_map.get(anc)

            if j is not None and j != i:
                mask[i, j] = True

    return mask


def propagate_scores(y_score, ancestor_mask):
    """
    Corrected propagation:

        score(parent) <- max(score(parent), score(child))

    The public function signature is kept unchanged.

    `ancestor_mask` may be either:

      1. the original dense ancestor mask, or
      2. the cached sparse propagation structure returned internally.

    evaluate_cafa uses the sparse structure.
    """

    # New sparse representation:
    if isinstance(ancestor_mask, tuple):
        children, order = ancestor_mask

        y = y_score.copy()

        for child_i in order:
            parents = children[child_i]

            if parents.size:
                child_score = y[:, child_i:child_i + 1]

                y[:, parents] = np.maximum(
                    y[:, parents],
                    child_score
                )

        return y

    # Original dense-mask behavior retained for compatibility.
    y = y_score.copy()

    depth = ancestor_mask.sum(axis=1)

    for i in np.argsort(-depth):
        anc = np.where(ancestor_mask[i])[0]

        if anc.size:
            y[:, anc] = np.maximum(
                y[:, anc],
                y[:, i:i + 1]
            )

    return y


# ----------------------------------------------------------------- metrics --
def _threshold_stats(
    tau,
    y_true_bool,
    y_score,
    ic32,
    true_counts,
    ic_true,
    block=None
):
    """
    One threshold's worth of CAFA metric calculations.

    Accumulated in blocks of proteins so that no full n_proteins x n_terms
    temporary is ever alive. Mathematically identical to the unblocked form.
    """

    block = block or _STAT_BLOCK
    n = y_score.shape[0]

    pred_counts = np.empty(n, dtype=np.int64)
    tp_counts = np.empty(n, dtype=np.int64)
    ic_pred = np.empty(n, dtype=np.float32)
    ic_tp = np.empty(n, dtype=np.float32)

    for s in range(0, n, block):
        e = min(s + block, n)

        preds = y_score[s:e] >= tau
        tp = preds & y_true_bool[s:e]

        pred_counts[s:e] = preds.sum(axis=1)
        tp_counts[s:e] = tp.sum(axis=1)

        # Two matrix-vector products per block, as before.
        ic_pred[s:e] = preds.astype(np.float32) @ ic32
        ic_tp[s:e] = tp.astype(np.float32) @ ic32

        del preds, tp

    p_n = tp_counts / np.maximum(pred_counts, 1e-10)
    r_n = tp_counts / np.maximum(true_counts, 1e-10)

    avg_pr = (
        p_n[pred_counts > 0].mean()
        if np.any(pred_counts > 0)
        else 0.0
    )

    avg_rc = r_n.mean()

    wp = ic_tp / np.maximum(ic_pred, 1e-10)
    wr = ic_tp / np.maximum(ic_true, 1e-10)

    avg_wp = (
        wp[ic_pred > 0].mean()
        if np.any(ic_pred > 0)
        else 0.0
    )

    avg_wr = wr.mean()

    # Algebraically equivalent to:
    #
    #   (y_true - tp) @ ic
    #   (pred - tp) @ ic
    #
    # but avoids two additional large matrix multiplications.
    ru = ic_true - ic_tp
    mi = ic_pred - ic_tp

    smin_val = np.sqrt(
        ru.mean() ** 2 +
        mi.mean() ** 2
    )

    return avg_pr, avg_rc, avg_wp, avg_wr, smin_val


def evaluate_cafa(
    y_true,
    y_score,
    go_terms,
    ontology,
    ic,
    thresholds=None,
    propagate=True,
    verbose=True,
    n_jobs=None,
    n_thresholds=101
):
    """
    Calculate CAFA metrics.

    n_thresholds:
        Number of threshold values when thresholds is not explicitly given.
        Use 101 for final reported CAFA results. Training can use a smaller
        grid such as 26.

    n_jobs:
        Number of workers for the threshold sweep. Keep this modest when
        several training jobs share the same node. BLAS thread counts should
        be limited separately by the launcher/environment.
    """
    if thresholds is None:
        thresholds = np.linspace(0, 1, n_thresholds)
    thresholds = np.asarray(thresholds)
    ic = np.asarray(ic, dtype=float)
    ic32 = np.ascontiguousarray(ic, dtype=np.float32)
    y_true_bool = y_true > 0

    if propagate:
        propagation_structure = _get_propagation_structure(go_terms, ontology)
        y_score = propagate_scores(y_score, propagation_structure)

    # The sweep only ever compares y_score against a threshold, so a
    # contiguous float32 copy both halves the memory traffic and makes the
    # per-block slices cheap.
    y_score = np.ascontiguousarray(y_score, dtype=np.float32)

    true_counts = y_true.sum(axis=1)
    y_true32 = np.ascontiguousarray(y_true, dtype=np.float32)
    ic_true = y_true32 @ ic32
    del y_true32

    n_jobs = max(1, int(n_jobs or 1))

    if n_jobs == 1 or len(thresholds) == 1:
        it = tqdm(thresholds, desc="thresholds", leave=False) if verbose \
            else thresholds
        results = [
            _threshold_stats(
                tau, y_true_bool, y_score, ic32, true_counts, ic_true
            )
            for tau in it
        ]
    else:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=n_jobs) as ex:
            futures = [
                ex.submit(
                    _threshold_stats,
                    tau,
                    y_true_bool,
                    y_score,
                    ic32,
                    true_counts,
                    ic_true
                )
                for tau in thresholds
            ]
            # Submission order is intentional: AuPRC is computed from the
            # threshold curve and must remain deterministic.
            results = [f.result() for f in futures]

    fmax = 0.0
    wfmax = 0.0
    smin = np.inf
    pr_arr = []
    rc_arr = []

    for avg_pr, avg_rc, avg_wp, avg_wr, smin_val in results:
        if avg_pr + avg_rc > 0:
            fmax = max(
                fmax,
                2 * avg_pr * avg_rc / (avg_pr + avg_rc)
            )
        if avg_wp + avg_wr > 0:
            wfmax = max(
                wfmax,
                2 * avg_wp * avg_wr / (avg_wp + avg_wr)
            )
        smin = min(smin, smin_val)
        pr_arr.append(avg_pr)
        rc_arr.append(avg_rc)

    pr_arr = np.asarray(pr_arr)
    rc_arr = np.asarray(rc_arr)
    order = np.argsort(rc_arr)
    auprc = _trapz(pr_arr[order], rc_arr[order])

    return {
        "Fmax": fmax,
        "wFmax": wfmax,
        "Smin": smin,
        "AuPRC": auprc
    }


def macro_auprc(y_true, y_score):
    """Per-term macro AUPRC."""

    from sklearn.metrics import average_precision_score

    scores = [
        average_precision_score(
            y_true[:, j],
            y_score[:, j]
        )
        for j in range(y_true.shape[1])
        if y_true[:, j].sum() > 0
    ]

    return float(np.mean(scores)) if scores else 0.0


def evaluate_by_degree(
    y_true,
    y_score,
    go_terms,
    ontology,
    ic,
    degrees,
    bins=(0, 1, 10, 100, np.inf),
    n_jobs=None
):
    """
    Fmax split by how much real network evidence each protein had.

    n_jobs is forwarded to evaluate_cafa. Previously it was not, so every bin
    ran the full 101-threshold sweep single-threaded.
    """

    out = {}

    for lo, hi in zip(bins[:-1], bins[1:]):

        sel = (
            (degrees >= lo) &
            (degrees < hi)
        )

        if sel.sum() < 50:
            continue

        r = evaluate_cafa(
            y_true[sel],
            y_score[sel],
            go_terms,
            ontology,
            ic,
            verbose=False,
            n_jobs=n_jobs
        )

        out[f"degree[{lo},{hi})"] = {
            "n": int(sel.sum()),
            **r
        }

    return out