"""Loss functions."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Class-balanced focal loss.

    BUG FIXED HERE. The previous version computed

        loss = alpha * (1 - pt)**gamma * bce

    with a single scalar `alpha` applied to every element. That does no class
    balancing at all -- it multiplies the whole loss by a constant, which is
    mathematically identical to scaling the learning rate. Verified: loss at
    alpha=0.5 was exactly 2.0x loss at alpha=0.25 for the same inputs.

    Standard focal loss uses alpha_t = alpha for positives and (1 - alpha) for
    negatives. That distinction matters enormously here: with 6445 GO terms and
    roughly 30 true per protein, ~99.5% of entries are negative, so the
    unbalanced version let negatives dominate the gradient. Positives now need
    alpha > 0.5 to be up-weighted; 0.75 is the usual starting point.

    Set balanced=False to reproduce the old behaviour exactly.
    """

    def __init__(self, alpha=0.75, gamma=2.0, reduction="mean", balanced=True):
        super().__init__()
        self.alpha, self.gamma, self.reduction = alpha, gamma, reduction
        self.balanced = balanced

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets,
                                                 reduction="none")
        pt = torch.exp(-bce)
        if self.balanced:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        else:
            alpha_t = self.alpha
        loss = alpha_t * (1 - pt) ** self.gamma * bce
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class HierarchicalFocalLoss(nn.Module):
    """
    Force ancestors of every true term to be true before computing focal loss.

    Warning: if the label CSVs are already ancestor-closed -- which most
    CAFA-style releases are -- this is mathematically a no-op and must not be
    reported as a contribution. Call `check_labels_propagated` once and read
    the printout before using it.
    """

    def __init__(self, ancestor_matrix, alpha=0.75, gamma=2.0, balanced=True):
        super().__init__()
        self.register_buffer("ancestor_matrix", ancestor_matrix)
        self.focal = FocalLoss(alpha=alpha, gamma=gamma, balanced=balanced)

    def forward(self, logits, targets):
        prop = torch.clamp(targets @ self.ancestor_matrix, 0, 1)
        return self.focal(logits, torch.clamp(targets + prop, 0, 1))


def build_ancestor_matrix(go_terms, ontology):
    """M[i, j] = 1 iff go_terms[j] is an ancestor of go_terms[i]."""
    go_map = {g: i for i, g in enumerate(go_terms)}
    n = len(go_terms)
    m = np.zeros((n, n), dtype=np.float32)
    for i, go in enumerate(go_terms):
        if go not in ontology:
            continue
        for anc in ontology[go].get("ancestors", []):
            if anc in go_map and go_map[anc] != i:
                m[i, go_map[anc]] = 1.0
    return torch.from_numpy(m)


def check_labels_propagated(y, ancestor_matrix):
    """Return the number of label entries the hierarchy would add."""
    y_t = torch.as_tensor(y, dtype=torch.float32)
    prop = torch.clamp(y_t @ ancestor_matrix.cpu(), 0, 1)
    added = float(torch.clamp(prop - y_t, min=0).sum())
    total = float(y_t.sum())
    print(f"hierarchical check: labels={total:.0f}, "
          f"ancestors the DAG would add={added:.0f} "
          f"({100 * added / max(total, 1):.2f}%)")
    if added == 0:
        print("  -> labels are already ancestor-closed; "
              "HierarchicalFocalLoss is a NO-OP here.")
    return added
