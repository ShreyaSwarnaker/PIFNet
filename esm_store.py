"""
Single-file storage for ESM residue embeddings.

Replaces the one-.pt-per-protein scheme. For 70k proteins that was 70k files,
which is slow to copy, slow to `rsync`, and on many cluster filesystems slow to
even `ls`. Here each split becomes exactly two files:

    {split}.dat        float16 memmap, shape [total_residues, esm_dim]
    {split}.index.npz  pids, offsets, lengths, dim, done-flags

Residues are stored contiguously and looked up by (offset, length), so a
protein's block is one slice. The .dat is opened with np.memmap, so the OS
pages in only what each batch touches -- RAM usage stays flat no matter how
large the file gets, and DataLoader workers share the same page cache.

SIZE WARNING. Residue-level embeddings are inherently large:

    total_residues x esm_dim x 2 bytes

For ~55M residues at 640 dims that is ~70 GB per split-set. This is the same
volume the .pt files occupied; it is simply in two files instead of 70,000.
Run `estimate_size()` before starting so the number is not a surprise.
"""
import json
import os

import numpy as np


class ESMStore:
    """Read-only accessor for a built store."""

    def __init__(self, directory, split):
        self.dir = directory
        self.split = split
        idx_path = os.path.join(directory, f"{split}.index.npz")
        if not os.path.exists(idx_path):
            raise FileNotFoundError(
                f"{idx_path} not found. Run esm_features.py first."
            )
        idx = np.load(idx_path, allow_pickle=True)
        self.pids = [str(p) for p in idx["pids"]]
        self.offsets = idx["offsets"].astype(np.int64)
        self.lengths = idx["lengths"].astype(np.int64)
        self.dim = int(idx["dim"])
        self.done = idx["done"].astype(bool)
        self.pid_to_row = {p: i for i, p in enumerate(self.pids)}

        n_missing = int((~self.done).sum())
        if n_missing:
            raise RuntimeError(
                f"{split}: {n_missing} proteins were never embedded. "
                "Re-run esm_features.py to finish the store."
            )

        self._data = None
        self._path = os.path.join(directory, f"{split}.dat")
        self._total = int(self.offsets[-1] + self.lengths[-1]) if len(self.pids) else 0

    @property
    def data(self):
        # Opened lazily and per-process, so DataLoader workers each get their
        # own memmap handle rather than inheriting one across a fork.
        if self._data is None:
            self._data = np.memmap(self._path, dtype=np.float16, mode="r",
                                   shape=(self._total, self.dim))
        return self._data

    def __contains__(self, pid):
        return pid in self.pid_to_row

    def __len__(self):
        return len(self.pids)

    def get(self, pid):
        """Residue embeddings for one protein, shape [L, dim], float16."""
        i = self.pid_to_row[pid]
        o, l = self.offsets[i], self.lengths[i]
        return np.asarray(self.data[o:o + l])

    def length_of(self, pid):
        return int(self.lengths[self.pid_to_row[pid]])

    def all_lengths(self, pids):
        return [int(self.lengths[self.pid_to_row[p]]) for p in pids]


class ESMStoreWriter:
    """
    Builds a store. Offsets are computed up front from sequence lengths, so the
    file is allocated once and filled in place -- no appending, no rewriting,
    and a crash only loses the proteins not yet marked done.
    """

    def __init__(self, directory, split, pids, seq_lengths, dim):
        os.makedirs(directory, exist_ok=True)
        self.dir = directory
        self.split = split
        self.dim = int(dim)
        self.idx_path = os.path.join(directory, f"{split}.index.npz")
        self.dat_path = os.path.join(directory, f"{split}.dat")

        pids = [str(p) for p in pids]
        lengths = np.asarray(seq_lengths, dtype=np.int64)
        offsets = np.zeros(len(pids), dtype=np.int64)
        if len(pids) > 1:
            np.cumsum(lengths[:-1], out=offsets[1:])
        total = int(lengths.sum())

        resume = False
        if os.path.exists(self.idx_path) and os.path.exists(self.dat_path):
            old = np.load(self.idx_path, allow_pickle=True)
            same = (list(map(str, old["pids"])) == pids
                    and int(old["dim"]) == self.dim
                    and np.array_equal(old["lengths"].astype(np.int64), lengths))
            if same:
                resume = True
                self.done = old["done"].astype(bool)
                print(f"{split}: resuming, {int(self.done.sum())}/{len(pids)} "
                      "already embedded")
            else:
                print(f"{split}: index does not match current inputs, rebuilding")

        self.pids = pids
        self.lengths = lengths
        self.offsets = offsets
        self.total = total
        self.pid_to_row = {p: i for i, p in enumerate(pids)}
        if not resume:
            self.done = np.zeros(len(pids), dtype=bool)

        mode = "r+" if (resume and os.path.exists(self.dat_path)) else "w+"
        self.data = np.memmap(self.dat_path, dtype=np.float16, mode=mode,
                              shape=(max(total, 1), self.dim))
        if not resume:
            self._flush_index()

    def pending(self):
        return [p for p, d in zip(self.pids, self.done) if not d]

    def write(self, pid, arr):
        i = self.pid_to_row[pid]
        l = int(self.lengths[i])
        arr = np.asarray(arr, dtype=np.float16)
        if arr.shape[0] != l:
            # Trust the stored layout; a mismatch means the sequence changed.
            raise ValueError(
                f"{pid}: expected {l} residues, model returned {arr.shape[0]}"
            )
        o = int(self.offsets[i])
        self.data[o:o + l] = arr
        self.done[i] = True

    def _flush_index(self):
        np.savez(
            self.idx_path,
            pids=np.array(self.pids, dtype=object),
            offsets=self.offsets, lengths=self.lengths,
            dim=self.dim, done=self.done,
        )

    def checkpoint(self):
        self.data.flush()
        self._flush_index()

    def close(self):
        self.checkpoint()
        del self.data
        n = int(self.done.sum())
        gb = self.total * self.dim * 2 / 1e9
        print(f"{self.split}: {n}/{len(self.pids)} proteins, "
              f"{self.total:,} residues, {gb:.1f} GB")


def estimate_size(seq_lengths, dim=640, max_len=1022):
    total = int(sum(min(l, max_len) for l in seq_lengths))
    gb = total * dim * 2 / 1e9
    return total, gb
