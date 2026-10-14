"""
Dataset loading, label handling, and the torch Dataset.

Key fix vs the notebook: ESM features are addressed by protein ID, not by
positional index. In the notebook, precompute_esm_embeddings() caught OOM and
*skipped* the batch, which silently shifted every later prot_{i}.pt file by one
-- permanently misaligning sequences and labels with no error. ID-keyed lookup
makes that class of bug impossible.

Labels are held as uint8, not float32. MSNGO's label sets are far larger than
CAFA-5's 499 terms (MFO 6445, BPO 19323), and at ~50k proteins a float32 BPO
matrix is ~3.9 GB per split before a single batch is drawn. uint8 costs a
quarter of that and is cast per sample.
"""
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from esm_store import ESMStore


def load_split(cfg, split: str) -> pd.DataFrame:
    """Reads .parquet in preference to .csv. With MSNGO's 6445 (MFO) or 19323
    (BPO) label columns a wide CSV is enormous and slow to parse; parquet
    stores the same table far smaller and loads in seconds."""
    base = cfg.paths.split_csv(cfg.branch, split)
    parquet = base.replace(".csv", ".parquet")
    if os.path.exists(parquet):
        df = pd.read_parquet(parquet)
    elif os.path.exists(base):
        df = pd.read_csv(base)
    else:
        raise FileNotFoundError(
            f"neither {parquet} nor {base} exists. "
            "Run msngo convert first."
        )
    used_path = parquet if os.path.exists(parquet) else base
    assert "ID" in df.columns and "sequence" in df.columns, \
        f"{used_path} must have 'ID' and 'sequence' columns"
    if df["ID"].duplicated().any():
        n = int(df["ID"].duplicated().sum())
        raise ValueError(f"{used_path} has {n} duplicate IDs; de-duplicate first")
    return df


def load_all_splits(cfg):
    train = load_split(cfg, "train")
    val = load_split(cfg, "val")
    test = load_split(cfg, "test")

    go_terms = np.array(train.columns[2:])
    for name, df in [("val", val), ("test", test)]:
        other = np.array(df.columns[2:])
        if not np.array_equal(go_terms, other):
            raise ValueError(
                f"GO term columns differ between train and {name}. "
                "All splits must share identical label columns in identical order."
            )
    return train, val, test, go_terms
    


#def get_labels(df, go_terms) -> np.ndarray:
#    """Label matrix aligned to go_terms order, uint8 to bound memory."""
#    y = df[list(go_terms)].values.astype(np.uint8)
#    assert y.shape[1] == len(go_terms)
#    return y
    
    
def get_labels(df, go_terms) -> np.ndarray:
    """Label matrix aligned to go_terms order, uint8 to bound memory."""
    cols = df.columns[2:]
    if len(cols) == len(go_terms) and (np.asarray(cols) == np.asarray(go_terms)).all():
        y = df.iloc[:, 2:].to_numpy(dtype=np.uint8)      # no intermediate frame
    else:
        y = df[list(go_terms)].to_numpy(dtype=np.uint8)  # fallback
    assert y.shape[1] == len(go_terms)
    return y


def load_ic(cfg, go_terms):
    """Information content per GO term, aligned to go_terms."""
    ic_df = pd.read_csv(cfg.paths.ic_csv(cfg.branch))
    term_col = "terms" if "terms" in ic_df.columns else ic_df.columns[0]
    ic_col = "IC" if "IC" in ic_df.columns else ic_df.columns[1]
    go_to_ic = dict(zip(ic_df[term_col], ic_df[ic_col]))
    ic = np.array([go_to_ic.get(g, np.nan) for g in go_terms], dtype=float)
    return ic, go_to_ic


def build_seq_dict(*dfs):
    d = {}
    for df in dfs:
        d.update(dict(zip(df["ID"], df["sequence"])))
    return d


class ESMPPIDataset(Dataset):
    """
    Returns (x_seq, mask, x_ppi, y).

    x_seq : [esm_dim, max_len] float32, zero-padded on the right
    mask  : [max_len] bool, True on real residues -- used to mask attention
            pooling. The notebook had no mask, so padded positions received
            softmax weight and diluted every short protein's representation.
    """

    def __init__(self, protein_ids, esm_dir, split, ppi_matrix, labels,
                 max_len=1024, esm_dim=640):
        self.ids = list(protein_ids)
        self.esm_dir = esm_dir
        self.split = split
        self.max_len = max_len
        self.esm_dim = esm_dim
        self.ppi = torch.as_tensor(np.asarray(ppi_matrix), dtype=torch.float32)
        self.labels = torch.as_tensor(np.asarray(labels, dtype=np.uint8))

        self.store = ESMStore(esm_dir, split)
        missing = [p for p in self.ids if p not in self.store]
        if missing:
            raise KeyError(
                f"{len(missing)} proteins are absent from the ESM store "
                f"(first few: {missing[:5]}). Re-run esm_features.py."
            )
        assert len(self.ids) == len(self.labels) == len(self.ppi)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        pid = self.ids[idx]
        emb = torch.from_numpy(np.array(self.store.get(pid))).float()
        L = min(emb.shape[0], self.max_len)
        # Variable length: collate_variable_length pads to the batch maximum,
        # not to max_len. Median proteins are ~350 residues, so padding every
        # one to 1024 spent roughly 3x the FLOPs for nothing.
        return emb[:L], self.ppi[idx], self.labels[idx].float()

    def lengths(self):
        """Residue counts, read straight from the index -- no file I/O."""
        return self.store.all_lengths(self.ids)


def collate_variable_length(batch):
    """Pad to the longest sequence in this batch and build the mask."""
    embs, ppis, labels = zip(*batch)
    B = len(embs)
    L = max(e.shape[0] for e in embs)
    D = embs[0].shape[1]

    x = torch.zeros(B, L, D)
    mask = torch.zeros(B, L, dtype=torch.bool)
    for i, e in enumerate(embs):
        x[i, :e.shape[0]] = e
        mask[i, :e.shape[0]] = True

    return x.transpose(1, 2), mask, torch.stack(ppis), torch.stack(labels)


class LengthGroupedSampler(torch.utils.data.Sampler):
    """
    Shuffle, then form batches from length-similar proteins so that padding
    inside a batch stays small. Batch order is shuffled each epoch, so this
    keeps stochasticity while cutting wasted compute.
    """

    def __init__(self, lengths, batch_size, shuffle=True, seed=0):
        self.lengths = np.asarray(lengths)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.epoch = 0
        self.seed = seed

    def __iter__(self):
        g = np.random.default_rng(self.seed + self.epoch)
        idx = np.arange(len(self.lengths))
        if self.shuffle:
            g.shuffle(idx)
        # megabatch sort: local ordering by length, global randomness preserved
        mega = self.batch_size * 50
        chunks = [idx[i:i + mega] for i in range(0, len(idx), mega)]
        idx = np.concatenate([c[np.argsort(self.lengths[c])] for c in chunks])
        batches = [idx[i:i + self.batch_size]
                   for i in range(0, len(idx), self.batch_size)]
        if self.shuffle:
            g.shuffle(batches)
        self.epoch += 1
        return iter(np.concatenate(batches).tolist())

    def __len__(self):
        return len(self.lengths)


def make_loaders(cfg, datasets, length_grouped=True):
    from torch.utils.data import DataLoader
    train_ds, val_ds, test_ds = datasets
    nw = cfg.train.num_workers
    common = dict(num_workers=nw, pin_memory=True,
                  collate_fn=collate_variable_length)
    if nw > 0:
        # persistent_workers/prefetch_factor are only valid when num_workers>0
        common["persistent_workers"] = cfg.train.persistent_workers
        common["prefetch_factor"] = cfg.train.prefetch_factor

    train_sampler = None
    if length_grouped:
        print("computing sequence lengths for grouped batching...")
        train_sampler = LengthGroupedSampler(
            train_ds.lengths(), cfg.train.batch_size, True, cfg.train.seed)

    return (
        DataLoader(train_ds, batch_size=cfg.train.batch_size,
                   shuffle=(train_sampler is None), sampler=train_sampler,
                   drop_last=False, **common),
        DataLoader(val_ds, batch_size=cfg.train.eval_batch_size, shuffle=False,
                   **common),
        DataLoader(test_ds, batch_size=cfg.train.eval_batch_size, shuffle=False,
                   **common),
    )
