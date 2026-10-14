"""
Precompute frozen ESM-2 residue embeddings into a single-file store.

  python -m esm_features --branch mf --split all
  python -m esm_features --branch mf --split all --estimate-only

Storage is now two files per split (see esm_store.py) instead of one .pt per
protein. Everything else about the original notebook's failure modes is still
fixed here:
  * addressed by protein ID, never by loop index
  * OOM retries the batch one sequence at a time rather than skipping it
  * length-sorted batching, so short proteins are not padded to 1022
  * resume is per-protein
"""
import argparse
import os

import numpy as np
import torch
from tqdm import tqdm

from esm_store import ESMStoreWriter, estimate_size


def load_esm(cfg, device):
    import esm
    fn = getattr(esm.pretrained, cfg.model.esm_model)
    model, alphabet = fn()
    model = model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, alphabet, alphabet.get_batch_converter()


@torch.no_grad()
def embed_batch(model, converter, seqs, layer, device):
    data = [(f"p{i}", s) for i, s in enumerate(seqs)]
    _, _, tokens = converter(data)
    tokens = tokens.to(device)
    out = model(tokens=tokens, repr_layers=[layer])
    reps = out["representations"][layer]
    result = [reps[i, 1:len(s) + 1].float().cpu().numpy().astype(np.float16)
              for i, s in enumerate(seqs)]
    del out, reps, tokens
    return result


def precompute(cfg, pids, seqs, out_dir, split, device, batch_tokens=16384,
               checkpoint_every=2000):
    max_res = cfg.model.max_len - 2
    seqs = [s[:max_res] for s in seqs]
    lengths = [len(s) for s in seqs]

    total, gb = estimate_size(lengths, cfg.model.esm_dim, max_res)
    print(f"{split}: {len(pids)} proteins, {total:,} residues -> {gb:.1f} GB")

    writer = ESMStoreWriter(out_dir, split, pids, lengths, cfg.model.esm_dim)
    seq_of = dict(zip(pids, seqs))
    todo = writer.pending()
    if not todo:
        print(f"{split}: already complete")
        writer.close()
        return

    model, alphabet, converter = load_esm(cfg, device)
    todo.sort(key=lambda p: len(seq_of[p]))

    batch, batch_ids, since_ckpt = [], [], 0
    pbar = tqdm(total=len(todo), desc=split)

    def flush():
        nonlocal batch, batch_ids, since_ckpt
        if not batch:
            return
        try:
            embs = embed_batch(model, converter, batch, cfg.model.esm_layer,
                               device)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            embs = []
            for s in batch:
                embs.extend(embed_batch(model, converter, [s],
                                        cfg.model.esm_layer, device))
        for pid, emb in zip(batch_ids, embs):
            writer.write(pid, emb)
        pbar.update(len(batch))
        since_ckpt += len(batch)
        if since_ckpt >= checkpoint_every:
            writer.checkpoint()
            since_ckpt = 0
        batch, batch_ids = [], []

    for pid in todo:
        s = seq_of[pid]
        longest = max([len(x) for x in batch] + [len(s)])
        if batch and (len(batch) + 1) * longest > batch_tokens:
            flush()
        batch.append(s)
        batch_ids.append(pid)
    flush()
    pbar.close()
    writer.close()


def main():
    from config import get_config
    from data import load_split

    ap = argparse.ArgumentParser()
    ap.add_argument("--branch", default="mf")
    ap.add_argument("--split", default="all",
                    choices=["train", "val", "test", "all"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-tokens", type=int, default=16384)
    ap.add_argument("--estimate-only", action="store_true",
                    help="report disk requirement and exit")
    args = ap.parse_args()

    cfg = get_config(branch=args.branch)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    splits = ["train", "val", "test"] if args.split == "all" else [args.split]
    out_dir = os.path.join(cfg.paths.work_root, "esm", cfg.model.esm_model)

    if args.estimate_only:
        grand = 0.0
        for split in splits:
            df = load_split(cfg, split)
            lens = df["sequence"].str.len().tolist()
            total, gb = estimate_size(lens, cfg.model.esm_dim,
                                      cfg.model.max_len - 2)
            print(f"{split:6s} {len(df):>8,} proteins  {total:>14,} residues  "
                  f"{gb:8.1f} GB")
            grand += gb
        print(f"{'TOTAL':6s} {'':>8} {'':>14}  {grand:8.1f} GB")
        return

    for split in splits:
        df = load_split(cfg, split)
        precompute(cfg, df["ID"].tolist(), df["sequence"].tolist(),
                   out_dir, split, device, args.batch_tokens)


if __name__ == "__main__":
    main()
