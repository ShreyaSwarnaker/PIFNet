#"""
#Train and evaluate.
#
#  python -m train --branch mf --fusion gate
#  python -m train --branch mf --fusion seq_only    # THE ablation
#  python -m train --branch mf --fusion ppi_only
#  python -m train --branch mf --fusion concat
#  python -m train --branch mf --fusion crossattn
#
#Everything is seeded and every run writes results.json + predictions.npz to
#work/runs/<tag>/, so ablations are directly comparable.
#
#Additions vs the notebook: seeding, mixed precision, a cosine LR schedule,
#gradient clipping, restoring best weights by validation Fmax rather than
#validation loss (loss is a poor proxy under focal loss with extreme
#imbalance), and per-degree evaluation.
#
#THIS REVISION -- three changes, all aimed at the silent stalls:
#
#1. Validation used to make THREE passes over the val loader per epoch: a
#   run_epoch() pass for the loss, then a predict() pass for the probabilities.
#   Together with the train pass that is three DataLoader worker-spawn cycles
#   per epoch. They are now fused into one pass (`evaluate_pass`) that returns
#   loss and probabilities together. The probabilities are bit-identical to the
#   old predict() (still full fp32, no autocast); only the logged val_loss
#   changes slightly, and it is never used for any decision.
#
#2. Every print is flushed, and each startup phase is timed. An empty log now
#   means "did not reach the first phase", not "output is sitting in an 8 KB
#   buffer". Run with `python -u` as well.
#
#3. `if args.x:` guards became `if args.x is not None:` so that a legitimate
#   zero (e.g. --dropout 0) is not silently ignored.
#"""
#import argparse
#import json
#import os
#import random
#import time
#
#import numpy as np
#import torch
#from tqdm import tqdm
#
#from config import get_config
#from data import (ESMPPIDataset, load_all_splits, get_labels, load_ic,
#                   make_loaders)
#from evaluate import evaluate_cafa, evaluate_by_degree, load_ontology, macro_auprc
#from losses import (FocalLoss, HierarchicalFocalLoss, build_ancestor_matrix,
#                     check_labels_propagated)
#from models import LightProteinFusion, count_parameters
#
#
#_T0 = time.time()
#
#
#def log(msg):
#    """Timestamped, flushed. The timestamp is what tells you WHERE a hang is."""
#    print(f"[+{time.time() - _T0:7.1f}s] {msg}", flush=True)
#
#
#def set_seed(seed):
#    random.seed(seed)
#    np.random.seed(seed)
#    torch.manual_seed(seed)
#    torch.cuda.manual_seed_all(seed)
#
#
#def load_ppi_features(cfg, df, split, method, completion):
#    path = cfg.paths.work("embed",
#                          f"{split}_{method}_{completion}_{cfg.branch}.pt")
#    if not os.path.exists(path):
#        # Pre-fix runs wrote embeddings without the completion mode in the
#        # filename, so every completion variant silently overwrote the last
#        # one on disk. If only the OLD unsuffixed file exists, fall back to
#        # it -- but only for the default (triadic) completion, since that is
#        # the only mode the old filename could possibly correspond to. Any
#        # other completion mode MUST be rebuilt; there is no way to recover
#        # which mode the stale file was actually built from.
#        legacy = cfg.paths.work("embed", f"{split}_{method}_{cfg.branch}.pt")
#        if completion == "triadic" and os.path.exists(legacy):
#            print(f"note: using pre-fix embedding file {legacy} "
#                  "(assumed triadic; rebuild with ppi_embed.py to be sure)",
#                  flush=True)
#            path = legacy
#        elif os.path.exists(legacy):
#            raise FileNotFoundError(
#                f"{path} not found, and the legacy file {legacy} exists but "
#                f"cannot be trusted to be '{completion}' completion -- it "
#                "could be from any prior run. Rebuild explicitly:\n"
#                f"  python -m ppi_graph --branch {cfg.branch} --completion {completion}\n"
#                f"  python -m ppi_embed --branch {cfg.branch} --completion {completion} "
#                f"--method {method}"
#            )
#    d = torch.load(path, weights_only=False)
#    missing = [p for p in df["ID"] if p not in d]
#    if missing:
#        raise KeyError(f"{len(missing)} IDs missing from {path} "
#                       f"(e.g. {missing[:3]})")
#    return np.stack([d[p].numpy() for p in df["ID"]])
#
#
#def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None,
#              accum=1, desc="train"):
#    """Training pass. (Validation now goes through evaluate_pass instead.)"""
#    train = optimizer is not None
#    model.train(train)
#    total, nb = 0.0, 0
#    gates = []
#    if train:
#        optimizer.zero_grad(set_to_none=True)
#
#    ctx = torch.enable_grad() if train else torch.no_grad()
#    with ctx:
#        for i, (x_seq, mask, x_ppi, y) in enumerate(
#                tqdm(loader, desc=desc, leave=False)):
#            x_seq = x_seq.to(device, non_blocking=True)
#            mask = mask.to(device, non_blocking=True)
#            x_ppi = x_ppi.to(device, non_blocking=True)
#            y = y.to(device, non_blocking=True)
#
#            with torch.autocast("cuda", enabled=scaler is not None):
#                logits, g = model(x_seq, mask, x_ppi)
#                loss = criterion(logits, y)
#
#            if train:
#                scaled = loss / accum
#                if scaler is not None:
#                    scaler.scale(scaled).backward()
#                else:
#                    scaled.backward()
#                if (i + 1) % accum == 0 or (i + 1) == len(loader):
#                    if scaler is not None:
#                        scaler.unscale_(optimizer)
#                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
#                    if scaler is not None:
#                        scaler.step(optimizer); scaler.update()
#                    else:
#                        optimizer.step()
#                    optimizer.zero_grad(set_to_none=True)
#            else:
#                gates.append(g.float().cpu())
#
#            total += loss.item()
#            nb += 1
#
#    g = torch.cat(gates).numpy() if gates else None
#    return total / max(nb, 1), g
#
#
#@torch.no_grad()
#def evaluate_pass(model, loader, device, criterion=None, desc="predict"):
#    """
#    One fp32 forward pass over `loader`, returning probabilities, labels,
#    gates and (optionally) the mean loss.
#
#    This replaces the old run_epoch(val) + predict(val) pair. The
#    probabilities are computed exactly as predict() computed them -- full
#    fp32, no autocast -- so checkpoint selection and the final reported
#    metrics are unchanged. Only the logged loss differs (fp32 rather than
#    autocast fp16), and the loss is never used for selection or early
#    stopping.
#    """
#    model.eval()
#    probs, trues, gates = [], [], []
#    total, nb = 0.0, 0
#
#    for x_seq, mask, x_ppi, y in tqdm(loader, desc=desc, leave=False):
#        x_seq = x_seq.to(device, non_blocking=True)
#        mask = mask.to(device, non_blocking=True)
#        x_ppi = x_ppi.to(device, non_blocking=True)
#
#        logits, g = model(x_seq, mask, x_ppi)
#
#        if criterion is not None:
#            total += criterion(logits, y.to(device, non_blocking=True)).item()
#            nb += 1
#
#        probs.append(torch.sigmoid(logits.float()).cpu().numpy())
#        trues.append(y.numpy())
#        gates.append(g.float().cpu().numpy())
#
#    loss = total / max(nb, 1) if criterion is not None else float("nan")
#    return np.vstack(probs), np.vstack(trues), np.vstack(gates), loss
#
#
#@torch.no_grad()
#def predict(model, loader, device):
#    """Kept for external callers; delegates to evaluate_pass."""
#    probs, trues, gates, _ = evaluate_pass(model, loader, device)
#    return probs, trues, gates
#
#
#def main():
#    ap = argparse.ArgumentParser()
#    ap.add_argument("--branch", default="mf")
#    ap.add_argument("--fusion", default="gate",
#                    choices=["gate", "concat", "crossattn", "seq_only",
#                             "ppi_only"])
#    ap.add_argument("--ppi-method", default="deepwalk")
#    ap.add_argument("--completion", default="triadic")
#    ap.add_argument("--epochs", type=int, default=None)
#    ap.add_argument("--seed", type=int, default=42)
#    ap.add_argument("--device", default="cuda")
#    ap.add_argument("--tag", default=None)
#    ap.add_argument("--tag-suffix", default=None,
#                    help="append to the auto-generated tag, e.g. 'v2', so "
#                         "improved runs never overwrite a baseline with the "
#                         "same fusion/seed")
#    ap.add_argument("--no-hier-loss", action="store_true")
#    ap.add_argument("--dropout", type=float, default=None,
#                    help="override classifier dropout (cfg default 0.3). "
#                         "Does NOT affect CrossModalAttention's dropout, "
#                         "which is hardcoded at 0.1 independently of this.")
#    ap.add_argument("--weight-decay", type=float, default=None,
#                    help="override Adam weight_decay (cfg default 0.0, i.e. "
#                         "off -- no ablation run so far has used this)")
#    ap.add_argument("--num-blocks", type=int, default=None)
#    ap.add_argument("--dilation-schedule", default=None,
#                    choices=["moderate", "aggressive"])
#    ap.add_argument("--focal-unbalanced", action="store_true",
#                    help="reproduce the ORIGINAL (buggy) scalar-alpha focal "
#                         "loss, for a like-for-like baseline")
#    ap.add_argument("--hidden-dim", type=int, default=None)
#    ap.add_argument("--focal-alpha", type=float, default=None)
#    ap.add_argument("--lr", type=float, default=None)
#    ap.add_argument("--num-workers", type=int, default=None,
#                    help="override the DataLoader worker count (default 2, "
#                         "safe for several concurrent jobs). Raise only if "
#                         "you are certain you are the sole job on the box. "
#                         "Use 0 to rule the DataLoader out when diagnosing "
#                         "a hang.")
#    ap.add_argument("--eval-threads", type=int, default=None,
#                    help="override evaluate_cafa's thread count (default 4, "
#                         "same concurrency caveat as --num-workers)")
#    args = ap.parse_args()
#
#    cfg = get_config(branch=args.branch)
#    cfg.model.fusion = args.fusion
#    cfg.train.seed = args.seed
#    if args.epochs is not None:
#        cfg.train.num_epochs = args.epochs
#    if args.num_blocks is not None:
#        cfg.model.num_blocks = args.num_blocks
#    if args.dilation_schedule is not None:
#        cfg.model.dilation_schedule = args.dilation_schedule
#    if args.focal_unbalanced:
#        cfg.train.focal_balanced = False
#    if args.hidden_dim is not None:
#        cfg.model.hidden_dim = args.hidden_dim
#    if args.focal_alpha is not None:
#        cfg.train.focal_alpha = args.focal_alpha
#    if args.lr is not None:
#        cfg.train.lr = args.lr
#    if args.num_workers is not None:
#        cfg.train.num_workers = args.num_workers
#    if args.eval_threads is not None:
#        cfg.train.eval_threads = args.eval_threads
#    if args.dropout is not None:
#        cfg.model.dropout = args.dropout
#    if args.weight_decay is not None:
#        cfg.train.weight_decay = args.weight_decay
#
#    tag = args.tag or (f"{args.branch}_{args.fusion}_{args.ppi_method}"
#                       f"_{args.completion}_s{args.seed}"
#                       if args.completion != "triadic"
#                       else f"{args.branch}_{args.fusion}_{args.ppi_method}_s{args.seed}")
#    if args.tag_suffix and not args.tag:
#        tag += f"_{args.tag_suffix}"
#    run_dir = cfg.paths.work("runs", tag, "_")[:-1]
#    os.makedirs(run_dir, exist_ok=True)
#
#    log(f"start pid={os.getpid()} tag={tag}")
#    log(f"threads: OMP={os.environ.get('OMP_NUM_THREADS', 'unset')} "
#        f"MKL={os.environ.get('MKL_NUM_THREADS', 'unset')} "
#        f"torch={torch.get_num_threads()}")
#
#    set_seed(cfg.train.seed)
#    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
#    if device.type == "cpu":
#        print("*** WARNING: running on CPU. Training will be 10-50x slower "
#              "than on GPU. If you expected CUDA, check `nvidia-smi` and "
#              "`python -c \"import torch; print(torch.cuda.is_available())\"` "
#              "before assuming the pipeline itself got slower. ***", flush=True)
#    else:
#        log(f"device: {device} ({torch.cuda.get_device_name(device)})")
#
#    # ---- data -------------------------------------------------------------
#    train_df, val_df, test_df, go_terms = load_all_splits(cfg)
#    log(f"splits loaded: {len(train_df)}/{len(val_df)}/{len(test_df)}, "
#        f"{len(go_terms)} terms")
#
#    y_tr = get_labels(train_df, go_terms)
#    y_va = get_labels(val_df, go_terms)
#    y_te = get_labels(test_df, go_terms)
#    ic, go_to_ic = load_ic(cfg, go_terms)
#    log("labels + IC loaded")
#
#    ppi = {s: load_ppi_features(cfg, df, s, args.ppi_method, args.completion)
#           for s, df in [("train", train_df), ("val", val_df),
#                         ("test", test_df)]}
#    log(f"PPI features loaded (dim={ppi['train'].shape[1]})")
#
#    esm_dir = os.path.join(cfg.paths.work_root, "esm", cfg.model.esm_model)
#
#    datasets = [
#        ESMPPIDataset(df["ID"], esm_dir, s, ppi[s], y,
#                      cfg.model.max_len, cfg.model.esm_dim)
#        for s, df, y in [("train", train_df, y_tr), ("val", val_df, y_va),
#                         ("test", test_df, y_te)]
#    ]
#    train_loader, val_loader, test_loader = make_loaders(cfg, datasets)
#    log(f"loaders built (num_workers={cfg.train.num_workers})")
#
#    # ---- ontology ---------------------------------------------------------
#    ontology = load_ontology(cfg.paths.ontology, cfg.namespace)
#    valid = [i for i, g in enumerate(go_terms)
#             if g in ontology and g in go_to_ic and np.isfinite(ic[i])]
#    log(f"labels: {len(go_terms)} total, {len(valid)} usable for IC metrics")
#    go_valid = go_terms[valid]
#    ic_valid = ic[valid]
#
#    # ---- model ------------------------------------------------------------
#    model = LightProteinFusion(
#        num_classes=len(go_terms), esm_dim=cfg.model.esm_dim,
#        ppi_dim=ppi["train"].shape[1], hidden_dim=cfg.model.hidden_dim,
#        dropout=cfg.model.dropout, fusion=cfg.model.fusion,
#        num_blocks=cfg.model.num_blocks,
#        dilation_schedule=cfg.model.dilation_schedule,
#    ).to(device)
#    log(f"trainable parameters: {count_parameters(model):,}")
#    if cfg.model.fusion != "ppi_only":
#        print(f"encoder theoretical receptive field: "
#              f"{model.encoder.receptive_field} residues "
#              f"({cfg.model.num_blocks} blocks, {cfg.model.dilation_schedule})",
#              flush=True)
#
##    anc = build_ancestor_matrix(go_terms, ontology)
##    added = check_labels_propagated(y_tr, anc)
##    use_hier = cfg.train.hierarchical_loss and not args.no_hier_loss and added > 0
#    
#    
#    if cfg.train.hierarchical_loss and not args.no_hier_loss:
#        anc = build_ancestor_matrix(go_terms, ontology)
#        added = check_labels_propagated(y_tr, anc)
#        use_hier = added > 0
#    else:
#        anc, use_hier = None, False
#        log("hierarchical loss disabled; skipping ancestor matrix")
#    
#    
#    
#    
#    criterion = (HierarchicalFocalLoss(anc.to(device), cfg.train.focal_alpha,
#                                       cfg.train.focal_gamma,
#                                       cfg.train.focal_balanced).to(device)
#                 if use_hier
#                 else FocalLoss(cfg.train.focal_alpha, cfg.train.focal_gamma,
#                                balanced=cfg.train.focal_balanced))
#    print(f"loss: {'hierarchical ' if use_hier else ''}focal "
#          f"(alpha={cfg.train.focal_alpha}, "
#          f"balanced={cfg.train.focal_balanced})", flush=True)
#    print(f"batch={cfg.train.batch_size} x accum={cfg.train.accumulation_steps} "
#          f"= effective {cfg.train.batch_size * cfg.train.accumulation_steps}",
#          flush=True)
#
#    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr,
#                                 weight_decay=cfg.train.weight_decay,
#                                 amsgrad=True)
#    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#        optimizer, T_max=cfg.train.num_epochs)
#    scaler = torch.amp.GradScaler("cuda") if (cfg.train.amp and
#                                              device.type == "cuda") else None
#
#    log("setup complete, entering training loop")
#
#    # ---- train ------------------------------------------------------------
#    best_fmax, best_state, bad = -1.0, None, 0
#    history = []
#    for epoch in range(cfg.train.num_epochs):
#        t_ep = time.time()
#
#        tr_loss, _ = run_epoch(model, train_loader, criterion, device,
#                               optimizer, scaler, cfg.train.accumulation_steps,
#                               f"epoch {epoch + 1} train")
#        t_tr = time.time() - t_ep
#
#        # ONE val pass: loss + probabilities together.
#        vp, vt, _, va_loss = evaluate_pass(model, val_loader, device,
#                                           criterion,
#                                           desc=f"epoch {epoch + 1} val")
#        t_va = time.time() - t_ep - t_tr
#
#        # Coarse threshold grid for per-epoch checkpoint selection only.
#        # The full 101-point CAFA grid costs ~4x more and picks the same
#        # epoch; final val/test numbers below use the standard grid.
#        va = evaluate_cafa(
#            vt[:, valid],
#            vp[:, valid],
#            go_valid,
#            ontology,
#            ic_valid,
#            verbose=False,
#            n_jobs=cfg.train.eval_threads,
#            n_thresholds=cfg.train.eval_thresholds_train
#        )
#        t_ev = time.time() - t_ep - t_tr - t_va
#
#        scheduler.step()
#
#        print(f"[{epoch + 1}/{cfg.train.num_epochs}] "
#              f"train {tr_loss:.5f} | val {va_loss:.5f} | "
#              f"val Fmax {va['Fmax']:.4f} | "
#              f"{t_tr:.0f}s train / {t_va:.0f}s val / {t_ev:.0f}s eval",
#              flush=True)
#        history.append({"epoch": epoch + 1, "train_loss": tr_loss,
#                        "val_loss": va_loss,
#                        "t_train": t_tr, "t_val": t_va, "t_eval": t_ev,
#                        **va})
#
#        # Select on Fmax, not loss: focal loss on a 499-way problem where
#        # 99% of entries are zero keeps falling while ranking quality plateaus.
##        if va["Fmax"] > best_fmax:
##            best_fmax, bad = va["Fmax"], 0
##            best_state = {k: v.detach().cpu().clone()
##                          for k, v in model.state_dict().items()}
#                          
#                          
#        if va["Fmax"] > best_fmax:
#            best_fmax, bad = va["Fmax"], 0
#            best_state = {k: v.detach().cpu().clone()
#                          for k, v in model.state_dict().items()}
#            torch.save({"epoch": epoch + 1, "fmax": best_fmax,
#                        "state": best_state},
#                       os.path.join(run_dir, "best.pt"))
#        else:
#            bad += 1
#            if bad >= cfg.train.patience:
#                print("early stopping", flush=True)
#                break
#
#    if best_state:
#        model.load_state_dict(best_state)
#        torch.save(best_state, os.path.join(run_dir, "model.pt"))
#
#    # ---- evaluate ---------------------------------------------------------
#    results = {"tag": tag, "fusion": args.fusion, "seed": args.seed,
#               "params": count_parameters(model), "history": history}
#
#    for name, loader, df in [("val", val_loader, val_df),
#                             ("test", test_loader, test_df)]:
#        log(f"final evaluation: {name}")
#        probs, true, gates, _ = evaluate_pass(model, loader, device,
#                                              desc=f"final {name}")
#        yt, ys = true[:, valid], probs[:, valid]
#        
#        
##        r = evaluate_cafa(yt, ys, go_valid, ontology, ic_valid,
##                          verbose=False, n_jobs=cfg.train.eval_threads)
##        r["macro_AUPRC"] = macro_auprc(yt, ys)
##        results[name] = r
##        print(f"\n== {name} ==", flush=True)
##        for k, v in r.items():
##            print(f"  {k}: {v:.4f}", flush=True)
##        np.savez_compressed(os.path.join(run_dir, f"pred_{name}.npz"),
##                            y_score=ys, y_true=yt, go_terms=go_valid,
##                            ic=ic_valid, gates=gates,
##                            ids=df["ID"].values)
##                            
#        print(f"\nFINAL {name.upper()} evaluation started (101 thresholds)...", flush=True)
#        t_eval = time.time()              
#                            
#        r = evaluate_cafa(yt, ys, go_valid, ontology, ic_valid,
#                          verbose=False, n_jobs=cfg.train.eval_threads)
#                          
#        print(f"FINAL {name.upper()} evaluation finished in "
#          f"{time.time() - t_eval:.1f}s", flush=True)
#          
#        log(f"{name}: CAFA metrics done")
#        r["macro_AUPRC"] = macro_auprc(yt, ys)
#        log(f"{name}: macro AUPRC done")
#        results[name] = r
#        print(f"\n== {name} ==", flush=True)
#        for k, v in r.items():
#            print(f"  {k}: {v:.4f}", flush=True)
#        np.savez(os.path.join(run_dir, f"pred_{name}.npz"),
#                 y_score=ys, y_true=yt.astype(np.uint8), go_terms=go_valid,
#                 ic=ic_valid, gates=gates, ids=df["ID"].values)
#        log(f"{name}: predictions saved")
#        
#
#    with open(os.path.join(run_dir, "results.json"), "w") as f:
#        json.dump(results, f, indent=2, default=float)
#    log(f"saved to {run_dir}")
#
#
#if __name__ == "__main__":
#    main()



"""
Train and evaluate.

  python -m train --branch mf --fusion gate
  python -m train --branch mf --fusion seq_only    # THE ablation
  python -m train --branch mf --fusion ppi_only
  python -m train --branch mf --fusion concat
  python -m train --branch mf --fusion crossattn

Everything is seeded and every run writes results.json + predictions.npz to
work/runs/<tag>/, so ablations are directly comparable.

Additions vs the notebook: seeding, mixed precision, a cosine LR schedule,
gradient clipping, restoring best weights by validation Fmax rather than
validation loss (loss is a poor proxy under focal loss with extreme
imbalance), and per-degree evaluation.

THIS REVISION -- three changes, all aimed at the silent stalls:

1. Validation used to make THREE passes over the val loader per epoch: a
   run_epoch() pass for the loss, then a predict() pass for the probabilities.
   Together with the train pass that is three DataLoader worker-spawn cycles
   per epoch. They are now fused into one pass (`evaluate_pass`) that returns
   loss and probabilities together. The probabilities are bit-identical to the
   old predict() (still full fp32, no autocast); only the logged val_loss
   changes slightly, and it is never used for any decision.

2. Every print is flushed, and each startup phase is timed. An empty log now
   means "did not reach the first phase", not "output is sitting in an 8 KB
   buffer". Run with `python -u` as well.

3. `if args.x:` guards became `if args.x is not None:` so that a legitimate
   zero (e.g. --dropout 0) is not silently ignored.

4. Static split IDs, uint8 labels, GO terms, and IC are cached after the first
   run. Subsequent seeds load the cache instead of reconstructing the huge
   wide parquet DataFrames. The cache is invalidated when source files change.

5. Training-time CAFA evaluation runs every 5 epochs on an 11-threshold grid.
   Final validation/test CAFA evaluation remains the standard 101-threshold grid.

4. Static split IDs, uint8 labels, GO terms, and IC are cached after the first
   run. Subsequent seeds load the cache instead of reconstructing the huge
   wide parquet DataFrames. The cache is invalidated when source files change.

5. Training-time CAFA evaluation runs every 5 epochs on an 11-threshold grid.
   Final validation/test CAFA evaluation remains the standard 101-threshold grid.
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch
from tqdm import tqdm

from config import get_config
from data import (ESMPPIDataset, load_all_splits, get_labels, load_ic,
                   make_loaders)
from evaluate import evaluate_cafa, evaluate_by_degree, load_ontology, macro_auprc
from losses import (FocalLoss, HierarchicalFocalLoss, build_ancestor_matrix,
                     check_labels_propagated)
from models import LightProteinFusion, GOEmbeddingClassifier, count_parameters


_T0 = time.time()


def log(msg):
    """Timestamped, flushed. The timestamp is what tells you WHERE a hang is."""
    print(f"[+{time.time() - _T0:7.1f}s] {msg}", flush=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_ppi_features(cfg, ids, split, method, completion):
    path = cfg.paths.work("embed",
                          f"{split}_{method}_{completion}_{cfg.branch}.pt")
    if not os.path.exists(path):
        # Pre-fix runs wrote embeddings without the completion mode in the
        # filename, so every completion variant silently overwrote the last
        # one on disk. If only the OLD unsuffixed file exists, fall back to
        # it -- but only for the default (triadic) completion, since that is
        # the only mode the old filename could possibly correspond to. Any
        # other completion mode MUST be rebuilt; there is no way to recover
        # which mode the stale file was actually built from.
        legacy = cfg.paths.work("embed", f"{split}_{method}_{cfg.branch}.pt")
        if completion == "triadic" and os.path.exists(legacy):
            print(f"note: using pre-fix embedding file {legacy} "
                  "(assumed triadic; rebuild with ppi_embed.py to be sure)",
                  flush=True)
            path = legacy
        elif os.path.exists(legacy):
            raise FileNotFoundError(
                f"{path} not found, and the legacy file {legacy} exists but "
                f"cannot be trusted to be '{completion}' completion -- it "
                "could be from any prior run. Rebuild explicitly:\n"
                f"  python -m ppi_graph --branch {cfg.branch} --completion {completion}\n"
                f"  python -m ppi_embed --branch {cfg.branch} --completion {completion} "
                f"--method {method}"
            )
    d = torch.load(path, weights_only=False)
    ids = np.asarray(ids)
    missing = [p for p in ids if p not in d]
    if missing:
        raise KeyError(f"{len(missing)} IDs missing from {path} "
                       f"(e.g. {missing[:3]})")
    return np.stack([d[p].numpy() for p in ids])


def _data_cache_dir(cfg):
    return os.path.join(cfg.paths.work_root, "cache", cfg.branch)


def _cache_source_signature(cfg):
    """Return file signatures used to invalidate the static data cache."""
    sig = {}
    for split in ("train", "val", "test"):
        base = cfg.paths.split_csv(cfg.branch, split)
        path = base.replace(".csv", ".parquet")
        if not os.path.exists(path):
            path = base
        if not os.path.exists(path):
            raise FileNotFoundError(f"missing split file: {path}")
        st = os.stat(path)
        sig[split] = {"path": path, "size": st.st_size,
                      "mtime_ns": st.st_mtime_ns}
    ic_path = cfg.paths.ic_csv(cfg.branch)
    st = os.stat(ic_path)
    sig["ic"] = {"path": ic_path, "size": st.st_size,
                  "mtime_ns": st.st_mtime_ns}
    return sig


def load_static_data(cfg):
    """Load IDs/labels/GO terms/IC, using a branch-local persistent cache.

    The first run still reads the wide parquet files and builds the uint8 label
    arrays. Subsequent seeds load only compact NumPy arrays and never rebuild
    the enormous pandas DataFrames. The cache is invalidated automatically if
    any source split or the IC file changes.
    """
    cache_dir = _data_cache_dir(cfg)
    os.makedirs(cache_dir, exist_ok=True)
    meta_path = os.path.join(cache_dir, "meta.json")
#    required = {
#        split: os.path.join(cache_dir, f"{split}_labels.npy")
#        for split in ("train", "val", "test")
#    }
    required = {
        "train_ids": os.path.join(cache_dir, "train_ids.npy"),
        "val_ids": os.path.join(cache_dir, "val_ids.npy"),
        "test_ids": os.path.join(cache_dir, "test_ids.npy"),
        "train_labels": os.path.join(cache_dir, "train_labels.npy"),
        "val_labels": os.path.join(cache_dir, "val_labels.npy"),
        "test_labels": os.path.join(cache_dir, "test_labels.npy"),
        "go_terms": os.path.join(cache_dir, "go_terms.npy"),
        "ic": os.path.join(cache_dir, "ic.npy"),
    }
#    required.update({
#        "train_ids": os.path.join(cache_dir, "train_ids.npy"),
#        "val_ids": os.path.join(cache_dir, "val_ids.npy"),
#        "test_ids": os.path.join(cache_dir, "test_ids.npy"),
#        "go_terms": os.path.join(cache_dir, "go_terms.npy"),
#        "ic": os.path.join(cache_dir, "ic.npy"),
#    })

    source_sig = _cache_source_signature(cfg)
    cache_ok = False
    if os.path.exists(meta_path) and all(os.path.exists(p) for p in required.values()):
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
            cache_ok = (meta.get("version") == 1 and
                        meta.get("source_signature") == source_sig)
        except (OSError, ValueError, TypeError):
            cache_ok = False

    if cache_ok:
        log(f"loading static data cache: {cache_dir}")
        ids = {
            split: np.load(required[f"{split}_ids"], allow_pickle=False)
            for split in ("train", "val", "test")
        }
        labels = {
            split: np.load(required[f"{split}_labels"], allow_pickle=False)
            for split in ("train", "val", "test")
        }
        go_terms = np.load(required["go_terms"], allow_pickle=False)
        ic = np.load(required["ic"], allow_pickle=False)
        go_to_ic = dict(zip(go_terms.tolist(), ic.tolist()))
        log("static data cache loaded")
        return ids, labels, go_terms, ic, go_to_ic

    log("static data cache missing/stale; building from source parquet files")
    train_df, val_df, test_df, go_terms = load_all_splits(cfg)
    log(f"splits loaded: {len(train_df)}/{len(val_df)}/{len(test_df)}, "
        f"{len(go_terms)} terms")

    y_tr = get_labels(train_df, go_terms)
    y_va = get_labels(val_df, go_terms)
    y_te = get_labels(test_df, go_terms)
    ic, go_to_ic = load_ic(cfg, go_terms)
    log("labels + IC loaded; writing static cache")

    ids = {
        "train": np.asarray(train_df["ID"].values, dtype=str),
        "val": np.asarray(val_df["ID"].values, dtype=str),
        "test": np.asarray(test_df["ID"].values, dtype=str),
    }
    labels = {"train": y_tr, "val": y_va, "test": y_te}
    go_terms = np.asarray(go_terms, dtype=str)
    ic = np.asarray(ic, dtype=float)

    for split in ("train", "val", "test"):
        np.save(required[f"{split}_ids"], ids[split])
        np.save(required[f"{split}_labels"], labels[split])
    np.save(required["go_terms"], go_terms)
    np.save(required["ic"], ic)
    with open(meta_path, "w") as f:
        json.dump({"version": 1, "source_signature": source_sig}, f, indent=2)
    log(f"static data cache written: {cache_dir}")
    return ids, labels, go_terms, ic, go_to_ic


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None,
              accum=1, desc="train"):
    """Training pass. (Validation now goes through evaluate_pass instead.)"""
    train = optimizer is not None
    model.train(train)
    total, nb = 0.0, 0
    gates = []
    if train:
        optimizer.zero_grad(set_to_none=True)

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for i, (x_seq, mask, x_ppi, y) in enumerate(
                tqdm(loader, desc=desc, leave=False)):
            x_seq = x_seq.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            x_ppi = x_ppi.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.autocast("cuda", enabled=scaler is not None):
                logits, g = model(x_seq, mask, x_ppi)
                loss = criterion(logits, y)

            if train:
                scaled = loss / accum
                if scaler is not None:
                    scaler.scale(scaled).backward()
                else:
                    scaled.backward()
                if (i + 1) % accum == 0 or (i + 1) == len(loader):
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    if scaler is not None:
                        scaler.step(optimizer); scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
            else:
                gates.append(g.float().cpu())

            total += loss.item()
            nb += 1

    g = torch.cat(gates).numpy() if gates else None
    return total / max(nb, 1), g


@torch.no_grad()
def evaluate_pass(model, loader, device, criterion=None, desc="predict"):
    """
    One fp32 forward pass over `loader`, returning probabilities, labels,
    gates and (optionally) the mean loss.

    This replaces the old run_epoch(val) + predict(val) pair. The
    probabilities are computed exactly as predict() computed them -- full
    fp32, no autocast -- so checkpoint selection and the final reported
    metrics are unchanged. Only the logged loss differs (fp32 rather than
    autocast fp16), and the loss is never used for selection or early
    stopping.
    """
    model.eval()
    probs, trues, gates = [], [], []
    total, nb = 0.0, 0

    for x_seq, mask, x_ppi, y in tqdm(loader, desc=desc, leave=False):
        x_seq = x_seq.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        x_ppi = x_ppi.to(device, non_blocking=True)

        logits, g = model(x_seq, mask, x_ppi)

        if criterion is not None:
            total += criterion(logits, y.to(device, non_blocking=True)).item()
            nb += 1

        probs.append(torch.sigmoid(logits.float()).cpu().numpy())
        trues.append(y.numpy())
        gates.append(g.float().cpu().numpy())

    loss = total / max(nb, 1) if criterion is not None else float("nan")
    return np.vstack(probs), np.vstack(trues), np.vstack(gates), loss


@torch.no_grad()
def predict(model, loader, device):
    """Kept for external callers; delegates to evaluate_pass."""
    probs, trues, gates, _ = evaluate_pass(model, loader, device)
    return probs, trues, gates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--branch", default="mf")
    ap.add_argument("--fusion", default="gate",
                    choices=["gate", "concat", "crossattn", "seq_only",
                             "ppi_only"])
    ap.add_argument("--ppi-method", default="deepwalk")
    ap.add_argument("--completion", default="triadic")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--tag-suffix", default=None,
                    help="append to the auto-generated tag, e.g. 'v2', so "
                         "improved runs never overwrite a baseline with the "
                         "same fusion/seed")
    ap.add_argument("--no-hier-loss", action="store_true")
    ap.add_argument("--dropout", type=float, default=None,
                    help="override classifier dropout (cfg default 0.3). "
                         "Does NOT affect CrossModalAttention's dropout, "
                         "which is hardcoded at 0.1 independently of this.")
    ap.add_argument("--weight-decay", type=float, default=None,
                    help="override Adam weight_decay (cfg default 0.0, i.e. "
                         "off -- no ablation run so far has used this)")
    ap.add_argument("--num-blocks", type=int, default=None)
    ap.add_argument("--dilation-schedule", default=None,
                    choices=["moderate", "aggressive"])
    ap.add_argument("--focal-unbalanced", action="store_true",
                    help="reproduce the ORIGINAL (buggy) scalar-alpha focal "
                         "loss, for a like-for-like baseline")
    ap.add_argument("--hidden-dim", type=int, default=None)
    ap.add_argument(
        "--go-embeddings",
        default=None,
        help="path to pretrained GO embedding .pt file; "
             "if omitted, use the original classifier"
    )
    ap.add_argument("--focal-alpha", type=float, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--num-workers", type=int, default=None,
                    help="override the DataLoader worker count (default 2, "
                         "safe for several concurrent jobs). Raise only if "
                         "you are certain you are the sole job on the box. "
                         "Use 0 to rule the DataLoader out when diagnosing "
                         "a hang.")
    ap.add_argument("--eval-threads", type=int, default=None,
                    help="override evaluate_cafa's thread count (default 4, "
                         "same concurrency caveat as --num-workers)")
                         
    ap.add_argument(
                      "--clf-hidden",
                      type=int,
                      default=512,
                      help="Hidden dimension of the classifier MLP.",
                  )
    args = ap.parse_args()

    cfg = get_config(branch=args.branch)
    cfg.model.fusion = args.fusion
    cfg.train.seed = args.seed
    if args.epochs is not None:
        cfg.train.num_epochs = args.epochs
    if args.num_blocks is not None:
        cfg.model.num_blocks = args.num_blocks
    if args.dilation_schedule is not None:
        cfg.model.dilation_schedule = args.dilation_schedule
    if args.focal_unbalanced:
        cfg.train.focal_balanced = False
    if args.hidden_dim is not None:
        cfg.model.hidden_dim = args.hidden_dim
    if args.focal_alpha is not None:
        cfg.train.focal_alpha = args.focal_alpha
    if args.lr is not None:
        cfg.train.lr = args.lr
    if args.num_workers is not None:
        cfg.train.num_workers = args.num_workers
    if args.eval_threads is not None:
        cfg.train.eval_threads = args.eval_threads
    if args.dropout is not None:
        cfg.model.dropout = args.dropout
    if args.weight_decay is not None:
        cfg.train.weight_decay = args.weight_decay

    tag = args.tag or (f"{args.branch}_{args.fusion}_{args.ppi_method}"
                       f"_{args.completion}_s{args.seed}"
                       if args.completion != "triadic"
                       else f"{args.branch}_{args.fusion}_{args.ppi_method}_s{args.seed}")
    if args.tag_suffix and not args.tag:
        tag += f"_{args.tag_suffix}"
    run_dir = cfg.paths.work("runs", tag, "_")[:-1]
    os.makedirs(run_dir, exist_ok=True)

    log(f"start pid={os.getpid()} tag={tag}")
    log(f"threads: OMP={os.environ.get('OMP_NUM_THREADS', 'unset')} "
        f"MKL={os.environ.get('MKL_NUM_THREADS', 'unset')} "
        f"torch={torch.get_num_threads()}")

    set_seed(cfg.train.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("*** WARNING: running on CPU. Training will be 10-50x slower "
              "than on GPU. If you expected CUDA, check `nvidia-smi` and "
              "`python -c \"import torch; print(torch.cuda.is_available())\"` "
              "before assuming the pipeline itself got slower. ***", flush=True)
    else:
        log(f"device: {device} ({torch.cuda.get_device_name(device)})")

    # ---- data -------------------------------------------------------------
    ids, labels, go_terms, ic, go_to_ic = load_static_data(cfg)
    log(f"static data ready: {len(ids['train'])}/{len(ids['val'])}/{len(ids['test'])}, "
        f"{len(go_terms)} terms")

    ppi = {s: load_ppi_features(cfg, ids[s], s, args.ppi_method, args.completion)
           for s in ("train", "val", "test")}
    log(f"PPI features loaded (dim={ppi['train'].shape[1]})")

    esm_dir = os.path.join(cfg.paths.work_root, "esm", cfg.model.esm_model)

    datasets = [
        ESMPPIDataset(ids[s], esm_dir, s, ppi[s], labels[s],
                      cfg.model.max_len, cfg.model.esm_dim)
        for s in ("train", "val", "test")
    ]
    train_loader, val_loader, test_loader = make_loaders(cfg, datasets)
    log(f"loaders built (num_workers={cfg.train.num_workers})")

    # ---- ontology ---------------------------------------------------------
    ontology = load_ontology(cfg.paths.ontology, cfg.namespace)
    valid = [i for i, g in enumerate(go_terms)
             if g in ontology and g in go_to_ic and np.isfinite(ic[i])]
    log(f"labels: {len(go_terms)} total, {len(valid)} usable for IC metrics")
    go_valid = go_terms[valid]
    ic_valid = ic[valid]

    # ---- model ------------------------------------------------------------
#    model = LightProteinFusion(
#        num_classes=len(go_terms), esm_dim=cfg.model.esm_dim,
#        ppi_dim=ppi["train"].shape[1], hidden_dim=cfg.model.hidden_dim,
#        dropout=cfg.model.dropout, fusion=cfg.model.fusion,
#        num_blocks=cfg.model.num_blocks,
#        dilation_schedule=cfg.model.dilation_schedule,
#    ).to(device)
    
    
#    if args.go_embeddings is None:
#        model = LightProteinFusion(
#            num_classes=len(go_terms), esm_dim=cfg.model.esm_dim,
#            ppi_dim=ppi["train"].shape[1], hidden_dim=cfg.model.hidden_dim,
#            dropout=cfg.model.dropout, fusion=cfg.model.fusion,
#            num_blocks=cfg.model.num_blocks,
#            dilation_schedule=cfg.model.dilation_schedule,
#        ).to(device)
#        log("classifier: original MLP")
#    else:
#        go_data = torch.load(
#            args.go_embeddings,
#            map_location="cpu",
#            weights_only=False
#        )
#    
#        pretrained_go_terms = np.asarray(
#            go_data["go_terms"],
#            dtype=str
#        )
#    
#        if not np.array_equal(pretrained_go_terms, go_terms):
#            raise ValueError(
#                "GO embedding vocabulary/order does not exactly match "
#                "the current protein-function vocabulary."
#            )
#    
#        go_embeddings = go_data["embeddings"]
#    
#        if go_embeddings.shape[0] != len(go_terms):
#            raise ValueError(
#                f"GO embedding rows={go_embeddings.shape[0]} but "
#                f"current vocabulary={len(go_terms)}"
#            )
#    
#        if args.fusion == "concat":
#            classifier_input_dim = 2 * cfg.model.hidden_dim
#        else:
#            classifier_input_dim = cfg.model.hidden_dim
#    
#        model = LightProteinFusion(
#            num_classes=len(go_terms),
#            esm_dim=cfg.model.esm_dim,
#            ppi_dim=ppi["train"].shape[1],
#            hidden_dim=cfg.model.hidden_dim,
#            dropout=cfg.model.dropout,
#            fusion=cfg.model.fusion,
#            num_blocks=cfg.model.num_blocks,
#            dilation_schedule=cfg.model.dilation_schedule,
#        ).to(device)
#    
#        model.classifier = GOEmbeddingClassifier(
#            classifier_input_dim,
#            go_embeddings,
#            dropout=cfg.model.dropout,
#        ).to(device)
#
#        log(
#            f"classifier: pretrained GO embeddings "
#            f"{tuple(go_embeddings.shape)}"
#        )
#    
        
        
        
        
    go_embeddings = None

    if args.go_embeddings is not None:
        go_data = torch.load(
            args.go_embeddings,
            map_location="cpu",
            weights_only=False,
        )
    
        pretrained_go_terms = np.asarray(
            go_data["go_terms"],
            dtype=str,
        )
    
        if not np.array_equal(
            pretrained_go_terms,
            go_terms,
        ):
            raise ValueError(
                "GO embedding vocabulary does not match model vocabulary"
            )
    
        go_embeddings = go_data["embeddings"]
    
        if go_embeddings.shape[0] != len(go_terms):
            raise ValueError(
                f"GO embedding rows {go_embeddings.shape[0]} "
                f"!= number of GO terms {len(go_terms)}"
            )
    
        log(
            f"GO label attention enabled: "
            f"{tuple(go_embeddings.shape)}"
        )
    else:
        log("GO label attention disabled")
    
    
#    model = LightProteinFusion(
#        num_classes=len(go_terms),
#        esm_dim=cfg.model.esm_dim,
#        ppi_dim=ppi["train"].shape[1],
#        hidden_dim=cfg.model.hidden_dim,
#        dropout=cfg.model.dropout,
#        fusion=cfg.model.fusion,
#        num_blocks=cfg.model.num_blocks,
#        dilation_schedule=cfg.model.dilation_schedule,
#        go_embeddings=go_embeddings,
#    ).to(device)    
    
    
    model = LightProteinFusion(
        num_classes=len(go_terms),
        esm_dim=cfg.model.esm_dim,
        ppi_dim=ppi["train"].shape[1],
        hidden_dim=cfg.model.hidden_dim,
        dropout=cfg.model.dropout,
        fusion=cfg.model.fusion,
        num_blocks=cfg.model.num_blocks,
        dilation_schedule=cfg.model.dilation_schedule,
        go_embeddings=go_embeddings,
        clf_hidden=args.clf_hidden,
    ).to(device)
        
        
        
    log(f"trainable parameters: {count_parameters(model):,}")
    if cfg.model.fusion != "ppi_only":
        print(f"encoder theoretical receptive field: "
              f"{model.encoder.receptive_field} residues "
              f"({cfg.model.num_blocks} blocks, {cfg.model.dilation_schedule})",
              flush=True)

    if cfg.train.hierarchical_loss and not args.no_hier_loss:
        anc = build_ancestor_matrix(go_terms, ontology)
        added = check_labels_propagated(labels["train"], anc)
        use_hier = added > 0
    else:
        anc, use_hier = None, False
        log("hierarchical loss disabled; skipping ancestor matrix")
    
    
    
    
    criterion = (HierarchicalFocalLoss(anc.to(device), cfg.train.focal_alpha,
                                       cfg.train.focal_gamma,
                                       cfg.train.focal_balanced).to(device)
                 if use_hier
                 else FocalLoss(cfg.train.focal_alpha, cfg.train.focal_gamma,
                                balanced=cfg.train.focal_balanced))
    print(f"loss: {'hierarchical ' if use_hier else ''}focal "
          f"(alpha={cfg.train.focal_alpha}, "
          f"balanced={cfg.train.focal_balanced})", flush=True)
    print(f"batch={cfg.train.batch_size} x accum={cfg.train.accumulation_steps} "
          f"= effective {cfg.train.batch_size * cfg.train.accumulation_steps}",
          flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr,
                                 weight_decay=cfg.train.weight_decay,
                                 )
                                 #amsgrad=True) as Adam is changed to AdamW
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.train.num_epochs)
    scaler = torch.amp.GradScaler("cuda") if (cfg.train.amp and
                                              device.type == "cuda") else None

    log("setup complete, entering training loop")

    # ---- train ------------------------------------------------------------
    # CAFA is expensive on BP (19k labels). Use a small threshold grid only
    # for checkpoint selection, and run it every 5 epochs. Final reported
    # metrics below still use the standard 101-threshold CAFA evaluation.
    eval_every = 5
    train_eval_thresholds = 11
    patience_evals = max(1, int(np.ceil(cfg.train.patience / eval_every)))

    best_fmax, best_state, best_epoch, bad_evals = -1.0, None, None, 0
    history = []
    for epoch in range(cfg.train.num_epochs):
        t_ep = time.time()

        tr_loss, _ = run_epoch(model, train_loader, criterion, device,
                               optimizer, scaler, cfg.train.accumulation_steps,
                               f"epoch {epoch + 1} train")
        t_tr = time.time() - t_ep

        # One validation pass: loss + probabilities together.
        vp, vt, _, va_loss = evaluate_pass(model, val_loader, device,
                                           criterion,
                                           desc=f"epoch {epoch + 1} val")
        t_va = time.time() - t_ep - t_tr

        scheduler.step()

        do_eval = ((epoch + 1) % eval_every == 0 or
                   (epoch + 1) == cfg.train.num_epochs)

        if do_eval:
            t_eval = time.time()
            va = evaluate_cafa(
                vt[:, valid],
                vp[:, valid],
                go_valid,
                ontology,
                ic_valid,
                verbose=False,
                n_jobs=cfg.train.eval_threads,
                n_thresholds=train_eval_thresholds
            )
            t_ev = time.time() - t_eval

            print(f"[{epoch + 1}/{cfg.train.num_epochs}] "
                  f"train {tr_loss:.5f} | val {va_loss:.5f} | "
                  f"val Fmax {va['Fmax']:.4f} | "
                  f"{t_tr:.0f}s train / {t_va:.0f}s val / "
                  f"{t_ev:.0f}s eval ({train_eval_thresholds} thr)",
                  flush=True)

            history.append({"epoch": epoch + 1, "train_loss": tr_loss,
                            "val_loss": va_loss, "t_train": t_tr,
                            "t_val": t_va, "t_eval": t_ev,
                            "eval_thresholds": train_eval_thresholds,
                            **va})

            # Select on Fmax, not validation loss. Count patience in CAFA
            # evaluation events so patience=10 epochs remains approximately
            # equivalent to the previous behavior.
            if va["Fmax"] > best_fmax:
                best_fmax = va["Fmax"]
                best_epoch = epoch + 1
                bad_evals = 0
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
                torch.save({"epoch": best_epoch, "fmax": best_fmax,
                            "state": best_state},
                           os.path.join(run_dir, "best.pt"))
            else:
                bad_evals += 1
                if bad_evals >= patience_evals:
                    print("early stopping", flush=True)
                    break
        else:
            print(f"[{epoch + 1}/{cfg.train.num_epochs}] "
                  f"train {tr_loss:.5f} | val {va_loss:.5f} | "
                  f"CAFA skipped | "
                  f"{t_tr:.0f}s train / {t_va:.0f}s val", flush=True)
            history.append({"epoch": epoch + 1, "train_loss": tr_loss,
                            "val_loss": va_loss, "t_train": t_tr,
                            "t_val": t_va, "t_eval": None})

    if best_state:
        model.load_state_dict(best_state)
        torch.save(best_state, os.path.join(run_dir, "model.pt"))

    # ---- evaluate ---------------------------------------------------------
    results = {"tag": tag, "fusion": args.fusion, "seed": args.seed,
               "params": count_parameters(model), "history": history}

    for name, loader in [("val", val_loader), ("test", test_loader)]:
        log(f"final evaluation: {name}")
        probs, true, gates, _ = evaluate_pass(model, loader, device,
                                              desc=f"final {name}")
        yt, ys = true[:, valid], probs[:, valid]
        print(f"\nFINAL {name.upper()} evaluation started (101 thresholds)...", flush=True)
        t_eval = time.time()              
                            
        r = evaluate_cafa(yt, ys, go_valid, ontology, ic_valid,
                          verbose=False, n_jobs=cfg.train.eval_threads)
                          
        print(f"FINAL {name.upper()} evaluation finished in "
          f"{time.time() - t_eval:.1f}s", flush=True)
          
        log(f"{name}: CAFA metrics done")
        r["macro_AUPRC"] = macro_auprc(yt, ys)
        log(f"{name}: macro AUPRC done")
        results[name] = r
        print(f"\n== {name} ==", flush=True)
        for k, v in r.items():
            print(f"  {k}: {v:.4f}", flush=True)
        np.savez(os.path.join(run_dir, f"pred_{name}.npz"),
                 y_score=ys, y_true=yt.astype(np.uint8), go_terms=go_valid,
                 ic=ic_valid, gates=gates, ids=ids[name])
        log(f"{name}: predictions saved")
        

    with open(os.path.join(run_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)
    log(f"saved to {run_dir}")


if __name__ == "__main__":
    main()