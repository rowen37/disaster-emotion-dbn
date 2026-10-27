"""Leakage-safe training for behavior presence and multi-span extraction.

Examples:
  python train.py --check-only   # paths/schema/folds only; loads no model
  python train.py --mode cv
  python train.py --mode final   # normally run only after CV
"""
import argparse
import copy
import json
import math
import random
import statistics
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

import config as cfg
from data_utils import BehaviorSpanDataset, load_split, pick_device
from metrics_utils import evaluate_split
from model import BehaviorSpanExtractor


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def stratified_group_inner_split(rows, frac, seed):
    """Make an inner validation split without separating a leakage group."""
    grouped = []
    for group_id, frame in pd.DataFrame(rows).groupby("group_id", sort=False):
        records = frame.to_dict("records")
        counts = Counter(int(r["has_behavior"]) for r in records)
        grouped.append((group_id, records, counts))
    rng = random.Random(seed)
    rng.shuffle(grouped)
    grouped.sort(key=lambda item: -len(item[1]))
    n_bins = max(2, round(1 / frac))
    bins, sizes = [[] for _ in range(n_bins)], [0] * n_bins
    counts = [Counter() for _ in range(n_bins)]
    totals = Counter(int(r["has_behavior"]) for r in rows)
    targets = {key: value / n_bins for key, value in totals.items()}
    for _, records, group_counts in grouped:
        def load(bin_id):
            class_load = sum((counts[bin_id][key] + value) / max(targets[key], 1)
                             for key, value in group_counts.items())
            return class_load + 0.25 * (sizes[bin_id] + len(records)) / max(len(rows) / n_bins, 1)
        chosen = min(range(n_bins), key=lambda b: (load(b), sizes[b], b))
        bins[chosen].extend(records)
        sizes[chosen] += len(records)
        counts[chosen].update(group_counts)
    val_bin = seed % n_bins
    val = bins[val_bin]
    train = [row for i, part in enumerate(bins) if i != val_bin for row in part]
    return train, val


def binary_weights(rows, field):
    counts = Counter(int(row[field]) for row in rows)
    total = len(rows)
    return torch.tensor([total / max(2 * counts[i], 1) for i in (0, 1)], dtype=torch.float)


def effective_number_weights(counts, beta, max_weight):
    """Cui et al. class-balanced weights, normalized to mean one."""
    values = torch.tensor([float(counts[i]) for i in range(cfg.N_TAGS)])
    if bool((values <= 0).any()):
        raise ValueError(f"every BIO tag must occur in training data: {counts}")
    weights = (1.0 - beta) / (1.0 - torch.pow(torch.tensor(beta), values))
    weights = weights / weights.mean()
    return weights.clamp(max=max_weight)


def tag_weights(dataset, field):
    counts = Counter(tag for example in dataset.examples for tag in example[field] if tag != -100)
    return effective_number_weights(counts, cfg.CLASS_BALANCE_BETA, cfg.TAG_WEIGHT_MAX)


def compute_loss(model, output, batch, losses, device):
    bt = batch["behavior_tags"].to(device)
    hb = batch["has_behavior"].to(device)
    behavior_mask = bt[:, 1:] != -100
    behavior_tags = bt[:, 1:].clamp_min(0)
    # Keep CRF/log-sum-exp and both CE losses in FP32 even when the encoder
    # forward pass runs under mixed precision.
    with torch.cuda.amp.autocast(enabled=False):
        behavior_emissions = output["behavior_tags"].float()
        behavior_crf = model.crf_loss(
            behavior_emissions[:, 1:], behavior_tags, behavior_mask)
        behavior_aux = losses["behavior_tag"](
            behavior_emissions.reshape(-1, cfg.N_TAGS), bt.reshape(-1))
        behavior = behavior_crf + cfg.TAG_AUX_CE_WEIGHT * behavior_aux
        presence = losses["behavior_presence"](output["has_behavior"].float(), hb)
    return behavior + cfg.PRESENCE_LOSS_WEIGHT * presence


def make_losses(rows, dataset, device):
    return {
        "behavior_tag": nn.CrossEntropyLoss(weight=tag_weights(dataset, "behavior_tags").to(device), ignore_index=-100),
        "behavior_presence": nn.CrossEntropyLoss(weight=binary_weights(rows, "has_behavior").to(device)),
    }


def new_model(device):
    model = BehaviorSpanExtractor(cfg.MODEL_NAME, cfg.DROPOUT, cfg.N_TAGS)
    if cfg.GRADIENT_CHECKPOINTING:
        model.encoder.gradient_checkpointing_enable()
    return model.to(device)


def train_epoch(model, loader, optimizer, scheduler, scaler, losses, device):
    """Train one epoch with gradient accumulation (AMP only on CUDA)."""
    model.train()
    optimizer.zero_grad(set_to_none=True)
    running = 0.0
    use_amp = cfg.USE_AMP and device.type == "cuda"
    for step, batch in enumerate(loader, 1):
        with torch.cuda.amp.autocast(enabled=use_amp):
            output = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
            raw_loss = compute_loss(model, output, batch, losses, device)
            loss = raw_loss / cfg.GRAD_ACCUM_STEPS
        scaler.scale(loss).backward()
        should_step = step % cfg.GRAD_ACCUM_STEPS == 0 or step == len(loader)
        if should_step:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
        running += raw_loss.item()
    return running / max(len(loader), 1)


def checkpoint_payload(model, **metadata):
    return {
        "format_version": cfg.CHECKPOINT_FORMAT_VERSION,
        "architecture": "encoder-harmful-signal-presence-and-linear-chain-crf",
        "model_name": cfg.MODEL_NAME, "max_len": cfg.MAX_LEN,
        "tag_scheme": ["O", "B", "I"],
        "class_balance_beta": cfg.CLASS_BALANCE_BETA,
        "tag_aux_ce_weight": cfg.TAG_AUX_CE_WEIGHT,
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        **metadata,
    }


def run_one(fold, seed, folds_data, tokenizer, device, log_rows, save_ckpt):
    set_seed(seed)
    test_rows = folds_data[fold]
    pool = [row for f, rows in folds_data.items() if f != fold for row in rows]
    train_rows, val_rows = stratified_group_inner_split(pool, cfg.INNER_VAL_FRAC, seed * 1000 + fold)
    train_ds = BehaviorSpanDataset(train_rows, tokenizer, cfg.MAX_LEN, True, cfg.MASK_AUG_PROB)
    val_ds = BehaviorSpanDataset(val_rows, tokenizer, cfg.MAX_LEN)
    test_ds = BehaviorSpanDataset(test_rows, tokenizer, cfg.MAX_LEN)
    train_loader = DataLoader(train_ds, cfg.BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, cfg.EVAL_BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, cfg.EVAL_BATCH_SIZE, shuffle=False, num_workers=0)

    model = new_model(device)
    optimizer = AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    steps_per_epoch = math.ceil(len(train_loader) / cfg.GRAD_ACCUM_STEPS)
    total_steps = max(steps_per_epoch * cfg.EPOCHS, 1)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * cfg.WARMUP_RATIO), total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.USE_AMP and device.type == "cuda")
    losses = make_losses(train_rows, train_ds, device)
    best_score, best_epoch, best_state, patience = -1.0, 0, None, 0
    for epoch in range(1, cfg.EPOCHS + 1):
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, scaler, losses, device)
        metrics, _, _ = evaluate_split(model, val_loader, device)
        log_rows.append({"architecture_version": cfg.RUN_ID,
                         "run": f"f{fold}s{seed}", "fold": fold, "seed": seed,
                         "epoch": epoch, "train_loss": train_loss, **metrics})
        score = metrics["selection_span_F1"]
        print(
            f"[fold {fold} seed {seed}] epoch {epoch}: "
            f"behavior_presence_F1={metrics['behavior_presence_F1']:.4f} "
            f"behavior_span_char_F1={metrics['behavior_span_char_F1']:.4f} "
            f"selection_span_F1={score:.4f}"
        )
        if score > best_score:
            best_score, best_epoch, patience = score, epoch, 0
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        else:
            patience += 1
            if patience >= cfg.EARLY_STOP_PATIENCE:
                break

    model.load_state_dict(best_state)
    metrics, _, _ = evaluate_split(model, test_loader, device)
    metrics.update(architecture_version=cfg.RUN_ID,
                   fold=fold, seed=seed, best_epoch=best_epoch,
                   inner_best_selection_span_F1=best_score)
    if save_ckpt:
        cfg.CKPT_DIR.mkdir(parents=True, exist_ok=True)
        torch.save({"format_version": cfg.CHECKPOINT_FORMAT_VERSION,
                    "architecture": "encoder-harmful-signal-presence-and-linear-chain-crf",
                    "class_balance_beta": cfg.CLASS_BALANCE_BETA,
                    "tag_aux_ce_weight": cfg.TAG_AUX_CE_WEIGHT,
                    "model_name": cfg.MODEL_NAME,
                    "max_len": cfg.MAX_LEN, "tag_scheme": ["O", "B", "I"],
                    "state_dict": best_state, "fold": fold, "seed": seed},
                   cfg.CKPT_DIR / f"fold{fold}.pth")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics


def train_final(tokenizer, device, epochs, log_rows):
    set_seed(cfg.DEPLOY_SEED)
    rows = [row for part in load_split(cfg.DATA_TRAIN, cfg.FOLDS_FILE).values() for row in part]
    dataset = BehaviorSpanDataset(rows, tokenizer, cfg.MAX_LEN, True, cfg.MASK_AUG_PROB)
    loader = DataLoader(dataset, cfg.BATCH_SIZE, shuffle=True, num_workers=0)
    model = new_model(device)
    optimizer = AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    steps_per_epoch = math.ceil(len(loader) / cfg.GRAD_ACCUM_STEPS)
    total_steps = max(steps_per_epoch * epochs, 1)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * cfg.WARMUP_RATIO), total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.USE_AMP and device.type == "cuda")
    losses = make_losses(rows, dataset, device)
    for epoch in range(1, epochs + 1):
        value = train_epoch(model, loader, optimizer, scheduler, scaler, losses, device)
        log_rows.append({"architecture_version": cfg.RUN_ID,
                         "run": "final", "fold": -1, "seed": cfg.DEPLOY_SEED,
                         "epoch": epoch, "train_loss": value})
        print(f"[final] epoch {epoch}: train_loss={value:.4f}")
    cfg.CKPT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_payload(
        model, epochs=epochs, seed=cfg.DEPLOY_SEED,
    ),
               cfg.CKPT_DIR / "final_model.pth")
    cfg.RUNS_DIR.mkdir(parents=True, exist_ok=True)
    with open(cfg.FINAL_INFO, "w", encoding="utf-8") as handle:
        json.dump({"epochs": epochs, "seed": cfg.DEPLOY_SEED, "model": cfg.MODEL_NAME,
                   "architecture_version": cfg.RUN_ID,
                   "architecture": "encoder-harmful-signal-presence-and-linear-chain-crf",
                   "checkpoint_format_version": cfg.CHECKPOINT_FORMAT_VERSION,
                   "class_balance_beta": cfg.CLASS_BALANCE_BETA,
                   "tag_aux_ce_weight": cfg.TAG_AUX_CE_WEIGHT,
                   "checkpoint": str(cfg.CKPT_DIR / "final_model.pth")}, handle, indent=2)


def check_inputs():
    if not cfg.DATA_TRAIN.is_file():
        raise FileNotFoundError(cfg.DATA_TRAIN)
    if not cfg.FOLDS_FILE.is_file():
        raise FileNotFoundError(f"{cfg.FOLDS_FILE}; run prepare_data.py first")
    folds = load_split(cfg.DATA_TRAIN, cfg.FOLDS_FILE)
    rows = [row for part in folds.values() for row in part]
    expected_folds = set(range(cfg.N_FOLDS))
    if set(folds) != expected_folds:
        raise ValueError(f"expected folds {sorted(expected_folds)}, got {sorted(folds)}")
    if len(rows) % cfg.N_FOLDS:
        raise ValueError("training row count must be divisible by the number of folds")
    target_size = len(rows) // cfg.N_FOLDS
    unequal = {fold: len(part) for fold, part in folds.items() if len(part) != target_size}
    if unequal:
        raise ValueError(f"every fold must contain {target_size} rows: {unequal}")
    groups = {}
    for fold, part in folds.items():
        for row in part:
            previous = groups.setdefault(row["group_id"], fold)
            if previous != fold:
                raise ValueError(f"group leakage: {row['group_id']}")
    print(f"OK: {len(rows)} rows, {len(groups)} groups, folds="
          + ", ".join(f"{fold}:{len(part)}" for fold, part in sorted(folds.items())))
    print(f"input: {cfg.DATA_TRAIN}\ncheckpoints: {cfg.CKPT_DIR}\nlogs: {cfg.RUNS_DIR}")


def upsert_csv(path, rows, key_columns, sort_columns):
    """Merge completed work into a CSV atomically for safe resume."""
    if not rows:
        return
    incoming = pd.DataFrame(rows)
    if path.is_file():
        existing = pd.read_csv(path)
        if "architecture_version" in incoming:
            if "architecture_version" in existing:
                existing = existing[existing["architecture_version"] == cfg.RUN_ID]
            else:
                existing = existing.iloc[0:0]
        combined = (incoming if existing.empty else
                    pd.concat([existing, incoming], ignore_index=True, sort=False))
    else:
        combined = incoming
    combined = (combined.drop_duplicates(subset=key_columns, keep="last")
                        .sort_values(sort_columns).reset_index(drop=True))
    temporary = path.with_suffix(path.suffix + ".tmp")
    combined.to_csv(temporary, index=False)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["cv", "final"])
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--folds", default="all")
    parser.add_argument("--seeds", default=",".join(map(str, cfg.SEEDS)))
    args = parser.parse_args()
    check_inputs()
    if args.check_only:
        return
    if not args.mode:
        parser.error("--mode is required unless --check-only is used")

    cfg.RUNS_DIR.mkdir(parents=True, exist_ok=True)
    cfg.CKPT_DIR.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    print(f"device: {device}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_NAME, use_fast=True)
    folds_data = load_split(cfg.DATA_TRAIN, cfg.FOLDS_FILE)
    log_rows = []
    if args.mode == "cv":
        fold_ids = sorted(folds_data) if args.folds == "all" else [int(x) for x in args.folds.split(",")]
        seeds = [int(value) for value in args.seeds.split(",")]
        if cfg.METRICS_CSV.is_file():
            old_metrics = pd.read_csv(cfg.METRICS_CSV)
            if "architecture_version" in old_metrics:
                old_metrics = old_metrics[old_metrics["architecture_version"] == cfg.RUN_ID]
            else:
                old_metrics = old_metrics.iloc[0:0]
            completed = {(int(row.fold), int(row.seed)) for row in old_metrics.itertuples()}
        else:
            completed = set()
        plan = [(fold, seed) for fold in fold_ids for seed in seeds
                if (fold, seed) not in completed]
        print(f"resume: {len(completed)} completed, {len(plan)} remaining")
        for index, (fold, seed) in enumerate(plan, 1):
            print(f"=== remaining run {index}/{len(plan)}: fold {fold} seed {seed} ===")
            start = len(log_rows)
            metric = run_one(fold, seed, folds_data, tokenizer, device, log_rows,
                             save_ckpt=(seed == cfg.DEPLOY_SEED))
            upsert_csv(cfg.METRICS_CSV, [metric], ["architecture_version", "fold", "seed"],
                       ["fold", "seed"])
            upsert_csv(cfg.TRAIN_LOG_CSV, log_rows[start:],
                       ["architecture_version", "run", "epoch"],
                       ["fold", "seed", "epoch"])
    else:
        if cfg.METRICS_CSV.is_file():
            completed = pd.read_csv(cfg.METRICS_CSV)
            if "architecture_version" in completed:
                completed = completed[completed["architecture_version"] == cfg.RUN_ID]
            if completed.empty:
                raise ValueError("no completed CV metrics for the current architecture")
            epochs = max(1, int(statistics.median(completed["best_epoch"])))
        else:
            raise FileNotFoundError(f"{cfg.METRICS_CSV}; run --mode cv before --mode final")
        train_final(tokenizer, device, epochs, log_rows)
        upsert_csv(cfg.TRAIN_LOG_CSV, log_rows,
                   ["architecture_version", "run", "epoch"],
                   ["fold", "seed", "epoch"])


if __name__ == "__main__":
    main()
