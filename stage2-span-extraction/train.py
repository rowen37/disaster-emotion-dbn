"""
Stage 2 - emotion-gated harm-span extraction: training loop.

Loss = span-start CE + span-end CE + weighted binary harm CE. The harm head
is up-weighted for no-harm samples and further for no-harm samples whose
dominant emotion is happy (the main confusion direction in this task).
Early stopping on validation span F1; best checkpoint to checkpoints/.
"""
import os
import torch
import torch.nn as nn
import pandas as pd
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import wandb
import numpy as np

from config import Config
from model import RoBERTaGateSpanExtractor
from data_utils import HarmSpanDataset

cfg = Config()

def compute_span_f1(start_logits, end_logits, start_labels, end_labels, attention_mask, ignore_index=-100):
    """Exact-boundary span accuracy computed as P=R=F1 (one prediction per gold span)."""
    pred_starts = torch.argmax(start_logits, dim=-1)
    pred_ends = torch.argmax(end_logits, dim=-1)
    valid_mask = (start_labels != ignore_index)
    if valid_mask.sum() == 0:
        return 0.0, 0.0, 0.0
    pred_starts = pred_starts[valid_mask]
    pred_ends = pred_ends[valid_mask]
    true_starts = start_labels[valid_mask]
    true_ends = end_labels[valid_mask]
    correct = ((pred_starts == true_starts) & (pred_ends == true_ends)).sum().item()
    total = len(true_starts)
    p = correct / total if total > 0 else 0.0
    r = correct / total
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1

def train_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        emo_probs = batch['emo_probs'].to(device)
        start_labels = batch['start_label'].to(device)
        end_labels = batch['end_label'].to(device)
        has_harm_labels = batch['has_harm_label'].to(device)

        start_logits, end_logits, has_harm_logits = model(input_ids, attention_mask, emo_probs)

        loss_start = nn.CrossEntropyLoss(ignore_index=-100)(start_logits, start_labels)
        loss_end = nn.CrossEntropyLoss(ignore_index=-100)(end_logits, end_labels)

        # Emotion-aware loss weighting
        happy_prob = emo_probs[:, 5]  # happy
        # base weight: up-weight negative (no-harm) samples
        base_weights = torch.where(has_harm_labels == 0, torch.tensor(cfg.negative_weight).to(device), torch.tensor(1.0).to(device))
        # extra weight on no-harm samples with happy probability > 0.5
        extra_weight = torch.where((happy_prob > 0.5) & (has_harm_labels == 0),
                                   torch.tensor(cfg.happy_penalty_weight).to(device),
                                   torch.tensor(1.0).to(device))
        final_weights = base_weights * extra_weight

        loss_has_harm = nn.CrossEntropyLoss(reduction='none')(has_harm_logits, has_harm_labels)
        loss_has_harm = (loss_has_harm * final_weights).mean()

        loss = loss_start + loss_end + loss_has_harm
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

    return total_loss / len(dataloader)

def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    all_has_harm_pred = []
    all_has_harm_true = []
    all_start_logits = []
    all_end_logits = []
    all_start_labels = []
    all_end_labels = []
    all_attention_masks = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            emo_probs = batch['emo_probs'].to(device)
            start_labels = batch['start_label'].to(device)
            end_labels = batch['end_label'].to(device)
            has_harm_labels = batch['has_harm_label'].to(device)

            start_logits, end_logits, has_harm_logits = model(input_ids, attention_mask, emo_probs)

            loss_start = nn.CrossEntropyLoss(ignore_index=-100)(start_logits, start_labels)
            loss_end = nn.CrossEntropyLoss(ignore_index=-100)(end_logits, end_labels)
            loss_has_harm = nn.CrossEntropyLoss()(has_harm_logits, has_harm_labels)
            total_loss += (loss_start + loss_end + loss_has_harm).item()

            has_harm_pred = torch.argmax(has_harm_logits, dim=-1)
            all_has_harm_pred.extend(has_harm_pred.cpu().numpy())
            all_has_harm_true.extend(has_harm_labels.cpu().numpy())

            all_start_logits.append(start_logits.cpu())
            all_end_logits.append(end_logits.cpu())
            all_start_labels.append(start_labels.cpu())
            all_end_labels.append(end_labels.cpu())
            all_attention_masks.append(attention_mask.cpu())

    acc = accuracy_score(all_has_harm_true, all_has_harm_pred)
    p, r, f1, _ = precision_recall_fscore_support(all_has_harm_true, all_has_harm_pred, average='binary', zero_division=0)

    start_logits_cat = torch.cat(all_start_logits, dim=0)
    end_logits_cat = torch.cat(all_end_logits, dim=0)
    start_labels_cat = torch.cat(all_start_labels, dim=0)
    end_labels_cat = torch.cat(all_end_labels, dim=0)
    attn_mask_cat = torch.cat(all_attention_masks, dim=0)
    span_p, span_r, span_f1 = compute_span_f1(start_logits_cat, end_logits_cat,
                                              start_labels_cat, end_labels_cat,
                                              attn_mask_cat)

    return total_loss / len(dataloader), acc, p, r, f1, span_p, span_r, span_f1

def train():
    wandb.init(project="harm-span-extraction", config={
        "max_len": cfg.max_len,
        "batch_size": cfg.batch_size,
        "epochs": cfg.epochs,
        "lr": cfg.lr,
        "warmup_ratio": cfg.warmup_ratio,
        "weight_decay": cfg.weight_decay,
        "negative_weight": cfg.negative_weight,
        "happy_penalty_weight": cfg.happy_penalty_weight,
    })

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # load training data (deterministic 90/10 split)
    df = pd.read_csv(cfg.train_file, encoding='utf-8')
    df = df.sample(frac=1, random_state=cfg.seed).reset_index(drop=True)
    split_idx = int(0.9 * len(df))
    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path)

    # load negative samples when enabled and available
    negative_df = None
    if cfg.use_negative_sampling and os.path.exists(cfg.negative_samples_file):
        negative_df = pd.read_csv(cfg.negative_samples_file, encoding='utf-8')
        print(f"Loaded {len(negative_df)} negative samples")
    else:
        print("No negative samples file found or use_negative_sampling=False")

    train_dataset = HarmSpanDataset(train_df, tokenizer, cfg.max_len, is_train=True, negative_df=negative_df)
    val_dataset = HarmSpanDataset(val_df, tokenizer, cfg.max_len, is_train=True)

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False)

    model = RoBERTaGateSpanExtractor(cfg.model_name_or_path).to(device)
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = len(train_loader) * cfg.epochs
    warmup_steps = int(total_steps * cfg.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    best_span_f1 = 0.0
    patience = 3
    patience_counter = 0

    for epoch in range(cfg.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        val_loss, acc, p, r, f1, span_p, span_r, span_f1 = evaluate(model, val_loader, device)

        wandb.log({
            "epoch": epoch+1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": acc,
            "val_precision": p,
            "val_recall": r,
            "val_f1": f1,
            "val_span_precision": span_p,
            "val_span_recall": span_r,
            "val_span_f1": span_f1,
        })

        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, "
              f"val_acc={acc:.4f}, val_f1={f1:.4f}, span_f1={span_f1:.4f}")

        if span_f1 > best_span_f1:
            best_span_f1 = span_f1
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(cfg.checkpoint_dir, "best_model.pth"))
            print(f"Saved best model with span_f1={span_f1:.4f}")
        else:
            patience_counter += 1
            print(f"No improvement for {patience_counter} epoch(s).")
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    wandb.finish()

if __name__ == "__main__":
    train()
