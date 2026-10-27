"""Presence and multi-span metrics for behavior extraction."""
import numpy as np
import torch

from data_utils import decode_bio, drop_single_character_spans


def prf(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def pr_auc(y_true, probs):
    y_true, probs = np.asarray(y_true), np.asarray(probs)
    if not y_true.sum():
        return 0.0
    order = np.argsort(-probs)
    y = y_true[order]
    precision = np.cumsum(y) / (np.arange(len(y)) + 1)
    recall = np.cumsum(y) / y.sum()
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def _channel_metrics(name, gold_presence, probs, gold_offsets, pred_offsets):
    gold_presence = np.asarray(gold_presence)
    # The CRF is the span decision. The presence head remains an auxiliary
    # continuous score for ranking and Stage 3, but never deletes CRF output.
    predicted_presence = np.asarray([bool(offsets) for offsets in pred_offsets], dtype=int)
    tp = int(((gold_presence == 1) & (predicted_presence == 1)).sum())
    fp = int(((gold_presence == 0) & (predicted_presence == 1)).sum())
    fn = int(((gold_presence == 1) & (predicted_presence == 0)).sum())
    tn = int(((gold_presence == 0) & (predicted_presence == 0)).sum())
    bp, br, bf = prf(tp, fp, fn)

    exact_tp = exact_fp = exact_fn = 0
    char_tp = char_fp = char_fn = 0
    for gold, pred in zip(gold_offsets, pred_offsets):
        gs = {tuple(pair) for pair in gold}
        ps = {tuple(pair) for pair in pred}
        exact_tp += len(gs & ps)
        exact_fp += len(ps - gs)
        exact_fn += len(gs - ps)
        gc = {pos for start, end in gs for pos in range(start, end)}
        pc = {pos for start, end in ps for pos in range(start, end)}
        char_tp += len(gc & pc)
        char_fp += len(pc - gc)
        char_fn += len(gc - pc)
    ep, er, ef = prf(exact_tp, exact_fp, exact_fn)
    cp, cr, cf = prf(char_tp, char_fp, char_fn)
    predicted_spans = [pair for offsets in pred_offsets for pair in offsets]
    single_char_spans = sum(end - start == 1 for start, end in predicted_spans)
    return {
        f"{name}_presence_acc": (tp + tn) / max(len(gold_presence), 1),
        f"{name}_presence_P": bp, f"{name}_presence_R": br,
        f"{name}_presence_F1": bf, f"{name}_PR_AUC": pr_auc(gold_presence, probs),
        f"{name}_span_exact_P": ep, f"{name}_span_exact_R": er,
        f"{name}_span_exact_F1": ef,
        f"{name}_span_char_P": cp, f"{name}_span_char_R": cr,
        f"{name}_span_char_F1": cf,
        f"{name}_predicted_spans": len(predicted_spans),
        f"{name}_single_char_spans": single_char_spans,
        f"{name}_single_char_span_rate": (
            single_char_spans / len(predicted_spans) if predicted_spans else 0.0),
    }


@torch.no_grad()
def evaluate_split(model, dataloader, device):
    model.eval()
    probs, gold_presence, predictions = [], [], []
    for batch in dataloader:
        output = model(batch["input_ids"].to(device), batch["attention_mask"].to(device))
        channel_probs = torch.softmax(output["has_behavior"], dim=-1)[:, 1].cpu()
        valid_mask = (batch["behavior_tags"][:, 1:] != -100).to(device)
        decoded = model.crf_decode(
            output["behavior_tags"][:, 1:], valid_mask).cpu()
        tag_ids = torch.zeros(
            (decoded.shape[0], decoded.shape[1] + 1), dtype=torch.long)
        tag_ids[:, 1:] = decoded
        probs.extend(channel_probs.tolist())
        gold_presence.extend(batch["has_behavior"].numpy().tolist())
        for i, text in enumerate(batch["text"]):
            spans, offsets = decode_bio(tag_ids[i].tolist(),
                                        batch["token_offsets"][i].tolist(), text)
            _, offsets = drop_single_character_spans(spans, offsets)
            predictions.append(offsets)

    examples = dataloader.dataset.examples
    metrics = {"n": len(examples)}
    metrics.update(_channel_metrics(
        "behavior", gold_presence, probs,
        [example["behavior_offsets"] for example in examples],
        predictions,
    ))
    # Character overlap stabilizes early training; exact match makes checkpoint
    # selection sensitive to complete boundaries rather than fragments alone.
    metrics["selection_span_F1"] = (
        metrics["behavior_span_char_F1"] + metrics["behavior_span_exact_F1"]
    ) / 2
    return metrics, probs, gold_presence
