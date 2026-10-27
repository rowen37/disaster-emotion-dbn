"""Presence and multi-span metrics for behavior extraction."""
import numpy as np
import torch

from data_utils import decode_bio, drop_single_character_spans

FOCUS_BEHAVIOR_TYPE = "失范扰序"


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


def best_f1_threshold(y_true, probs):
    best = (0.5, -1.0)
    y_true, probs = np.asarray(y_true), np.asarray(probs)
    for tau in np.linspace(0.0, 1.0, 101):
        pred = (probs > tau).astype(int)
        tp = int(((y_true == 1) & (pred == 1)).sum())
        fp = int(((y_true == 0) & (pred == 1)).sum())
        fn = int(((y_true == 1) & (pred == 0)).sum())
        f1 = prf(tp, fp, fn)[2]
        if f1 > best[1]:
            best = (float(tau), f1)
    return best


def _channel_metrics(name, gold_presence, probs, gold_offsets, pred_offsets, threshold):
    gold_presence = np.asarray(gold_presence)
    predicted_presence = (np.asarray(probs) > threshold).astype(int)
    gated_offsets = [offsets if fire else []
                     for offsets, fire in zip(pred_offsets, predicted_presence)]
    tp = int(((gold_presence == 1) & (predicted_presence == 1)).sum())
    fp = int(((gold_presence == 0) & (predicted_presence == 1)).sum())
    fn = int(((gold_presence == 1) & (predicted_presence == 0)).sum())
    tn = int(((gold_presence == 0) & (predicted_presence == 0)).sum())
    bp, br, bf = prf(tp, fp, fn)

    exact_tp = exact_fp = exact_fn = 0
    char_tp = char_fp = char_fn = 0
    for gold, pred in zip(gold_offsets, gated_offsets):
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
    predicted_spans = [pair for offsets in gated_offsets for pair in offsets]
    single_char_spans = sum(end - start == 1 for start, end in predicted_spans)
    unresolved = sum(bool(fire) and not offsets
                     for offsets, fire in zip(pred_offsets, predicted_presence))
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
        f"{name}_positive_unresolved": unresolved,
    }


def _focus_type_metrics(examples, probs, pred_offsets, threshold):
    """Evaluate unified span predictions on posts containing the focus type.

    The model does not predict types. Therefore this is a strict slice metric:
    on posts with at least one 失范扰序 gold span, all predicted behavior spans
    are compared only with the 失范扰序 gold spans. Predictions of another
    behavior type in a mixed post count as false positives.
    """
    predicted_presence = (np.asarray(probs) > threshold).astype(int)
    exact_tp = exact_fp = exact_fn = 0
    char_tp = char_fp = char_fn = 0
    subset_n = gold_span_n = 0
    for example, pred, fire in zip(examples, pred_offsets, predicted_presence):
        gold = [tuple(offset) for offset, span_type in zip(
            example["behavior_offsets"], example["behavior_types"]
        ) if span_type == FOCUS_BEHAVIOR_TYPE]
        if not gold:
            continue
        subset_n += 1
        gold_span_n += len(gold)
        gs = set(gold)
        ps = {tuple(pair) for pair in pred} if fire else set()
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
    return {
        "disorder_subset_n": subset_n,
        "disorder_gold_spans": gold_span_n,
        "disorder_span_exact_P": ep,
        "disorder_span_exact_R": er,
        "disorder_span_exact_F1": ef,
        "disorder_span_char_P": cp,
        "disorder_span_char_R": cr,
        "disorder_span_char_F1": cf,
    }


@torch.no_grad()
def evaluate_split(model, dataloader, device, threshold=0.5):
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
    if threshold is None:
        threshold, _ = best_f1_threshold(gold_presence, probs)
    metrics = {"n": len(examples)}
    metrics.update(_channel_metrics(
        "behavior", gold_presence, probs,
        [example["behavior_offsets"] for example in examples],
        predictions, threshold,
    ))
    metrics.update(_focus_type_metrics(examples, probs, predictions, threshold))
    metrics["behavior_threshold"] = float(threshold)
    # Character overlap stabilizes early training; exact match makes checkpoint
    # selection sensitive to complete boundaries rather than fragments alone.
    metrics["selection_span_F1"] = (
        metrics["behavior_span_char_F1"] + metrics["behavior_span_exact_F1"]
    ) / 2
    return metrics, probs, gold_presence
