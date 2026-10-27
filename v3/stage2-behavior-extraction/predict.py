"""Predict behavior presence probability and behavior spans for CSV/RDS datasets."""
import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

import config as cfg
from data_utils import decode_bio, drop_single_character_spans, pick_device
from model import BehaviorSpanExtractor

STALE_COLS = [
    "raw_prob", "has_harm", "pred_span", "calibrated_prob", "neg_sum", "fused_prob",
    "behavior_prob", "has_behavior", "behavior_spans", "behavior_offsets",
    "behavior_span_fallback",
    "behavior_span_unresolved",
    "consequence_prob", "has_consequence", "consequence_spans", "consequence_offsets",
    "consequence_span_fallback",
    "consequence_span_unresolved",
]


def read_table(path):
    path = Path(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() == ".rds":
        import pyreadr
        objects = pyreadr.read_r(str(path))
        if len(objects) != 1:
            raise ValueError(f"{path}: RDS must contain exactly one data.frame")
        return next(iter(objects.values()))
    raise ValueError(f"unsupported input format: {path}")


def write_table(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv":
        frame.to_csv(path, index=False, encoding="utf-8-sig")
    elif path.suffix.lower() == ".rds":
        import pyreadr
        pyreadr.write_rds(str(path), frame, compress="gzip")
    else:
        raise ValueError(f"unsupported output format: {path}")


def load_models(mode, device):
    paths = ([cfg.CKPT_DIR / "final_model.pth"] if mode == "final" else
             [cfg.CKPT_DIR / f"fold{fold}.pth" for fold in range(cfg.N_FOLDS)])
    models = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"{path} missing; train the requested model first")
        payload = torch.load(path, map_location=device)
        if (not isinstance(payload, dict)
                or payload.get("format_version") != cfg.CHECKPOINT_FORMAT_VERSION):
            raise ValueError(f"{path} has an incompatible checkpoint format")
        if int(payload.get("max_len", -1)) != cfg.MAX_LEN:
            raise ValueError(
                f"{path}: checkpoint max_len={payload.get('max_len')} but config max_len={cfg.MAX_LEN}"
            )
        model = BehaviorSpanExtractor(payload["model_name"], cfg.DROPOUT, cfg.N_TAGS).to(device)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        models.append(model)
    return models


@torch.no_grad()
def predict_texts(texts, models, tokenizer, device, batch_size):
    result = {key: [] for key in (
        "behavior_prob", "has_behavior", "behavior_spans", "behavior_offsets",
        "behavior_span_fallback",
        "behavior_span_unresolved",
    )}
    for start in tqdm(range(0, len(texts), batch_size), desc="predicting"):
        original = texts[start:start + batch_size]
        fed = [text if text.strip() else "空" for text in original]
        encoded = tokenizer(fed, max_length=cfg.MAX_LEN, padding=True, truncation=True,
                            return_offsets_mapping=True, return_tensors="pt")
        offsets = encoded.pop("offset_mapping")
        encoded = encoded.to(device)
        sums = {}
        for model in models:
            output = model(encoded["input_ids"], encoded["attention_mask"])
            for key, value in output.items():
                sums[key] = value if key not in sums else sums[key] + value
        averaged = {key: value / len(models) for key, value in sums.items()}
        probabilities = torch.softmax(averaged["has_behavior"], -1)[:, 1].cpu()
        valid_mask = (offsets[:, 1:, 1] > offsets[:, 1:, 0]).to(device)
        ensemble_parameters = None
        if len(models) > 1:
            crfs = [model.behavior_crf for model in models]
            ensemble_parameters = (
                torch.stack([crf.start_transitions for crf in crfs]).mean(0),
                torch.stack([crf.end_transitions for crf in crfs]).mean(0),
                torch.stack([crf.transitions for crf in crfs]).mean(0),
            )
        # Some cleaned posts have no tokenizable characters. Leave them all-O.
        nonempty = valid_mask.any(dim=1)
        decoded = torch.zeros_like(valid_mask, dtype=torch.long, device=device)
        if bool(nonempty.any()):
            decoded[nonempty] = models[0].crf_decode(
                averaged["behavior_tags"][nonempty, 1:],
                valid_mask[nonempty], ensemble_parameters=ensemble_parameters)
        decoded = decoded.cpu()
        tags = torch.zeros((decoded.shape[0], decoded.shape[1] + 1), dtype=torch.long)
        tags[:, 1:] = decoded
        for i, text in enumerate(original):
            spans, char_offsets = decode_bio(tags[i].tolist(), offsets[i].tolist(), text)
            spans, char_offsets = drop_single_character_spans(spans, char_offsets)
            has_behavior = bool(spans)
            result["behavior_prob"].append(float(probabilities[i]))
            result["has_behavior"].append(has_behavior)
            result["behavior_spans"].append(json.dumps(spans, ensure_ascii=False))
            result["behavior_offsets"].append(json.dumps(
                [{"start": a, "end": b} for a, b in char_offsets], ensure_ascii=False))
            result["behavior_span_fallback"].append(False)
            result["behavior_span_unresolved"].append(False)
    return result


def predict_file(input_path, output_path, models, tokenizer, device, batch_size):
    frame = read_table(input_path)
    if "text" not in frame:
        raise ValueError(f"{input_path}: required column 'text' missing")
    frame = frame.drop(columns=[col for col in STALE_COLS if col in frame], errors="ignore")
    values = predict_texts(frame.text.fillna("").astype(str).tolist(), models, tokenizer,
                           device, batch_size)
    for key, value in values.items():
        frame[key] = value
    write_table(frame, output_path)
    print(f"{input_path} -> {output_path} ({len(frame)} posts)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(cfg.PREDICT_INPUT))
    parser.add_argument("--output", default=str(cfg.PREDICT_OUTPUT))
    parser.add_argument("--model", choices=["final", "ensemble"], default="final")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    device = pick_device()
    tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_NAME, use_fast=True)
    models = load_models(args.model, device)
    input_path, output_path = Path(args.input), Path(args.output)
    if input_path.is_dir():
        output_path.mkdir(parents=True, exist_ok=True)
        files = sorted(path for path in input_path.iterdir()
                       if path.suffix.lower() in {".csv", ".rds"} and path.name != "eventlist.csv")
        for path in files:
            predict_file(path, output_path / path.name, models, tokenizer, device,
                         args.batch_size)
    else:
        predict_file(input_path, output_path, models, tokenizer, device,
                     args.batch_size)


if __name__ == "__main__":
    main()
