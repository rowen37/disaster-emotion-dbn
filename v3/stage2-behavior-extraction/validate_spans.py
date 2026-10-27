"""Validate tokenizer alignment without changing or dropping annotations."""
import json

import pandas as pd
from transformers import AutoTokenizer

import config as cfg
from data_utils import BehaviorSpanDataset


def main():
    rows = pd.read_csv(cfg.DATA_TRAIN, encoding="utf-8-sig").to_dict("records")
    tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_NAME, use_fast=True)
    dataset = BehaviorSpanDataset(rows, tokenizer, cfg.MAX_LEN)
    raw_lengths = [len(tokenizer(ex["text"], add_special_tokens=True)["input_ids"])
                   for ex in dataset.examples]
    ordered = sorted(raw_lengths)
    quantile = lambda p: ordered[min(round(p * (len(ordered) - 1)), len(ordered) - 1)]
    report = {
        "rows": len(dataset), "model": cfg.MODEL_NAME, "max_len": cfg.MAX_LEN,
        "all_visible_spans_align": True,
        "token_length": {"p50": quantile(0.50), "p90": quantile(0.90),
                         "p99": quantile(0.99), "max": max(ordered)},
        "truncated_rows": sum(length > cfg.MAX_LEN for length in raw_lengths),
        "truncated_behavior_spans": sum(
            ex["truncated_behavior_spans"] for ex in dataset.examples),
    }
    with open(cfg.TOKEN_REPORT, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"written: {cfg.TOKEN_REPORT}")


if __name__ == "__main__":
    main()
