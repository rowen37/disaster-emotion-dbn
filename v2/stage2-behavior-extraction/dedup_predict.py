"""Memory-conscious prediction that scores each distinct post text once."""
import argparse
from pathlib import Path

from transformers import AutoTokenizer

import config as cfg
from data_utils import pick_device
from predict import STALE_COLS, load_models, predict_texts, read_table, write_table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=str(cfg.PREDICT_INPUT))
    parser.add_argument("--output-dir", default=str(cfg.PREDICT_OUTPUT))
    parser.add_argument("--model", choices=["final", "ensemble"], default="final")
    parser.add_argument("--threshold", type=float, default=None,
                        help="override the inner-validation threshold saved in the checkpoint")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()
    input_dir, output_dir = Path(args.input_dir), Path(args.output_dir)
    files = sorted(path for path in input_dir.iterdir()
                   if path.suffix.lower() in {".csv", ".rds"} and path.name != "eventlist.csv")
    frames, texts = {}, set()
    for path in files:
        frame = read_table(path)
        if "text" not in frame:
            raise ValueError(f"{path}: required column 'text' missing")
        frames[path.name] = frame
        texts.update(frame.text.fillna("").astype(str))
    ordered = sorted(texts, key=len)
    device = pick_device()
    tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_NAME, use_fast=True)
    models, saved_threshold = load_models(args.model, device)
    threshold = saved_threshold if args.threshold is None else args.threshold
    values = predict_texts(ordered, models, tokenizer, device, threshold, args.batch_size)
    mapping = {text: {key: column[i] for key, column in values.items()}
               for i, text in enumerate(ordered)}
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        frame = frame.drop(columns=[col for col in STALE_COLS if col in frame], errors="ignore")
        rows = [mapping[text] for text in frame.text.fillna("").astype(str)]
        for key in values:
            frame[key] = [row[key] for row in rows]
        frame["raw_prob"] = frame["behavior_prob"]  # Stage 3 reads raw_prob
        write_table(frame, output_dir / name)
        print(f"{name}: {len(frame)} rows")


if __name__ == "__main__":
    main()
