"""Stage 1: add emotion labels and probabilities to post datasets.

The canonical output is ``../data/<hazard>.rds``. CSV and RDS inputs are
both supported; directory mode preserves each input filename. Stage 1 owns
only ``EMO`` and ``EMO_probs`` and replaces those columns when rerun.
"""

import argparse
import json
import os
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import BertForSequenceClassification, BertTokenizer


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT.parent / "data"
MODEL_PATH = os.environ.get("STAGE1_MODEL", "/Users/rogerbqw/Downloads/毕设数据/Thesis-ZJU/span_extract/resource/chinese-roberta-wwm-ext")
MODEL_WEIGHT = os.environ.get("STAGE1_WEIGHT", "/Users/rogerbqw/Downloads/毕设数据/Thesis-ZJU/SA_NER/checkpoints/best_cls_1.pt")
MAX_LEN = 64
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

label2id = {
    "fear": 0,
    "neutral": 1,
    "sad": 2,
    "surprise": 3,
    "angry": 4,
    "happy": 5,
}
id2label = {v: k for k, v in label2id.items()}


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() == ".rds":
        try:
            import pyreadr
        except ImportError as exc:
            raise RuntimeError("读取 RDS 需要安装 requirements.txt 中的 pyreadr") from exc
        objects = pyreadr.read_r(str(path))
        if len(objects) != 1:
            raise ValueError(f"{path}: RDS 必须只包含一个 data.frame")
        return next(iter(objects.values()))
    raise ValueError(f"不支持的输入格式: {path}")


def write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv":
        df.to_csv(path, index=False, encoding="utf-8-sig")
        return
    if path.suffix.lower() == ".rds":
        try:
            import pyreadr
        except ImportError as exc:
            raise RuntimeError("写入 RDS 需要安装 requirements.txt 中的 pyreadr") from exc
        pyreadr.write_rds(str(path), df, compress="gzip")
        return
    raise ValueError(f"不支持的输出格式: {path}")


def load_model():
    tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
    model = BertForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=6)
    model.load_state_dict(torch.load(MODEL_WEIGHT, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return tokenizer, model


def predict_with_probs(text, tokenizer, model):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=MAX_LEN,
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        probs = F.softmax(model(**inputs).logits, dim=1).cpu().numpy()[0]
    pred_id = int(probs.argmax())
    prob_dict = {id2label[i]: float(probs[i]) for i in range(len(id2label))}
    return id2label[pred_id], json.dumps(prob_dict, ensure_ascii=False)


def process_file(in_path: Path, out_path: Path, tokenizer, model) -> None:
    df = read_table(in_path)
    if "text" not in df.columns:
        raise ValueError(f"{in_path}: 缺少 text 列")
    df = df.drop(columns=[c for c in ("EMO", "EMO_probs") if c in df.columns])
    predictions = [
        predict_with_probs(text, tokenizer, model)
        for text in tqdm(df["text"].fillna("").astype(str), desc=in_path.name)
    ]
    df["EMO"] = [x[0] for x in predictions]
    df["EMO_probs"] = [x[1] for x in predictions]
    write_table(df, out_path)
    print(f"{in_path} -> {out_path} ({len(df)} posts)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="CSV/RDS 文件或包含它们的目录")
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help=f"Stage 1 输出文件或目录（默认: {DEFAULT_OUTPUT}）",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    tokenizer, model = load_model()

    if in_path.is_dir():
        files = sorted(
            p for p in in_path.iterdir()
            if p.suffix.lower() in {".csv", ".rds"} and p.name != "eventlist.csv"
        )
        if not files:
            raise FileNotFoundError(f"{in_path}: 没有 CSV/RDS 数据文件")
        out_path.mkdir(parents=True, exist_ok=True)
        for path in files:
            process_file(path, out_path / path.name, tokenizer, model)
    else:
        target = out_path / in_path.name if out_path.is_dir() else out_path
        process_file(in_path, target, tokenizer, model)


if __name__ == "__main__":
    main()
