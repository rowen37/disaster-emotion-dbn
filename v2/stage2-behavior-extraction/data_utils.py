"""Dataset, character-offset/BIO alignment, decoding, and split loading."""
import json
import random

import pandas as pd
import torch
from torch.utils.data import Dataset

O, B, I = 0, 1, 2


def parse_json_list(value, field):
    if isinstance(value, list):
        return value
    if value is None or (isinstance(value, float) and pd.isna(value)) or value == "":
        return []
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{field} is not valid JSON: {value!r}") from exc
    if not isinstance(parsed, list):
        raise ValueError(f"{field} must be a JSON list")
    return parsed


def normalize_offsets(value, field):
    result = []
    for item in parse_json_list(value, field):
        if isinstance(item, dict) and {"start", "end"} <= set(item):
            item = (item["start"], item["end"])
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(f"{field} items must be [start, end]: {item!r}")
        result.append((int(item[0]), int(item[1])))
    return result


def offsets_to_bio(text, char_offsets, token_offsets, field):
    """Convert non-overlapping [start,end) character offsets to BIO labels."""
    labels = [-100 if a == b else O for a, b in token_offsets]
    occupied = set()
    for start, end in sorted(char_offsets):
        if not (0 <= start < end <= len(text)):
            raise ValueError(f"{field} offset outside text: {(start, end)}")
        token_ids = [i for i, (a, b) in enumerate(token_offsets)
                     if a != b and b > start and a < end]
        if not token_ids:
            raise ValueError(f"{field} offset lost after truncation: {(start, end)}")
        if any(i in occupied for i in token_ids):
            raise ValueError(f"overlapping {field} offsets near {(start, end)}")
        labels[token_ids[0]] = B
        for i in token_ids[1:]:
            labels[i] = I
        occupied.update(token_ids)
    return labels


def visible_annotations(text, spans, char_offsets, span_types, token_offsets):
    """Keep aligned span annotations fully represented after truncation."""
    visible_spans, visible_offsets, visible_types = [], [], []
    covered = [(a, b) for a, b in token_offsets if a != b]
    for span, (start, end), span_type in zip(spans, char_offsets, span_types):
        token_ids = [(a, b) for a, b in covered if b > start and a < end]
        if token_ids and token_ids[0][0] <= start and token_ids[-1][1] >= end:
            visible_spans.append(span)
            visible_offsets.append((start, end))
            visible_types.append(span_type)
    return visible_spans, visible_offsets, visible_types


def decode_bio(tag_ids, token_offsets, text):
    """Decode BIO tags into text spans and exact [start,end) offsets."""
    offsets, current = [], None
    for tag, (start, end) in zip(tag_ids, token_offsets):
        start, end = int(start), int(end)
        if start == end:
            continue
        if tag == B or (tag == I and current is None):
            if current is not None:
                offsets.append(current)
            current = [start, end]
        elif tag == I and current is not None:
            current[1] = max(current[1], end)
        elif current is not None:
            offsets.append(current)
            current = None
    if current is not None:
        offsets.append(current)
    offsets = [[int(a), int(b)] for a, b in offsets]
    return [text[a:b] for a, b in offsets], offsets


def drop_single_character_spans(spans, offsets):
    """Remove one-character predictions; the annotation scheme has none."""
    if len(spans) != len(offsets):
        raise ValueError("span/offset count mismatch")
    kept = [(span, offset) for span, offset in zip(spans, offsets)
            if int(offset[1]) - int(offset[0]) >= 2]
    return [span for span, _ in kept], [offset for _, offset in kept]


class BehaviorSpanDataset(Dataset):
    """One post may contain zero or more behavior spans."""

    def __init__(self, rows, tokenizer, max_len, augment=False, mask_prob=0.1):
        self.tokenizer = tokenizer
        self.augment = augment
        self.mask_prob = mask_prob
        self.examples = []
        for row in rows:
            text = str(row["text"]).strip()
            if not text:
                raise ValueError(f"empty text in record {row.get('record_id')}")
            enc = tokenizer(text, max_length=max_len, padding="max_length",
                            truncation=True, return_offsets_mapping=True)
            token_offsets = [(int(a), int(b)) for a, b in enc.pop("offset_mapping")]
            behavior_offsets = normalize_offsets(row.get("behavior_offsets"), "behavior_offsets")
            behavior_spans = parse_json_list(row.get("behavior_spans"), "behavior_spans")
            if len(behavior_spans) != len(behavior_offsets):
                raise ValueError(f"behavior span/offset count mismatch in {row.get('record_id')}")
            for span, (a, b) in zip(behavior_spans, behavior_offsets):
                if text[a:b] != span:
                    raise ValueError(f"behavior text/offset mismatch in {row.get('record_id')}")
            behavior_types = parse_json_list(row.get("behavior_types"), "behavior_types")
            if len(behavior_types) != len(behavior_spans):
                raise ValueError(f"behavior type/span count mismatch in {row.get('record_id')}")
            source_hb = int(row["has_behavior"])
            if source_hb != int(bool(behavior_offsets)):
                raise ValueError(f"presence flag mismatch in {row.get('record_id')}")
            original_behavior_n = len(behavior_offsets)
            behavior_spans, behavior_offsets, behavior_types = visible_annotations(
                text, behavior_spans, behavior_offsets, behavior_types, token_offsets)
            hb = int(bool(behavior_offsets))
            self.examples.append({
                "record_id": str(row["record_id"]), "text": text,
                "input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"],
                "token_offsets": token_offsets,
                "behavior_tags": offsets_to_bio(text, behavior_offsets, token_offsets, "behavior"),
                "behavior_offsets": [list(x) for x in behavior_offsets],
                "behavior_types": behavior_types,
                "has_behavior": hb,
                "truncated_behavior_spans": original_behavior_n - len(behavior_offsets),
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        input_ids = torch.tensor(ex["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(ex["attention_mask"], dtype=torch.long)
        bt = torch.tensor(ex["behavior_tags"], dtype=torch.long)
        if self.augment:
            for pos in range(len(input_ids)):
                # Preserve annotated tokens; augmentation affects context only.
                if (attention_mask[pos] and bt[pos] == O
                        and random.random() < self.mask_prob):
                    input_ids[pos] = self.tokenizer.mask_token_id
        return {
            "input_ids": input_ids, "attention_mask": attention_mask,
            "behavior_tags": bt,
            "has_behavior": torch.tensor(ex["has_behavior"], dtype=torch.long),
            "token_offsets": torch.tensor(ex["token_offsets"], dtype=torch.long),
            "text": ex["text"], "record_id": ex["record_id"],
        }


def load_split(data_file, folds_file):
    data = pd.read_csv(data_file, encoding="utf-8-sig", dtype={"record_id": str})
    folds = pd.read_csv(folds_file, encoding="utf-8-sig", dtype={"record_id": str})
    df = data.merge(folds, on="record_id", validate="one_to_one")
    if len(df) != len(data):
        raise ValueError("folds.csv does not cover every training record")
    return {int(fold): part.to_dict("records") for fold, part in df.groupby("fold")}


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
