"""Validate the canonical 1000-row dataset and create leakage-safe CV folds.

This script never rewrites the annotations. Rows sharing an event_id are kept
in the same fold; rows without an event use text_hash as their group key.
"""
import hashlib
import json
import random
from collections import Counter, defaultdict

import pandas as pd

import config as cfg


def parse_json_list(value, field):
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

REQUIRED = {
    "record_id", "hazard_type", "event_id", "text", "behavior_spans",
    "behavior_offsets", "behavior_types", "has_behavior", "text_hash",
}


def validate_row(row):
    text = str(row.text)
    spans = parse_json_list(row.behavior_spans, "behavior_spans")
    offsets = normalize_offsets(row.behavior_offsets, "behavior_offsets")
    types = parse_json_list(row.behavior_types, "behavior_types")
    if not len(spans) == len(offsets) == len(types):
        raise ValueError(f"{row.record_id}: behavior span/offset/type count mismatch")
    previous_end = -1
    for span, (start, end) in sorted(zip(spans, offsets), key=lambda x: x[1]):
        if not 0 <= start < end <= len(text):
            raise ValueError(f"{row.record_id}: behavior offset outside text")
        if start < previous_end:
            raise ValueError(f"{row.record_id}: overlapping behavior offsets")
        if text[start:end] != span:
            raise ValueError(f"{row.record_id}: behavior offset/text mismatch")
        previous_end = end
    if int(row.has_behavior) != int(bool(spans)):
        raise ValueError(f"{row.record_id}: has_behavior disagrees with spans")


def group_key(row):
    event = "" if pd.isna(row.event_id) else str(row.event_id).strip()
    if event:
        return f"event:{row.hazard_type}:{event}"
    text_hash = "" if pd.isna(row.text_hash) else str(row.text_hash).strip()
    if not text_hash:
        text_hash = hashlib.sha256(str(row.text).encode("utf-8")).hexdigest()
    return f"text:{text_hash}"


def assign_folds(df, n_folds, seed):
    """Greedy group-stratification by hazard and the two binary labels."""
    work = df.copy()
    work["group_id"] = [group_key(r) for r in work.itertuples(index=False)]
    # Balance label combination, hazard, and source provenance together so a
    # fold cannot look artificially easy merely because it contains a larger
    # share of synthetic or original examples.
    work["stratum"] = [f"{h}|{source}|{int(b)}" for h, source, b in
                       zip(work.hazard_type, work.source_group, work.has_behavior)]
    strata = sorted(work.stratum.unique())
    total = Counter(work.stratum)
    target_size = len(work) / n_folds
    target = {s: total[s] / n_folds for s in strata}
    grouped = []
    for gid, part in work.groupby("group_id", sort=False):
        grouped.append((gid, len(part), Counter(part.stratum), part.record_id.tolist()))
    rng = random.Random(seed)
    rng.shuffle(grouped)
    grouped.sort(key=lambda x: (-x[1], -sum(1 / max(total[s], 1) for s in x[2])))
    fold_sizes = [0] * n_folds
    fold_counts = [Counter() for _ in range(n_folds)]
    fold_of = {}
    for gid, size, counts, record_ids in grouped:
        def score(fold):
            # Compare relative occupancy, not distance to the final target.
            # Distance-to-target wrongly rewards repeatedly filling one fold
            # early in the process; occupancy sends the next group to the
            # least-filled compatible fold.
            class_load = sum(
                (fold_counts[fold][s] + counts[s]) / max(target[s], 1)
                for s in counts
            ) / max(len(counts), 1)
            size_load = (fold_sizes[fold] + size) / max(target_size, 1)
            return class_load + 0.35 * size_load, fold_sizes[fold], fold
        chosen = min(range(n_folds), key=score)
        fold_sizes[chosen] += size
        fold_counts[chosen].update(counts)
        for record_id in record_ids:
            fold_of[record_id] = chosen
    folds = work[["record_id", "group_id"]].copy()
    folds["fold"] = folds.record_id.map(fold_of).astype(int)
    return folds


def main():
    data = pd.read_csv(cfg.DATA_TRAIN, encoding="utf-8-sig", dtype={"record_id": str})
    missing = sorted(REQUIRED - set(data.columns))
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    if data.record_id.duplicated().any():
        raise ValueError("record_id must be unique")
    if data.text_hash.astype(str).duplicated().any():
        raise ValueError("duplicate text_hash found in canonical training data")
    for row in data.itertuples(index=False):
        validate_row(row)

    folds = assign_folds(data, cfg.N_FOLDS, cfg.SPLIT_SEED)
    cfg.FOLDS_FILE.parent.mkdir(parents=True, exist_ok=True)
    folds.to_csv(cfg.FOLDS_FILE, index=False, encoding="utf-8-sig")
    merged = data.merge(folds, on="record_id", validate="one_to_one")
    if merged.groupby("group_id").fold.nunique().max() != 1:
        raise AssertionError("group leakage across folds")
    per_fold = []
    for fold, part in merged.groupby("fold"):
        per_fold.append({
            "fold": int(fold), "n": int(len(part)),
            "behavior_positive": int(part.has_behavior.sum()),
            "events_or_text_groups": int(part.group_id.nunique()),
            "hazards": {str(k): int(v) for k, v in part.hazard_type.value_counts().items()},
        })
    report = {
        "input": str(cfg.DATA_TRAIN), "rows": int(len(data)),
        "unique_record_ids": int(data.record_id.nunique()),
        "unique_text_hashes": int(data.text_hash.nunique()),
        "behavior_positive": int(data.has_behavior.sum()),
        "groups": int(folds.group_id.nunique()), "group_leakage": False,
        "n_folds": cfg.N_FOLDS, "split_seed": cfg.SPLIT_SEED,
        "per_fold": per_fold,
    }
    with open(cfg.DATA_REPORT, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"written: {cfg.FOLDS_FILE}\n         {cfg.DATA_REPORT}")


if __name__ == "__main__":
    main()
