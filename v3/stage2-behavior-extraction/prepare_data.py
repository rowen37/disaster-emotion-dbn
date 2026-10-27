"""Validate canonical annotations and the leakage-safe five-fold split.

The training CSV intentionally contains no source-dependent training level.
Existing event/text groups in folds.csv are the canonical leakage boundary.
Use --rebalance only to make the smallest group-safe moves needed for equal
fold sizes; annotation rows are never rewritten by this script.
"""
import argparse
import json
from collections import Counter

import pandas as pd

import config as cfg


REQUIRED = {
    "record_id", "hazard_type", "text", "behavior_spans",
    "behavior_offsets", "behavior_types", "has_behavior", "text_hash",
    "is_synthetic",
}
FOLD_REQUIRED = {"record_id", "group_id", "fold"}


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


def validate_row(row):
    text = str(row.text)
    spans = parse_json_list(row.behavior_spans, "behavior_spans")
    offsets = normalize_offsets(row.behavior_offsets, "behavior_offsets")
    types = parse_json_list(row.behavior_types, "behavior_types")
    if not len(spans) == len(offsets) == len(types):
        raise ValueError(f"{row.record_id}: behavior span/offset/type count mismatch")
    previous_end = -1
    for span, (start, end) in sorted(zip(spans, offsets), key=lambda item: item[1]):
        if not 0 <= start < end <= len(text):
            raise ValueError(f"{row.record_id}: behavior offset outside text")
        if start < previous_end:
            raise ValueError(f"{row.record_id}: overlapping behavior offsets")
        if text[start:end] != span:
            raise ValueError(f"{row.record_id}: behavior offset/text mismatch")
        if len(span) < 2:
            raise ValueError(f"{row.record_id}: one-character harmful span")
        previous_end = end
    if int(row.has_behavior) != int(bool(spans)):
        raise ValueError(f"{row.record_id}: has_behavior disagrees with spans")
    unexpected = sorted(set(types) - {cfg.TARGET_BEHAVIOR_TYPE})
    if unexpected:
        raise ValueError(f"{row.record_id}: non-target behavior types remain: {unexpected}")
    if not str(row.text_hash).strip():
        raise ValueError(f"{row.record_id}: empty text_hash")
    if int(row.is_synthetic) not in (0, 1):
        raise ValueError(f"{row.record_id}: is_synthetic must be 0 or 1")


def validate_coverage(data, folds):
    missing = sorted(FOLD_REQUIRED - set(folds.columns))
    if missing:
        raise ValueError(f"folds.csv missing required columns: {missing}")
    if folds.record_id.duplicated().any():
        raise ValueError("folds.csv record_id must be unique")
    data_ids, fold_ids = set(data.record_id), set(folds.record_id)
    if data_ids != fold_ids:
        raise ValueError(
            f"fold coverage mismatch: missing={len(data_ids-fold_ids)}, extra={len(fold_ids-data_ids)}")
    expected = set(range(cfg.N_FOLDS))
    observed = set(folds.fold.astype(int))
    if observed != expected:
        raise ValueError(f"fold IDs must be {sorted(expected)}, got {sorted(observed)}")


def rebalance_folds(data, folds):
    """Make minimal whole-group moves until every fold has the same row count."""
    if len(data) % cfg.N_FOLDS:
        raise ValueError("row count must be divisible by the number of folds")
    target_size = len(data) // cfg.N_FOLDS
    merged = data.merge(folds, on="record_id", validate="one_to_one")
    if merged.groupby("group_id").fold.nunique().max() != 1:
        raise ValueError("group leakage already exists in folds.csv")
    groups = {group_id: part for group_id, part in merged.groupby("group_id", sort=False)}
    assignment = {group_id: int(part.fold.iloc[0]) for group_id, part in groups.items()}
    strata_total = Counter(zip(merged.hazard_type, merged.has_behavior.astype(int)))
    target_strata = {key: count / cfg.N_FOLDS for key, count in strata_total.items()}

    def state():
        sizes = [0] * cfg.N_FOLDS
        counts = [Counter() for _ in range(cfg.N_FOLDS)]
        for group_id, part in groups.items():
            fold = assignment[group_id]
            sizes[fold] += len(part)
            counts[fold].update(zip(part.hazard_type, part.has_behavior.astype(int)))
        return sizes, counts

    sizes, counts = state()
    while max(sizes) > target_size:
        donor = max(range(cfg.N_FOLDS), key=lambda fold: sizes[fold] - target_size)
        receiver = max(range(cfg.N_FOLDS), key=lambda fold: target_size - sizes[fold])
        transfer = min(sizes[donor] - target_size, target_size - sizes[receiver])
        candidates = []
        for group_id, part in groups.items():
            if assignment[group_id] != donor or len(part) != transfer:
                continue
            group_counts = Counter(zip(part.hazard_type, part.has_behavior.astype(int)))
            error = 0.0
            for fold, direction in ((donor, -1), (receiver, 1)):
                for key, target in target_strata.items():
                    value = counts[fold][key] + direction * group_counts[key]
                    error += (value - target) ** 2 / max(target, 1)
            candidates.append((error, group_id, group_counts))
        if not candidates:
            raise ValueError(
                f"cannot move a whole group of {transfer} rows from fold {donor} to {receiver}")
        _, group_id, group_counts = min(candidates)
        assignment[group_id] = receiver
        sizes[donor] -= transfer
        sizes[receiver] += transfer
        counts[donor].subtract(group_counts)
        counts[receiver].update(group_counts)

    if sizes != [target_size] * cfg.N_FOLDS:
        raise ValueError(f"folds are not equal after rebalance: {sizes}")
    result = folds.copy()
    result["fold"] = result.group_id.map(assignment).astype(int)
    return result


def build_report(data, folds):
    merged = data.merge(folds, on="record_id", validate="one_to_one")
    if merged.groupby("group_id").fold.nunique().max() != 1:
        raise ValueError("group leakage across folds")
    target_size = len(data) // cfg.N_FOLDS
    sizes = merged.groupby("fold").size().to_dict()
    if any(sizes.get(fold) != target_size for fold in range(cfg.N_FOLDS)):
        raise ValueError(f"fold sizes must all be {target_size}; run prepare_data.py --rebalance")
    per_fold = []
    for fold, part in merged.groupby("fold", sort=True):
        per_fold.append({
            "fold": int(fold), "n": int(len(part)),
            "behavior_positive": int(part.has_behavior.sum()),
            "events_or_text_groups": int(part.group_id.nunique()),
            "hazards": {str(key): int(value)
                        for key, value in part.hazard_type.value_counts().items()},
        })
    return {
        "input": str(cfg.DATA_TRAIN), "rows": int(len(data)),
        "unique_record_ids": int(data.record_id.nunique()),
        "unique_text_hashes": int(data.text_hash.nunique()),
        "behavior_positive": int(data.has_behavior.sum()),
        "behavior_spans": int(sum(len(parse_json_list(value, "behavior_spans"))
                                  for value in data.behavior_spans)),
        "target_behavior_type": cfg.TARGET_BEHAVIOR_TYPE,
        "groups": int(folds.group_id.nunique()), "group_leakage": False,
        "source_dependent_training": False,
        "n_folds": cfg.N_FOLDS, "per_fold": per_fold,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebalance", action="store_true",
                        help="minimally rebalance existing groups to equal fold sizes")
    args = parser.parse_args()
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

    folds = pd.read_csv(cfg.FOLDS_FILE, encoding="utf-8-sig", dtype={"record_id": str})
    validate_coverage(data, folds)
    if args.rebalance:
        folds = rebalance_folds(data, folds)
        temporary = cfg.FOLDS_FILE.with_suffix(cfg.FOLDS_FILE.suffix + ".tmp")
        folds.to_csv(temporary, index=False, encoding="utf-8-sig")
        temporary.replace(cfg.FOLDS_FILE)
        print(f"rebalanced: {cfg.FOLDS_FILE}")
    print(json.dumps(build_report(data, folds), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
