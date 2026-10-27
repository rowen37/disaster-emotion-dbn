"""Aggregate completed CV metrics; this script performs no training."""
import json

import pandas as pd

import config as cfg


def main():
    if not cfg.METRICS_CSV.is_file():
        raise FileNotFoundError(f"{cfg.METRICS_CSV}; run CV training first")
    runs = pd.read_csv(cfg.METRICS_CSV)
    if "architecture_version" in runs:
        runs = runs[runs["architecture_version"] == cfg.RUN_ID].copy()
    metric_columns = [col for col in runs.columns if col not in {
        "architecture_version",
        "fold", "seed", "best_epoch", "n",
        "behavior_best_tau_on_test_diagnostic_only",
        "behavior_threshold",
    } and pd.api.types.is_numeric_dtype(runs[col])]
    summary = {
        "n_runs": len(runs), "folds": sorted(runs.fold.unique().tolist()),
        "seeds": sorted(runs.seed.unique().tolist()),
        "protocol": "event/text-grouped five-fold CV; inner group validation for early stopping",
        "metrics": {col: {"mean": float(runs[col].mean()),
                           "std": float(runs[col].std(ddof=1)) if len(runs) > 1 else 0.0}
                    for col in metric_columns},
        "note": "best-tau values are outer-test diagnostics and are not deployment-tuned thresholds",
    }
    cfg.RUNS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = cfg.RUNS_DIR / "evaluation_summary.json"
    md_path = cfg.RUNS_DIR / "evaluation_summary.md"
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    lines = ["| metric | mean | std |", "|---|---:|---:|"]
    for name, values in summary["metrics"].items():
        lines.append(f"| {name} | {values['mean']:.4f} | {values['std']:.4f} |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
