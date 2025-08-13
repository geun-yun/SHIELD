#!/usr/bin/env python3
import os, json, math, argparse, fnmatch
from collections import defaultdict
from statistics import mean
import pandas as pd

# Canonical vocab
DATASETS = {
    "diabetes": "Diabetes",
    "heart_disease": "Heart Disease",
    "breast_cancer": "Breast Cancer",
    "obesity": "Obesity",
    # "lung_cancer": "Lung Cancer",
}

MODELS = {
    "xgboost": "XGBoost",
    "randomforest": "RandomForest",
    "mlp": "MLP",
    "logisticregression": "LogisticRegression",
    "svm": "SVM",
}

GROUPINGS = {
    "bicriterion": "Bicriterion",
    "kplus": "K-plus",
    "random": "Random",
    "group_dissimilar": "Greedy",
    "benchmark": "Benchmark",
    "ungrouped": "Ungrouped",
}

ORDERED_DATASETS = ["Breast Cancer", "Heart Disease", "Lung Cancer", "Obesity", "Diabetes"]
ORDERED_GROUPINGS = ["Benchmark", "Ungrouped", "Random", "Greedy", "K-plus", "Bicriterion"]

# Fairness metric keys as they appear in files
FAIRNESS_KEYS = [
    "Disparate Impact",
    "Equal Opportunity",
    "Equalized Odds",
    "Predictive Parity",
    "N-Sigma (ErrorRate)",
    "Average distance from origin",
]

def safe_mean(values):
    vals = []
    for v in values:
        if v is None or v == 0:
            continue
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            continue
        # Ignore extremely small numbers close to zero
        if isinstance(v, float) and abs(v) < 1e-8:
            continue
        vals.append(v)
    return mean(vals) if vals else None


def _strip_prefix(s, prefix):
    if not s.startswith(prefix):
        return None
    return s[len(prefix):]

def _strip_suffix(s, suffix):
    if not s.endswith(suffix):
        return None
    return s[: -len(suffix)]

def _match_end_token(main_lower, candidates):
    # candidates is dict of token_lower -> Canonical
    toks = sorted(candidates.keys(), key=len, reverse=True)
    for tok in toks:
        suf = "_" + tok
        if main_lower.endswith(suf):
            cut = len(main_lower) - len(suf)
            return candidates[tok], cut
    return None, None

def parse_filename(path):
    base = os.path.basename(path)
    if not base.lower().endswith(".json"):
        return None

    if not base.startswith("fairness_metrics_"):
        return None

    core = base[len("fairness_metrics_"):-len(".json")]
    # remove _runX
    idx = core.rfind("_run")
    if idx == -1:
        return None
    main = core[:idx]  # "{dataset}_{model}_{grouping}"
    ml = main.lower()

    # 1) Extract grouping first
    grouping, cut = _match_end_token(ml, GROUPINGS)
    if grouping is None:
        return None
    main_before_grouping = main[:cut]
    ml_before_grouping = ml[:cut]

    # 2) Extract model
    model, cut2 = _match_end_token(ml_before_grouping, MODELS)
    if model is None:
        return None

    dataset_part = main_before_grouping[:cut2]
    dataset = DATASETS.get(dataset_part.lower(), None)
    if dataset is None:
        dataset = DATASETS.get(dataset_part.lower().replace("__","_").strip("_"), None)
    if dataset is None:
        return None

    return {
        "dataset": dataset,
        "model": model,
        "grouping": grouping,
    }


def collect_records(root, pattern):
    files = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fnmatch.fnmatch(fn, pattern):
                files.append(os.path.join(dirpath, fn))

    records, bad = [], []
    for f in files:
        meta = parse_filename(f)
        if not meta:
            bad.append(f)
            continue
        try:
            with open(f, "r", encoding="utf-8") as fh:
                row = json.load(fh)  # fairness files are dicts
            if not isinstance(row, dict):
                bad.append(f)
                continue

            rec = {
                "dataset": meta["dataset"],
                "grouping": meta["grouping"],
                "model": meta["model"],
            }
            for k in FAIRNESS_KEYS:
                rec[k] = row.get(k, None)
            records.append(rec)
        except Exception:
            bad.append(f)
            continue

    if bad:
        print(f"Skipped {len(bad)} files due to parse or read issues (showing up to 10):")
        for b in bad[:10]:
            print("  -", os.path.basename(b))

    counts = defaultdict(int)
    for r in records:
        counts[(r["dataset"], r["grouping"])] += 1
    if counts:
        print("Parsed counts per (dataset, grouping):")
        for (ds, gp), c in sorted(counts.items()):
            print(f"  {ds:15s} | {gp:12s} : {c}")
    return records

def pivot_average(records):
    # metric -> grouping -> dataset -> list
    buckets = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    datasets, groupings = set(), set()

    for r in records:
        datasets.add(r["dataset"])
        groupings.add(r["grouping"])
        for k in FAIRNESS_KEYS:
            buckets[k][r["grouping"]][r["dataset"]].append(r.get(k))

    averaged = defaultdict(lambda: defaultdict(dict))
    for metric, gmap in buckets.items():
        for grouping, dmap in gmap.items():
            for dataset, vals in dmap.items():
                averaged[metric][grouping][dataset] = safe_mean(vals)

    ordered_ds = [d for d in ORDERED_DATASETS if d in datasets] + sorted(list(datasets - set(ORDERED_DATASETS)))
    ordered_gp = [g for g in ORDERED_GROUPINGS if g in groupings] + sorted(list(groupings - set(ORDERED_GROUPINGS)))

    frames = {}
    for metric in FAIRNESS_KEYS:
        df = pd.DataFrame(
            [[averaged[metric].get(g, {}).get(d, None) for d in ordered_ds] for g in ordered_gp],
            index=ordered_gp,
            columns=ordered_ds
        )
        frames[metric] = df
    return frames

def main():
    ap = argparse.ArgumentParser(description="Aggregate fairness JSON metrics into averages per dataset and grouping.")
    ap.add_argument("--root", default=".", help="Root folder to search")
    ap.add_argument("--pattern", default="fairness_metrics_*.json", help="Glob-like filename pattern")
    ap.add_argument("--out-prefix", default="fairness_aggregated_", help="Output prefix for CSV and JSON files")
    args = ap.parse_args()

    records = collect_records(args.root, args.pattern)
    if not records:
        print("No fairness records found. Check --root and --pattern.")
        return

    frames = pivot_average(records)

    # Build JSON summary matching CSV order
    summary = {}
    for metric, df in frames.items():
        summary[metric] = {}
        for grouping in df.index:
            summary[metric][grouping] = {}
            for dataset in df.columns:
                val = df.loc[grouping, dataset]
                if pd.notna(val):
                    summary[metric][grouping][dataset] = float(val)

    out_json = args.out_prefix + "summary.json"
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    for metric, df in frames.items():
        safe_name = metric.replace(" ", "_").replace("-", "_").replace("(", "").replace(")", "")
        df.to_csv(f"{args.out_prefix}{safe_name}.csv", float_format="%.6f", na_rep="-")

    print(f"Wrote {out_json} and {len(frames)} CSV files.")
    print("Fairness metrics saved for:", ", ".join(frames.keys()))

if __name__ == "__main__":
    main()
