#!/usr/bin/env python3
import os, json, math, argparse
from collections import defaultdict
from statistics import mean
import fnmatch
import pandas as pd

# ---- Canonical vocab ----
DATASETS = {
    "diabetes": "Diabetes",
    "heart_disease": "Heart Disease",
    "breast_cancer": "Breast Cancer",
    "obesity": "Obesity",
    # If you later add lung_cancer, uncomment:
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

def safe_mean(values):
    vals = []
    for v in values:
        if v is None or v == 0:
            continue
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            continue
        vals.append(v)
    return mean(vals) if vals else None


def compute_recall(precision, f1):
    if precision is None or f1 is None:
        return None
    if precision <= 0 or f1 <= 0:
        return None
    denom = 2*precision - f1
    if denom <= 0:
        return None
    return (f1 * precision) / denom

def _strip_prefix(s, prefix):
    if not s.startswith(prefix):
        return None
    return s[len(prefix):]

def _strip_suffix(s, suffix):
    if not s.endswith(suffix):
        return None
    return s[: -len(suffix)]

def _match_end_token(main_lower, candidates_lower):
    """
    Given a lowercase string main_lower and a dict of {lower: Canonical},
    return (canonical, main_without_token) if main_lower endswith '_' + lower token.
    Preference: longest token first (to avoid partial overlaps).
    """
    candidates_sorted = sorted(candidates_lower.keys(), key=len, reverse=True)
    for tok in candidates_sorted:
        suffix = "_" + tok
        if main_lower.endswith(suffix):
            cut = len(main_lower) - len(suffix)
            return candidates_lower[tok], cut
    return None, None

def parse_filename(path):
    base = os.path.basename(path)
    if not base.lower().endswith(".json"):
        return None

    # Grouped pattern: performance_metrics_Grouped_{dataset}_{model}_{grouping}_run{n}.json
    # Ungrouped:       performance_metrics_Ungrouped_{dataset}_{model}_ungrouped_run{n}.json
    if base.startswith("performance_metrics_Grouped_"):
        core = _strip_prefix(base, "performance_metrics_Grouped_")
        core = _strip_suffix(core, ".json")
        # Remove the trailing _run{n}
        idx = core.rfind("_run")
        if idx == -1:
            return None
        main = core[:idx]  # "{dataset}_{model}_{grouping}"
        # Work in lowercase for matching, but keep original for slicing
        ml = main.lower()

        # 1) Extract grouping by checking known grouping tokens at the end
        grouping, cut = _match_end_token(ml, {k: v for k, v in GROUPINGS.items() if k != "ungrouped"})
        if grouping is None:
            return None
        main_before_grouping = main[:cut]
        ml_before_grouping = ml[:cut]

        # 2) Extract model by checking known model tokens at the end
        model, cut2 = _match_end_token(ml_before_grouping, MODELS)
        if model is None:
            return None
        dataset_part = main_before_grouping[:cut2]  # whatever remains is dataset (may contain underscores)
        dataset_key = dataset_part.lower()
        dataset = DATASETS.get(dataset_key, None)
        if dataset is None:
            # Try a gentle fallback: replace any double underscores, trim, etc.
            dataset = DATASETS.get(dataset_key.replace("__","_").strip("_"), None)
        if dataset is None:
            return None

        return {
            "dataset": dataset,
            "model": model,
            "grouping": grouping,
        }

    if base.startswith("performance_metrics_Ungrouped_"):
        core = _strip_prefix(base, "performance_metrics_Ungrouped_")
        core = _strip_suffix(core, ".json")
        idx = core.rfind("_run")
        if idx == -1:
            return None
        main = core[:idx]  # "{dataset}_{model}_ungrouped"
        ml = main.lower()

        # remove trailing _ungrouped
        if not ml.endswith("_ungrouped"):
            return None
        main_before = main[: -len("_ungrouped")]
        ml_before = ml[: -len("_ungrouped")]

        # Extract model at end
        model, cut = _match_end_token(ml_before, MODELS)
        if model is None:
            return None
        dataset_part = main_before[:cut]
        dataset_key = dataset_part.lower()
        dataset = DATASETS.get(dataset_key, None)
        if dataset is None:
            dataset = DATASETS.get(dataset_key.replace("__","_").strip("_"), None)
        if dataset is None:
            return None

        return {
            "dataset": dataset,
            "model": model,
            "grouping": GROUPINGS["ungrouped"],
        }

    return None

def collect_records(root, pattern):
    files = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fnmatch.fnmatch(fn, pattern):
                files.append(os.path.join(dirpath, fn))

    records = []
    bad = []
    for f in files:
        meta = parse_filename(f)
        if not meta:
            bad.append(f)
            continue
        try:
            with open(f, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if not isinstance(data, list) or not data:
                bad.append(f)
                continue
            row = data[0]
            acc = row.get("Accuracy", None)
            prec = row.get("Precision", None)
            f1 = row.get("F1-score", None)
            rec = compute_recall(prec, f1)

            records.append({
                "dataset": meta["dataset"],
                "grouping": meta["grouping"],
                "model": meta["model"],
                "Accuracy": acc,
                "F1-score": f1,
                "Precision": prec,
                "Recall": rec,
            })
        except Exception:
            bad.append(f)
            continue

    # quick diagnostics
    if bad:
        print(f"Skipped {len(bad)} files due to parse/read issues (showing up to 10):")
        for b in bad[:10]:
            print("  -", os.path.basename(b))

    # counts per (dataset, grouping)
    counts = defaultdict(int)
    for r in records:
        counts[(r["dataset"], r["grouping"])] += 1
    if counts:
        print("Parsed counts per (dataset, grouping):")
        for (ds, gp), c in sorted(counts.items()):
            print(f"  {ds:15s} | {gp:12s} : {c}")
    return records

def pivot_average(records):
    # metric -> grouping -> dataset -> list of values (aggregates over models & runs)
    buckets = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    datasets = set()
    groupings = set()

    for r in records:
        datasets.add(r["dataset"])
        groupings.add(r["grouping"])
        for metric in ["Accuracy", "F1-score", "Precision", "Recall"]:
            buckets[metric][r["grouping"]][r["dataset"]].append(r[metric])

    # Average
    averaged = defaultdict(lambda: defaultdict(dict))
    for metric, gmap in buckets.items():
        for grouping, dmap in gmap.items():
            for dataset, vals in dmap.items():
                averaged[metric][grouping][dataset] = safe_mean(vals)

    # Order
    ordered_ds = [d for d in ORDERED_DATASETS if d in datasets] + sorted(list(datasets - set(ORDERED_DATASETS)))
    ordered_gp = [g for g in ORDERED_GROUPINGS if g in groupings] + sorted(list(groupings - set(ORDERED_GROUPINGS)))

    frames = {}
    for metric in ["Accuracy", "F1-score", "Precision", "Recall"]:
        df = pd.DataFrame(
            [[averaged[metric].get(g, {}).get(d, None) for d in ordered_ds] for g in ordered_gp],
            index=ordered_gp,
            columns=ordered_ds
        )
        frames[metric] = df
    return frames

def main():
    ap = argparse.ArgumentParser(description="Aggregate JSON metrics into averages per dataset/grouping.")
    ap.add_argument("--root", default=".", help="Root folder to search")
    ap.add_argument("--pattern", default="performance_metrics_*.json", help="Glob-like filename pattern")
    ap.add_argument("--out-prefix", default="aggregated_", help="Output prefix for CSV/JSON files")
    args = ap.parse_args()

    records = collect_records(args.root, args.pattern)
    if not records:
        print("No records found. Check --root and --pattern.")
        return

    frames = pivot_average(records)

    # Build JSON from ordered frames so it matches CSV exactly
    summary_ordered = {}
    for metric, df in frames.items():
        summary_ordered[metric] = {}
        for grouping in df.index:
            summary_ordered[metric][grouping] = {}
            for dataset in df.columns:
                val = df.loc[grouping, dataset]
                if pd.notna(val):
                    summary_ordered[metric][grouping][dataset] = float(val)

    out_json = args.out_prefix + "summary.json"
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(summary_ordered, fh, indent=2)

    # CSVs per metric
    for metric, df in frames.items():
        df.to_csv(f"{args.out_prefix}{metric.replace(' ', '_')}.csv", float_format="%.6f", na_rep="-")

    print(f"Wrote {out_json} and {len(frames)} CSV files.")
    print("Metrics saved for:", ", ".join(frames.keys()))

if __name__ == "__main__":
    main()
