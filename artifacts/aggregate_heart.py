#!/usr/bin/env python3
import os, json, fnmatch, argparse, math
from collections import defaultdict, OrderedDict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

MARKERS = OrderedDict([
    ("Ungrouped", "o"),
    ("Random", "x"),
    ("Greedy", "^"),
    ("K-plus", "s"),
    ("Bicriterion", "D"),
])

COLORS = {
    "Ungrouped": "#1b9e77",
    "Random": "#d95f02",
    "Greedy": "#7570b3",
    "K-plus": "#e7298a",
    "Bicriterion": "#66a61e",
}

# small vertical jitter to reduce total overlap at identical y
JITTER = {
    "Ungrouped": -0.12,
    "Random": -0.06,
    "Greedy":  0.00,
    "K-plus":  +0.06,
    "Bicriterion": +0.12,
}


DATASETS = {
    "diabetes": "Diabetes",
    "heart_disease": "Heart Disease",
    "breast_cancer": "Breast Cancer",
    "obesity": "Obesity",
}
MODELS = {
    "xgboost": "Xgboost Classification",
    "randomforest": "Random Forest Classification",
    "mlp": "Neural Network Classification",
    "logisticregression": "Logistic Regression",
    "svm": "Support Vector Classification",
}
GROUPINGS = {
    "ungrouped": "Ungrouped",
    "random": "Random",
    "group_dissimilar": "Greedy",
    "kplus": "K-plus",
    "bicriterion": "Bicriterion",
    "benchmark": "Benchmark",
}
GROUPING_ORDER = ["Ungrouped", "Random", "Greedy", "K-plus", "Bicriterion", "Benchmark"]

def is_valid(v):
    return v is not None and v != 0 and not (isinstance(v, float) and (math.isnan(v) or math.isinf(v)))

def parse_filename(path: str):
    base = os.path.basename(path)
    if not base.endswith(".json"):
        return None
    name = base[:-5]

    if name.startswith("performance_metrics_Grouped_"):
        core = name[len("performance_metrics_Grouped_"):]
        if "_run" not in core: return None
        core = core[: core.rfind("_run")]
        # grouping from right
        grouping = None
        for gkey, gpretty in sorted(GROUPINGS.items(), key=lambda kv: len(kv[0]), reverse=True):
            if gkey == "ungrouped":
                continue
            suf = "_" + gkey
            if core.lower().endswith(suf):
                grouping = gpretty
                core = core[: -len(suf)]
                break
        if grouping is None: return None
        # model from right
        model = None
        for mkey, mpretty in sorted(MODELS.items(), key=lambda kv: len(kv[0]), reverse=True):
            suf = "_" + mkey
            if core.lower().endswith(suf):
                model = mpretty
                dataset_key = core[: -len(suf)].lower()
                break
        if model is None: return None
        dataset = DATASETS.get(dataset_key)
        if dataset is None: return None
        return {"dataset": dataset, "model": model, "grouping": grouping}

    if name.startswith("performance_metrics_Ungrouped_"):
        core = name[len("performance_metrics_Ungrouped_"):]
        if "_run" not in core: return None
        core = core[: core.rfind("_run")]
        if not core.lower().endswith("_ungrouped"): return None
        core = core[: -len("_ungrouped")]
        model = None
        for mkey, mpretty in sorted(MODELS.items(), key=lambda kv: len(kv[0]), reverse=True):
            suf = "_" + mkey
            if core.lower().endswith(suf):
                model = mpretty
                dataset_key = core[: -len(suf)].lower()
                break
        if model is None: return None
        dataset = DATASETS.get(dataset_key)
        if dataset is None: return None
        return {"dataset": dataset, "model": model, "grouping": GROUPINGS["ungrouped"]}

    return None

def collect_heart_accuracy(root, pattern):
    rows = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if not fnmatch.fnmatch(fn, pattern):
                continue
            meta = parse_filename(os.path.join(dirpath, fn))
            # ✅ filter to Heart Disease
            if not meta or meta["dataset"] != "Breast Cancer":
                continue
            try:
                with open(os.path.join(dirpath, fn), "r", encoding="utf-8") as f:
                    data = json.load(f)
                if not isinstance(data, list) or not data:
                    continue
                acc = data[0].get("Precision")
                if is_valid(acc):
                    rows.append((meta["model"], meta["grouping"], float(acc)))
            except Exception:
                continue
    return rows

def aggregate(rows):
    bucket = defaultdict(list)  # (model, grouping) -> [acc...]
    for model, grouping, acc in rows:
        if is_valid(acc):
            bucket[(model, grouping)].append(acc)
    per_model = defaultdict(dict)
    for (model, grouping), vals in bucket.items():
        per_model[model][grouping] = float(np.mean(vals))
    model_order = [
        MODELS["xgboost"],
        MODELS["svm"],
        MODELS["randomforest"],
        MODELS["mlp"],
        MODELS["logisticregression"],
    ]
    model_order = [m for m in model_order if m in per_model] or sorted(per_model.keys())
    return per_model, model_order

def save_results(per_model, model_order, out_prefix="heart_"):
    # Wide JSON (per model -> per grouping -> mean accuracy)
    with open(f"{out_prefix}accuracy_by_model_grouping.json", "w") as f:
        json.dump(per_model, f, indent=2)

    # Long CSV for convenient plotting elsewhere
    long_rows = []
    for model in model_order:
        for g in GROUPING_ORDER:
            v = per_model.get(model, {}).get(g)
            if v is not None:
                long_rows.append({"Model": model, "Grouping": g, "Precision": v})
    pd.DataFrame(long_rows).to_csv(f"{out_prefix}precision_long.csv", index=False)

    # Overlay arrays (ordered like your PNG’s y-axis): dict[grouping] -> list of accuracies (%)
    overlay = {}
    for g in GROUPING_ORDER:
        overlay[g] = [per_model.get(model, {}).get(g, None) for model in model_order]
        overlay[g] = [None if v is None else round(v*100, 3) for v in overlay[g]]
    meta = {"model_order": model_order, "grouping_order": GROUPING_ORDER, "values_pct": overlay}
    with open(f"{out_prefix}overlay_arrays.json", "w") as f:
        json.dump(meta, f, indent=2)

# ----- Benchmark CSV loader (Model, Min, Mean, Max in PERCENT) -----
def load_benchmark_csv(csv_path):
    df = pd.read_csv(csv_path)
    cols = {c.lower(): c for c in df.columns}
    # Try to find columns flexibly
    def pick(*variants):
        for v in variants:
            for k, orig in cols.items():
                if v in k:
                    return orig
        return None
    c_model = pick("model")
    c_min   = pick("min")
    c_mean  = pick("mean")
    c_max   = pick("max")
    if not all([c_model, c_min, c_mean, c_max]):
        raise ValueError("Benchmark CSV must contain columns for Model, Min, Mean, Max (in %).")
    bench = {}
    for _, r in df.iterrows():
        bench[str(r[c_model])] = {
            "min": float(r[c_min]),
            "mean": float(r[c_mean]),
            "max": float(r[c_max]),
        }
    return bench

# ----- Overlay plot: benchmark min/mean/max + grouping markers -----
def plot_overlay(per_model, model_order, benchmark_stats, out_path):
    fig, ax = plt.subplots(figsize=(12, 6))

    # benchmark whisker plus mean in a neutral color
    whisker_color = "#1f77b4"
    for i, model in enumerate(model_order):
        stats = benchmark_stats.get(model)
        if not stats:
            continue
        xmin, xmean, xmax = stats["min"], stats["mean"], stats["max"]
        ax.hlines(i, xmin, xmax, color=whisker_color, linewidth=2, alpha=0.9, zorder=1)
        ax.plot([xmin, xmax], [i, i], marker="P", linestyle="None", color=whisker_color, markersize=6, zorder=2)
        ax.plot(xmean, i, marker="o", color=whisker_color, markersize=6, zorder=2)

    # overlay grouping markers with fixed colors and jitter
    for i, model in enumerate(model_order):
        if model not in per_model:
            continue
        for grouping, acc in per_model[model].items():
            if grouping not in MARKERS or acc is None:
                continue
            y = i + JITTER[grouping]
            ax.plot(
                acc * 100.0, y,
                marker=MARKERS[grouping],
                linestyle="None",
                markersize=9,
                markerfacecolor="white",
                markeredgecolor=COLORS[grouping],
                markeredgewidth=2,
                alpha=0.95,
                zorder=3,
            )

    # legend with proxy handles so styles match exactly
    legend_handles = [
        Line2D(
            [0], [0],
            marker=MARKERS[g],
            linestyle="None",
            markersize=9,
            markerfacecolor="white",
            markeredgecolor=COLORS[g],
            markeredgewidth=2,
            label=g
        )
        for g in MARKERS.keys()
    ]
    ax.legend(legend_handles, list(MARKERS.keys()), title="Grouping", loc="lower left", ncol=2, frameon=True)

    ax.set_yticks(range(len(model_order)))
    ax.set_yticklabels(model_order)
    ax.set_xlabel("Precision (%)")
    ax.set_title("Baseline Model Performance - Breast Cancer (Precision)\nBenchmark min-mean-max with grouping overlays")

    # x limits padded from benchmark range
    all_vals = [v for m in benchmark_stats.values() for v in (m["min"], m["max"])]
    if all_vals:
        pad = max(1, 0.02 * (max(all_vals) - min(all_vals)))
        ax.set_xlim(min(all_vals) - pad, max(all_vals) + pad)

    ax.grid(True, axis="x", linestyle="--", alpha=0.35, zorder=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved overlay: {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=".", help="Root folder with JSON files")
    p.add_argument("--pattern", default="performance_metrics_*.json", help="Filename glob")
    p.add_argument("--out-prefix", default="heart_", help="Prefix for saved outputs")
    # optional overlay args
    p.add_argument("--benchmark-csv", default=None, help="CSV with columns Model, Min, Mean, Max (values in %)")
    p.add_argument("--plot-out", default=None, help="If set, write overlay plot to this file")
    args = p.parse_args()

    rows = collect_heart_accuracy(args.root, args.pattern)
    if not rows:
        print("No Heart Disease accuracy records found. Check --root/--pattern.")
        return
    per_model, model_order = aggregate(rows)
    save_results(per_model, model_order, args.out_prefix)

    # Optional overlay plot with benchmark stats
    if args.benchmark_csv and args.plot_out:
        benchmark = load_benchmark_csv(args.benchmark_csv)
        plot_overlay(per_model, model_order, benchmark, args.plot_out)
    else:
        print("Tip: to create the overlay, pass --benchmark-csv <file.csv> and --plot-out <image.png>")

if __name__ == "__main__":
    main()
