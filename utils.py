import os
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt


def ensure_dir(path: str):
    """Create directory if path is not empty and does not exist."""
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def save_plot(fig, path: str):
    """Save a matplotlib figure to disk safely."""
    ensure_dir(os.path.dirname(path))
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    print(f"[Saved] Plot: {path}")


def save_metrics(metrics: dict, path: str, run_id=None):
    """Save metrics as JSON."""
    ensure_dir(os.path.dirname(path))
    save_path = path if run_id is None else f"{os.path.splitext(path)[0]}_run{run_id}.json"
    with open(save_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"[Saved] Metrics: {save_path}")


def save_dataframe(df: pd.DataFrame, path: str):
    """Save a DataFrame to CSV."""
    ensure_dir(os.path.dirname(path))
    df.to_csv(path, index=False)
    print(f"[Saved] DataFrame: {path}")


def save_model(model, path: str, run_id=None):
    """Save a scikit-learn model using joblib."""
    ensure_dir(os.path.dirname(path))
    save_path = path if run_id is None else f"{os.path.splitext(path)[0]}_run{run_id}.joblib"
    joblib.dump(model, save_path)
    print(f"[Saved] Model: {save_path}")
