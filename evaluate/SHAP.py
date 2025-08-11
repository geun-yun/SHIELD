import os
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from shap.maskers import Independent
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from utils import save_plot


def _ensure_predict_proba_callable(model):
    if hasattr(model, "predict_proba"):
        return lambda X: model.predict_proba(X)
    elif hasattr(model, "decision_function"):
        def warn_predict(X):
            print("[Warning] Using decision_function fallback instead of predict_proba; SHAP semantics may differ.")
            return model.decision_function(X)
        return warn_predict
    else:
        raise AttributeError(f"Model {model} has neither predict_proba nor decision_function.")


def get_shap_explainer(model, X_train, background_size=50):
    centers = shap.kmeans(X_train, min(background_size, X_train.shape[0])).data
    masker = Independent(centers)

    estimator = model
    if isinstance(model, Pipeline):
        estimator = model.steps[-1][1]

    if isinstance(estimator, (XGBClassifier, RandomForestClassifier)):
        if not isinstance(model, Pipeline):
            # Native tree path: pass data directly (fast, exact-ish)
            return shap.TreeExplainer(estimator, data=X_train)  # ok
        
        f = _ensure_predict_proba_callable(model)
        return shap.Explainer(f, masker)
    elif isinstance(estimator, LogisticRegression):
        if isinstance(model, Pipeline):
            f = _ensure_predict_proba_callable(model)
            return shap.Explainer(f, masker)
        else:
            return shap.LinearExplainer(estimator, X_train, feature_perturbation="interventional")
    elif isinstance(estimator, (SVC, MLPClassifier)):
        f = _ensure_predict_proba_callable(model)
        return shap.Explainer(f, masker) 
    else:
        f = _ensure_predict_proba_callable(model)
        return shap.Explainer(f, masker)



def decompose_shap_to_features(shap_values, latent_cols, feature_mappings):
    if shap_values.ndim != 2:
        raise ValueError(f"decompose_shap_to_features: SHAP must be 2D, got {shap_values.shape}")
    latent_df = pd.DataFrame(shap_values, columns=latent_cols)
    all_feats = []
    for gid, mapping_df in feature_mappings.items():
        group_latent_cols = mapping_df.index.tolist()
        missing = [c for c in group_latent_cols if c not in latent_df.columns]
        if missing:
            raise KeyError(f"Latent columns {missing} referenced in mapping for group {gid} not found in SHAP values.")
        latent_vals = latent_df[group_latent_cols].values  # (n_samples, latent_dim_for_group)
        W = mapping_df.to_numpy()  # (latent_dim_for_group, original_features)
        W_abs = np.abs(W)
        norm = W_abs.sum(axis=0)
        norm[norm == 0] = 1.0
        weights = W_abs / norm  # normalized latent->original
        shap_feats = latent_vals @ weights  # (n_samples, original_features)
        df = pd.DataFrame(shap_feats, columns=mapping_df.columns, index=latent_df.index)
        all_feats.append(df)
    return pd.concat(all_feats, axis=1)


def _save_summary_plot(make_plot_fn, title, filename, run_id=None):
    make_plot_fn()
    fig = plt.gcf()
    plt.title(title)
    plt.tight_layout()
    if run_id is not None:
        base, ext = os.path.splitext(filename)
        filename = f"{base}_run{run_id}{ext}"
    save_plot(fig, filename)
    plt.close(fig)


def compute_shap_range_stats(shap_vals, feature_names, lower_pct=5, upper_pct=95,
                             plot_box=True, label="", run_id=None):
    if shap_vals.ndim != 2:
        raise ValueError(f"Expected 2D SHAP array, got {shap_vals.shape}")

    lower_bounds = np.percentile(shap_vals, lower_pct, axis=0)
    upper_bounds = np.percentile(shap_vals, upper_pct, axis=0)
    feature_ranges = upper_bounds - lower_bounds

    stats = {
        'max': float(np.max(feature_ranges)),
        'min': float(np.min(feature_ranges)),
        'mean': float(np.mean(feature_ranges)),
        'std': float(np.std(feature_ranges))
    }

    print(f"SHAP value range stats ({lower_pct}-{upper_pct} percentile):")
    print(f"Max range: {stats['max']:.4f}, Min range: {stats['min']:.4f}, "
          f"Mean range: {stats['mean']:.4f}, Std: {stats['std']:.4f}")

    ranges_df = pd.DataFrame({
        'Feature': feature_names,
        f'Range_{lower_pct}-{upper_pct}pct': feature_ranges
    }).sort_values(by=f'Range_{lower_pct}-{upper_pct}pct', ascending=False)

    print(ranges_df.head(10))

    if plot_box:
        def mk():
            sns.boxplot(x=feature_ranges)
            sns.stripplot(x=feature_ranges, color='black', alpha=0.3, jitter=True)
            plt.xlabel(f"Feature SHAP Range ({lower_pct}-{upper_pct} pctile)")

        fname = f"shap_range_boxplot_{label}.png"
        _save_summary_plot(mk, f"SHAP Range Boxplot {label}", fname, run_id)

    return stats


def plot_shap_raw(vals, X_test, feature_names, model_name, mode, z_scores, max_display, run_id, config_name=None):
    if vals.ndim == 3:
        n_classes = vals.shape[2]
        if mode == "each_label":
            for c in range(n_classes):
                def mk():
                    shap.summary_plot(vals[:, :, c], X_test, feature_names=feature_names,
                                      plot_type="dot", max_display=max_display, show=False)
                fname = f"shap_raw_{config_name}_class{c}.png"
                _save_summary_plot(mk, f"{config_name} – SHAP (raw) class {c}", fname, run_id)
        elif mode == "aggregated":
            if z_scores is not None:
                agg_vals = np.tensordot(vals, z_scores, axes=([2], [0]))
            else:
                agg_vals = np.sum(np.abs(vals), axis=2)
            def mk():
                shap.summary_plot(agg_vals, X_test, feature_names=feature_names,
                                  plot_type="dot", max_display=max_display, show=False)
            fname = f"shap_raw_{config_name}_aggregated.png"
            _save_summary_plot(mk, f"{config_name} – SHAP (raw aggregated)", fname, run_id)
    else:
        def mk():
            shap.summary_plot(vals, X_test, feature_names=feature_names,
                              plot_type="dot", max_display=max_display, show=False)
        fname = f"shap_raw_{config_name}.png"
        _save_summary_plot(mk, f"{config_name} – SHAP (raw)", fname, run_id)


def plot_shap_grouped(raw, X_test, latent_cols, feature_mappings, model_name, mode, z_scores, max_display, run_id, config_name=ModuleNotFoundError):
    if raw.ndim == 3:
        n_classes = raw.shape[2]
        if mode == "each_label":
            for c in range(n_classes):
                slice_raw = raw[:, :, c]
                shap_df = decompose_shap_to_features(slice_raw, latent_cols, feature_mappings)
                def mk():
                    shap.summary_plot(shap_df.values, X_test[shap_df.columns],
                                      plot_type="dot", max_display=max_display, show=False)
                fname = f"shap_grouped_{config_name}_class{c}.png"
                _save_summary_plot(mk, f"{config_name} – SHAP decomposed class {c}", fname, run_id)
        elif mode == "aggregated":
            if z_scores is not None:
                agg = np.tensordot(raw, z_scores, axes=([2], [0]))
            else:
                agg = np.sum(np.abs(raw), axis=2)
            shap_df = decompose_shap_to_features(agg, latent_cols, feature_mappings)
            def mk():
                shap.summary_plot(shap_df.values, X_test[shap_df.columns],
                                  plot_type="dot", max_display=max_display, show=False)
            fname = f"shap_grouped_{config_name}_aggregated.png"
            _save_summary_plot(mk, f"{config_name} – SHAP decomposed aggregated", fname, run_id)
    else:
        shap_df = decompose_shap_to_features(raw, latent_cols, feature_mappings)
        def mk():
            shap.summary_plot(shap_df.values, X_test[shap_df.columns],
                              plot_type="dot", max_display=max_display, show=False)
        fname = f"shap_grouped_{config_name}.png"
        _save_summary_plot(mk, f"{config_name} – SHAP decomposed", fname, run_id)


def prepare_latent(X_test, encoders, feature_mappings):
    latent_cols = []
    X_test_latent = pd.DataFrame(index=X_test.index)
    for gid, (scaler, ae, _) in encoders.items():
        feats = feature_mappings[gid].columns.tolist()
        if not set(feats).issubset(X_test.columns):
            missing = set(feats) - set(X_test.columns)
            raise KeyError(f"Missing features in X_test for group {gid}: {missing}")
        Xs = scaler.transform(X_test[feats].values)
        Z = Xs @ ae.coefs_[0] + ae.intercepts_[0]
        act = ae.activation
        if act == 'relu':
            codes = np.maximum(0, Z)
        elif act == 'identity':
            codes = Z
        elif act == 'logistic':
            codes = 1 / (1 + np.exp(-Z))
        elif act == 'tanh':
            codes = np.tanh(Z)
        else:
            raise ValueError(f"Unsupported activation: {act}")
        if ae.coefs_[1].shape[0] == 1:
            col = f'group_{gid}'
            X_test_latent[col] = codes.ravel()
            latent_cols.append(col)
        else:
            for d in range(codes.shape[1]):
                col = f'group_{gid}_dim{d}'
                X_test_latent[col] = codes[:, d]
                latent_cols.append(col)
    return X_test_latent, latent_cols


def evaluate_shap_features(
    X_train, X_test, trained_models, encoders, feature_mappings,
    max_display=10, mode="each_label", z_scores=None, run_id=None,
    background_size=50, config_name=None
):
    """
    Returns a dictionary of SHAP outputs. Always decomposes grouped features before range stats.
    """
    shap_results = {}

    # RAW FEATURE CASE
    if not encoders:
        for name, model in trained_models.items():
            print(f"\nSHAP (raw) → {name}")
            explainer = get_shap_explainer(model, X_train, background_size=background_size)
            shap_out = explainer(X_test)
            vals = shap_out.values if hasattr(shap_out, "values") else shap_out

            if vals.ndim == 3 and mode == "each_label":
                for c in range(vals.shape[2]):
                    compute_shap_range_stats(
                        vals[:, :, c], X_test.columns.tolist(), label=f"{config_name}_grouped_class{c}", run_id=run_id
                    )
            elif vals.ndim == 2 or mode == "aggregated":
                compute_shap_range_stats(vals, X_test.columns.tolist(), label=f"{config_name}_aggregated_agg", run_id=run_id)

            plot_shap_raw(vals, X_test, X_test.columns.tolist(), name, mode, z_scores, max_display, run_id, config_name=config_name)
            shap_results[name] = {"shap": vals, "mode": "raw", "raw": vals}


        return shap_results

    # GROUPED LATENT FEATURE CASE
    X_train_latent, _ = prepare_latent(X_train, encoders, feature_mappings)
    X_test_latent, latent_cols = prepare_latent(X_test, encoders, feature_mappings)

    for name, model in trained_models.items():
        print(f"\nSHAP (grouped) → {name}")
        explainer = get_shap_explainer(model, X_train_latent, background_size=background_size)
        shap_out = explainer(X_test_latent)
        raw = shap_out.values if hasattr(shap_out, "values") else shap_out

        if raw.ndim == 3 and mode == "each_label":
            for c in range(raw.shape[2]):
                shap_df = decompose_shap_to_features(raw[:, :, c], latent_cols, feature_mappings)
                compute_shap_range_stats(
                    shap_df.values, shap_df.columns.tolist(), label=f"{config_name}_grouped_class{c}", run_id=run_id
                )
        elif raw.ndim == 2 or mode == "aggregated":
            shap_df = decompose_shap_to_features(raw, latent_cols, feature_mappings)
            compute_shap_range_stats(shap_df.values, shap_df.columns.tolist(), label=f"{config_name}_agg", run_id=run_id)

        plot_shap_grouped(raw, X_test, latent_cols, feature_mappings, name, mode, z_scores, max_display, run_id, config_name=config_name)
        shap_results[name] = {"shap": raw, "mode": "grouped", "raw": raw}


    return shap_results
