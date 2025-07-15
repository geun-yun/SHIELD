import numpy as np
import pandas as pd
import seaborn as sns
import shap
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline


def get_shap_explainer(model, X_train):
    """
    Returns a SHAP explainer that handles Pipelines safely.
    """
    estimator = model

    if isinstance(model, Pipeline):
        estimator = model.steps[-1][1]  # Get final estimator

    if isinstance(estimator, (XGBClassifier, RandomForestClassifier)):
        # Tree models work with TreeExplainer directly, but wrap Pipeline with callable
        if isinstance(model, Pipeline):
            f = lambda X: model.predict_proba(X)
            return shap.Explainer(f, X_train)  # Or TreeExplainer with .model
        else:
            return shap.TreeExplainer(estimator)
    elif isinstance(estimator, MLPClassifier):
        # Neural nets → Kernel or DeepExplainer if using TF/PyTorch
        f = lambda X: model.predict_proba(X) if hasattr(model, "predict_proba") else model.predict(X)
        return shap.KernelExplainer(f, X_train)
    elif isinstance(estimator, (LogisticRegression, SVC)):
        f = lambda X: model.predict_proba(X) if hasattr(model, "predict_proba") else model.predict(X)
        return shap.KernelExplainer(f, X_train)
    else:
        # Fallback: always wrap with callable for Pipeline
        f = lambda X: model.predict_proba(X) if hasattr(model, "predict_proba") else model.predict(X)
        return shap.KernelExplainer(f, X_train)


def decompose_shap_to_features(shap_values, latent_cols, feature_mappings):
    # Defensive: must be 2D only
    if shap_values.ndim != 2:
        raise ValueError(f"decompose_shap_to_features: SHAP must be 2D, got {shap_values.shape}")
    latent_df = pd.DataFrame(shap_values, columns=latent_cols)
    all_feats = []
    for gid, mapping_df in feature_mappings.items():
        group_latent_cols = mapping_df.index.tolist()
        latent_vals = latent_df[group_latent_cols].values
        W = mapping_df.to_numpy()
        W_abs = np.abs(W)
        norm = W_abs.sum(axis=0)
        norm[norm == 0] = 1.0
        weights = W_abs / norm
        shap_feats = latent_vals @ weights
        df = pd.DataFrame(shap_feats, columns=mapping_df.columns, index=latent_df.index)
        all_feats.append(df)
    return pd.concat(all_feats, axis=1)


def plot_shap_raw(vals, X_test, feature_names, model_name, mode, z_scores, max_display):
    if vals.ndim == 3:
        n_classes = vals.shape[2]
        if mode == "each_label":
            for c in range(n_classes):
                shap.summary_plot(vals[:, :, c], X_test, feature_names=feature_names,
                                  plot_type="dot", max_display=max_display, show=False)
                plt.title(f"{model_name} – SHAP (raw) for class {c}")
                plt.tight_layout()
                plt.show()
        elif mode == "aggregated":
            agg_vals = (np.tensordot(vals, z_scores, axes=([2], [0]))
                        if z_scores is not None else np.sum(np.abs(vals), axis=2))
            shap.summary_plot(agg_vals, X_test, feature_names=feature_names,
                              plot_type="dot", max_display=max_display, show=False)
            plt.title(f"{model_name} – SHAP (raw, aggregated)")
            plt.tight_layout()
            plt.show()
    else:
        shap.summary_plot(vals, X_test, feature_names=feature_names,
                          plot_type="dot", max_display=max_display, show=False)
        plt.title(f"{model_name} – SHAP (raw)")
        plt.tight_layout()
        plt.show()


def plot_shap_grouped(raw, X_test, latent_cols, feature_mappings, model_name, mode, z_scores, max_display):
    if raw.ndim == 3:
        n_classes = raw.shape[2]
        if mode == "each_label":
            for c in range(n_classes):
                slice_raw = raw[:, :, c]  # (n_samples, n_latent)
                shap_df = decompose_shap_to_features(slice_raw, latent_cols, feature_mappings)
                shap.summary_plot(shap_df.values, X_test[shap_df.columns],
                                  plot_type="dot", max_display=max_display, show=False)
                plt.title(f"{model_name} – SHAP decomposed for class {c}")
                plt.tight_layout()
                plt.show()
        elif mode == "aggregated":
            agg = (np.tensordot(raw, z_scores, axes=([2], [0]))
                   if z_scores is not None else np.sum(np.abs(raw), axis=2))
            shap_df = decompose_shap_to_features(agg, latent_cols, feature_mappings)
            shap.summary_plot(shap_df.values, X_test[shap_df.columns],
                              plot_type="dot", max_display=max_display, show=False)
            plt.title(f"{model_name} – SHAP decomposed aggregated")
            plt.tight_layout()
            plt.show()
    else:
        shap_df = decompose_shap_to_features(raw, latent_cols, feature_mappings)
        shap.summary_plot(shap_df.values, X_test[shap_df.columns],
                          plot_type="dot", max_display=max_display, show=False)
        plt.title(f"{model_name} – SHAP decomposed")
        plt.tight_layout()
        plt.show()


def prepare_latent(X_test, encoders, feature_mappings):
    latent_cols = []
    X_test_latent = pd.DataFrame(index=X_test.index)
    for gid, (scaler, ae, _) in encoders.items():
        feats = feature_mappings[gid].columns.tolist()
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


def compute_shap_range_stats(shap_vals, feature_names, lower_pct=5, upper_pct=95,  plot_box=False, label=""):
    """ Compute robust range stats for each feature, excluding outliers. """
    if shap_vals.ndim != 2:
        raise ValueError(f"Expected 2D SHAP array, got {shap_vals.shape}")

    lower_bounds = np.percentile(shap_vals, lower_pct, axis=0)
    upper_bounds = np.percentile(shap_vals, upper_pct, axis=0)
    feature_ranges = upper_bounds - lower_bounds

    stats = {
        'max': np.max(feature_ranges),
        'min': np.min(feature_ranges),
        'mean': np.mean(feature_ranges),
        'std': np.std(feature_ranges)
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
        plt.figure(figsize=(6, 4))
        sns.boxplot(x=feature_ranges, color='skyblue')
        sns.stripplot(x=feature_ranges, color='black', alpha=0.3, jitter=True)
        plt.title(f"SHAP Range Boxplot {label}")
        plt.xlabel(f"Feature SHAP Range ({lower_pct}-{upper_pct} pctile)")
        plt.tight_layout()
        plt.show()

    return stats


def evaluate_shap_features(
    X_train, X_test, trained_models, encoders, feature_mappings,
    max_display=10, mode="each_label", z_scores=None
):
    """Robust SHAP + range stats, matching background dims correctly."""
    shap_results = {}

    # RAW FEATURE CASE
    if not encoders:
        for name, model in trained_models.items():
            print(f"\nSHAP (raw) → {name}")
            bg = shap.kmeans(X_train, min(200, X_train.shape[0]))
            # Inside RAW FEATURE CASE in evaluate_shap_features:
            if isinstance(model, Pipeline):
                scaler = model.named_steps['scaler']
                estimator = model.named_steps['model']
                cols = X_train.columns

                def model_predict(X):
                    if isinstance(X, np.ndarray):
                        X = pd.DataFrame(X, columns=cols)
                    X_scaled = scaler.transform(X)
                    return estimator.predict_proba(X_scaled)

                explainer = shap.KernelExplainer(model_predict, bg)
            else:
                explainer = shap.KernelExplainer(model.predict_proba, bg)

            shap_out = explainer(X_test)
            vals = shap_out.values if hasattr(shap_out, "values") else shap_out

            if vals.ndim == 3 and mode == "each_label":
                for c in range(vals.shape[2]):
                    class_vals = vals[:, :, c]
                    compute_shap_range_stats(class_vals, X_test.columns.tolist())
                    shap.summary_plot(
                        class_vals, X_test, feature_names=X_test.columns,
                        plot_type="dot", max_display=max_display, show=False
                    )
                    plt.title(f"{name} – SHAP (raw) class {c}")
                    plt.tight_layout()
                    plt.show()
            elif vals.ndim == 3 and mode == "aggregated":
                agg = (np.tensordot(vals, z_scores, axes=([2], [0]))
                       if z_scores is not None else np.sum(np.abs(vals), axis=2))
                compute_shap_range_stats(agg, X_test.columns.tolist(), plot_box=True, label=f"{name} RAW AGG")
                shap.summary_plot(
                    agg, X_test, feature_names=X_test.columns,
                    plot_type="dot", max_display=max_display, show=False
                )
                plt.title(f"{name} – SHAP (raw aggregated)")
                plt.tight_layout()
                plt.show()
            else:
                compute_shap_range_stats(vals, X_test.columns.tolist())
                shap.summary_plot(
                    vals, X_test, feature_names=X_test.columns,
                    plot_type="dot", max_display=max_display, show=False
                )
                plt.title(f"{name} – SHAP (raw)")
                plt.tight_layout()
                plt.show()
            shap_results[name] = {"raw": vals}
        return shap_results
    
    # GROUPED LATENT FEATURE CASE
    X_train_latent, _ = prepare_latent(X_train, encoders, feature_mappings)
    X_test_latent, latent_cols = prepare_latent(X_test, encoders, feature_mappings)

    for name, model in trained_models.items():
        print(f"\nSHAP (grouped) → {name}")
        if isinstance(model, (LogisticRegression, SVC, MLPClassifier)):
            bg = shap.kmeans(X_train_latent, min(50, X_train_latent.shape[0]))
            explainer = shap.KernelExplainer(model.predict_proba, bg)
        else:
            explainer = get_shap_explainer(model, X_train_latent)

        shap_out = explainer(X_test_latent)
        raw = shap_out.values if hasattr(shap_out, "values") else shap_out

        if raw.ndim == 3 and mode == "each_label":
            for c in range(raw.shape[2]):
                raw_c = raw[:, :, c]
                # Decompose latent SHAP to feature SHAP
                shap_feat_df = decompose_shap_to_features(raw_c, latent_cols, feature_mappings)
                # Compute stats on the decomposed features
                compute_shap_range_stats(shap_feat_df.values, shap_feat_df.columns.tolist())
                shap.summary_plot(
                    shap_feat_df.values,
                    X_test[shap_feat_df.columns],
                    plot_type="dot",
                    max_display=max_display,
                    show=False
                )
                plt.title(f"{name} – SHAP grouped class {c}")
                plt.tight_layout()
                plt.show()

        elif raw.ndim == 3 and mode == "aggregated":
            agg = (np.tensordot(raw, z_scores, axes=([2], [0]))
                if z_scores is not None else np.sum(np.abs(raw), axis=2))
            shap_feat_df = decompose_shap_to_features(agg, latent_cols, feature_mappings)
            compute_shap_range_stats(shap_feat_df.values, shap_feat_df.columns.tolist(), plot_box=True, label=f"{name} GROUPED AGG")
            shap.summary_plot(
                shap_feat_df.values,
                X_test[shap_feat_df.columns],
                plot_type="dot",
                max_display=max_display,
                show=False
            )
            plt.title(f"{name} – SHAP grouped aggregated")
            plt.tight_layout()
            plt.show()

        else:
            shap_feat_df = decompose_shap_to_features(raw, latent_cols, feature_mappings)
            compute_shap_range_stats(shap_feat_df.values, shap_feat_df.columns.tolist())
            shap.summary_plot(
                shap_feat_df.values,
                X_test[shap_feat_df.columns],
                plot_type="dot",
                max_display=max_display,
                show=False
            )
            plt.title(f"{name} – SHAP grouped")
            plt.tight_layout()
            plt.show()

        shap_results[name] = {"raw": raw, "decomposed": shap_feat_df}

    return shap_results