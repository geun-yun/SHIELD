import os
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.neighbors import KernelDensity, KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from utils import save_plot, save_metrics


def _compute_group_rates(y_true, y_pred):
    labels = np.unique(np.concatenate([y_true, y_pred]))
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    if cm.shape != (2, 2):
        padded_cm = np.zeros((2, 2), dtype=int)
        for i, label in enumerate(labels):
            for j, pred_label in enumerate(labels):
                padded_cm[label][pred_label] = cm[i][j]
        cm = padded_cm

    tn, fp, fn, tp = cm.ravel()
    total = tp + tn + fp + fn + 1e-6

    return {
        "TPR": tp / (tp + fn + 1e-6),
        "FPR": fp / (fp + tn + 1e-6),
        "Precision": tp / (tp + fp + 1e-6),
        "ErrorRate": (fp + fn) / total,
    }


def disparate_impact(y_pred, sensitive, privileged_value):
    privileged_mask = (sensitive == privileged_value)
    unprivileged_mask = ~privileged_mask

    p_priv = np.mean(y_pred[privileged_mask] == 1)
    p_unpriv = np.mean(y_pred[unprivileged_mask] == 1)

    if p_priv == 0:
        return np.nan
    return p_unpriv / p_priv


def equal_opportunity(rates_priv, rates_unpriv):
    return abs(rates_priv["TPR"] - rates_unpriv["TPR"])


def equalized_odds(rates_priv, rates_unpriv):
    return max(
        abs(rates_priv["TPR"] - rates_unpriv["TPR"]),
        abs(rates_priv["FPR"] - rates_unpriv["FPR"])
    )


def predictive_parity(rates_priv, rates_unpriv):
    return abs(rates_priv["Precision"] - rates_unpriv["Precision"])


def n_sigma(rates_priv, rates_unpriv, err_arr_priv, err_arr_unpriv):
    err_priv = rates_priv["ErrorRate"]
    err_unpriv = rates_unpriv["ErrorRate"]

    std_priv = np.std(err_arr_priv, ddof=1)
    std_unpriv = np.std(err_arr_unpriv, ddof=1)
    pooled_sigma = np.sqrt((std_priv**2 + std_unpriv**2) / 2)

    return abs(err_priv - err_unpriv) / pooled_sigma if pooled_sigma > 0 else np.nan


def explanation_bias(shap_values, sensitive, privileged_value, feature_names):
    privileged_mask = (sensitive == privileged_value)
    unprivileged_mask = ~privileged_mask

    bias_per_feature = {}
    for i, feat in enumerate(feature_names):
        shap_priv = shap_values[privileged_mask, i]
        shap_unpriv = shap_values[unprivileged_mask, i]
        bias_per_feature[feat] = abs(np.mean(shap_priv) - np.mean(shap_unpriv))
    return bias_per_feature


def _save_plot_with_runid(fig, path: str, run_id=None):
    """Embed run_id into filename if given, then delegate to save_plot."""
    if run_id is not None:
        base, ext = os.path.splitext(path)
        path = f"{base}_run{run_id}{ext}"
    save_plot(fig, path)


def plot_instance_level_quadrant(y_pred_proba, shap_values, sensitive, privileged_value,
                                 protected_idx, dataset_name, alpha=0.05, class_idx=1, run_id=None, config_name=None):
    p_pos = y_pred_proba[:, class_idx]

    if shap_values.ndim == 3:
        shap_class = shap_values[:, :, class_idx]
    elif shap_values.ndim == 2:
        shap_class = shap_values
    else:
        raise ValueError(f"Unexpected shap_values shape: {shap_values.shape}")

    priv_mask = (sensitive == privileged_value)
    mu_priv = np.mean(p_pos[priv_mask])
    mu_unpriv = np.mean(p_pos[~priv_mask])

    base_rates = np.where(priv_mask, mu_priv, mu_unpriv)
    diff = p_pos - base_rates
    mask = np.abs(diff) > alpha

    x_vals = shap_class[mask, protected_idx]
    y_vals = diff[mask]

    sensitive_masked = sensitive[mask]

    print(f"x_vals shape: {x_vals.shape}, y_vals shape: {y_vals.shape}, sensitive_masked: {sensitive_masked.shape}")

    if len(x_vals) != len(y_vals):
        raise ValueError(f"x and y must be same size! Got x: {x_vals.shape}, y: {y_vals.shape}")

    distances = np.sqrt(x_vals**2 + y_vals**2)
    avg_distance = np.mean(distances)
    print(f"Average distance from origin (bias magnitude): {avg_distance:.4f}")

    plt.figure(figsize=(8, 8))

    colors = np.where(sensitive_masked == privileged_value, 'blue', 'red')
    plt.scatter(x_vals, y_vals, c=colors, alpha=0.5)

    plt.axhline(0, color='grey', linestyle='--')
    plt.axvline(0, color='grey', linestyle='--')

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='Privileged',
               markerfacecolor='blue', markersize=8),
        Line2D([0], [0], marker='o', color='w', label='Unprivileged',
               markerfacecolor='red', markersize=8)
    ]
    plt.legend(handles=legend_elements, title='Sensitive Attribute')

    plt.xlabel("SHAP for protected attribute")
    plt.ylabel("Prediction minus Base Rate (R - μ_t)")
    plt.title(f"Instance-Level Bias Quadrant for {dataset_name}")
    plt.text(
        0.05, 0.95,
        f"Avg Distance: {avg_distance:.4f}",
        ha='left', va='top',
        transform=plt.gca().transAxes,
        fontsize=10, bbox=dict(facecolor='white', alpha=0.5)
    )

    fname = f"bias_quadrant_{config_name}.png"
    fig = plt.gcf()
    _save_plot_with_runid(fig, fname, run_id=run_id)
    plt.close(fig)

    return avg_distance


def estimate_observation_bias(X_empirical, X_removal):
    kde_data = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(X_empirical)
    kde_removal = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(X_removal)
    sample_points = X_empirical[np.random.choice(len(X_empirical), size=300)]
    log_p_data = kde_data.score_samples(sample_points)
    log_p_removal = kde_removal.score_samples(sample_points)
    tv_dist = 0.5 * np.mean(np.abs(np.exp(log_p_data) - np.exp(log_p_removal)))
    return tv_dist


def estimate_structural_bias(X_empirical, X_removal):
    X_all = np.vstack([X_empirical, X_removal])
    y_all = np.concatenate([np.zeros(len(X_empirical)), np.ones(len(X_removal))])
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.3, random_state=42)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred = knn.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred)
    return auc


def fairness_metrics(y_true, y_pred, sensitive, privileged_value,
                     shap_values=None, feature_names=None,
                     y_pred_proba=None, protected_attr=None,
                     alpha=0.05,
                     X_empirical=None, X_removal=None, dataset_name=None, run_id=None, config_name=None):
    results = {}

    privileged_mask = (sensitive == privileged_value)
    unprivileged_mask = ~privileged_mask

    rates_priv = _compute_group_rates(y_true[privileged_mask], y_pred[privileged_mask])
    rates_unpriv = _compute_group_rates(y_true[unprivileged_mask], y_pred[unprivileged_mask])

    results["Disparate Impact"] = disparate_impact(y_pred, sensitive, privileged_value)
    results["Equal Opportunity"] = equal_opportunity(rates_priv, rates_unpriv)
    results["Equalized Odds"] = equalized_odds(rates_priv, rates_unpriv)
    results["Predictive Parity"] = predictive_parity(rates_priv, rates_unpriv)

    err_arr_priv = np.abs(y_true[privileged_mask] != y_pred[privileged_mask]).astype(float)
    err_arr_unpriv = np.abs(y_true[unprivileged_mask] != y_pred[unprivileged_mask]).astype(float)
    results["N-Sigma (ErrorRate)"] = n_sigma(rates_priv, rates_unpriv, err_arr_priv, err_arr_unpriv)

    if shap_values is not None and y_pred_proba is not None and protected_attr is not None:
        if protected_attr in feature_names:
            protected_idx = feature_names.index(protected_attr)
            avg_dist = plot_instance_level_quadrant(
                y_pred_proba, shap_values, sensitive, privileged_value,
                protected_idx, dataset_name, alpha=alpha, class_idx=1, run_id=run_id, config_name=config_name
            )
            results["Average distance from origin"] = avg_dist

    if X_empirical is not None and X_removal is not None:
        results["Observation Bias (TV)"] = estimate_observation_bias(X_empirical, X_removal)
        results["Structural Bias (kNN AUC)"] = estimate_structural_bias(X_empirical, X_removal)

    if dataset_name is not None:
        fname = f"fairness_metrics_{config_name}.json"
        save_metrics(results, fname, run_id=run_id)

    return results
