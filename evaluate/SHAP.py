import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
import re
import shap

def get_shap_explainer(model, X_train):
    """
    Selects the appropriate SHAP explainer based on model type.
    """
    if isinstance(model, (XGBClassifier, RandomForestClassifier)):
        return shap.TreeExplainer(model)
    elif isinstance(model, MLPClassifier):
        return shap.DeepExplainer(model, X_train)
    elif isinstance(model, (LogisticRegression, SVC)):
        return shap.KernelExplainer(model.predict_proba, X_train)
    else:
        return shap.Explainer(model, X_train)  # fallback
    

def decompose_shap_to_features(
    shap_values: np.ndarray,
    feature_names: list,
    feature_mappings: dict
) -> pd.DataFrame:
    """
    Given:
      - shap_values: array of shape (n_samples, n_latent_dims_total)
      - feature_names: list of column names matching shap_values' columns
      - feature_mappings: dict[group_id] -> mapping_df (latent_dims × original_feats)
    Returns:
      - shap_per_feature: DataFrame (n_samples × total_original_features)
    """
    # wrap into DataFrame for easier slicing
    latent_df = pd.DataFrame(shap_values, columns=feature_names)
    all_feature_shap = []

    for group_id, mapping_df in feature_mappings.items():
        # identify the latent columns for this group
        latent_cols = mapping_df.index.tolist()
        # pull the shap values for them
        latent_shap = latent_df[latent_cols].to_numpy()  # (n_samples, encoding_dim)

        # normalize absolute decoder weights per feature
        W = mapping_df.to_numpy()  # (encoding_dim, d_k)
        w_abs = np.abs(W)
        norm = w_abs.sum(axis=0)  # sum over latent dims → for each feature
        # avoid division by zero
        norm[norm == 0] = 1.0
        weights = w_abs / norm  # (encoding_dim, d_k) now each column sums to 1

        # decompose: (n_samples × encoding_dim) dot (encoding_dim × d_k)
        shap_feats = latent_shap.dot(weights)  # (n_samples, d_k)

        # make DataFrame
        df_feats = pd.DataFrame(
            shap_feats,
            columns=mapping_df.columns,
            index=latent_df.index
        )
        all_feature_shap.append(df_feats)

    # concatenate all groups' feature SHAP
    shap_per_feature = pd.concat(all_feature_shap, axis=1)
    return shap_per_feature

# def evaluate_shap_features(
#     X_train: pd.DataFrame,
#     X_test: pd.DataFrame,
#     trained_models: dict,
#     encoders: dict,
#     feature_mappings: dict,
#     max_display: int = 10
# ):
#     """
#     For each model:
#       1. compute SHAP values on the latent group encodings
#       2. decompose them back to individual features
#       3. plot the top `max_display` features by mean absolute SHAP
#     """
#     if not encoders:
#         for name, model in trained_models.items():
#             print(f"\nSHAP (raw) → {name}")
#             # background = kmeans on X_train
#             bg_k = min(200, X_train.shape[0])
#             bg   = shap.kmeans(X_train, bg_k)
#             explainer = shap.KernelExplainer(model.predict_proba, bg)
#             shap_values = explainer(X_test)
#             # if multiclass
#             vals = shap_values.values if hasattr(shap_values, "values") else shap_values
#             if vals.ndim == 3:
#                 vals = np.sum(vals, axis=2)
#             shap.summary_plot(
#                 vals, 
#                 X_test, 
#                 feature_names=X_test.columns.tolist(),
#                 plot_type="dot", 
#                 max_display=max_display, 
#                 show=False
#             )
#             plt.title(f"{name} – SHAP (raw)")
#             plt.tight_layout()
#             plt.show()
#         return
#     # first, transform X_train/X_test via the same autoencoders to get latent features
#     # assume user has already obtained `group_embeddings_train, encoders, mappings`
#     # and `group_embeddings_test` via the same call to create_group_autoencoders.
#     # Here we just get `X_test_latent` and `latent_feature_names`.
#     latent_cols = []
#     X_test_latent = pd.DataFrame(index=X_test.index)

#     for gid, (scaler, ae, _) in encoders.items():
#         feats = feature_mappings[gid].columns.tolist()
#         enc_dim = ae.coefs_[1].shape[0]
#         X_scaled = scaler.transform(X_test[feats].to_numpy())
#         W0, b0 = ae.coefs_[0], ae.intercepts_[0]
#         Z = X_scaled.dot(W0) + b0
#         if ae.activation == 'relu':
#             codes = np.maximum(0, Z)
#         elif ae.activation == 'identity':
#             codes = Z
#         elif ae.activation == 'logistic':
#             codes = 1.0 / (1.0 + np.exp(-Z))
#         elif ae.activation == 'tanh':
#             codes = np.tanh(Z)
#         else:
#             raise ValueError(f"Unsupported activation: {ae.activation}")
        
#         if enc_dim == 1:
#             col = f'group_{gid}'
#             X_test_latent[col] = codes.ravel()
#             latent_cols.append(col)
#         else:
#             for d in range(enc_dim):
#                 col = f'group_{gid}_dim{d}'
#                 X_test_latent[col] = codes[:, d]
#                 latent_cols.append(col)

#     # run SHAP on latent space
#     for name, model in trained_models.items():
#         print(f"\nSHAP → individual features: {name}")

#         # If this would be a KernelExplainer, use a small background for speed
#         if isinstance(model, (LogisticRegression, SVC, MLPClassifier)):
#             # choose k cluster centers (e.g. 50) or drop-in shap.sample(X_train, 200)
#             bg_size = min(50, X_train.shape[0])
#             bg = shap.kmeans(X_train, bg_size)
#             explainer = shap.KernelExplainer(model.predict_proba, bg)
#         else:
#             # TreeExplainer, DeepExplainer, etc., untouched
#             explainer = get_shap_explainer(model, X_train)

#         shap_vals = explainer(X_test_latent)
#         # get raw array
#         raw = shap_vals.values if hasattr(shap_vals, 'values') else shap_vals
#         if raw.ndim == 3:
#             raw = np.sum(raw, axis=2)

#         # decompose back to original features
#         shap_feat_df = decompose_shap_to_features(
#             raw, latent_cols, feature_mappings
#         )

#         # aggregate mean absolute
#         mean_abs = shap_feat_df.mean().sort_values(ascending=False)
#         top = mean_abs.head(max_display)

#         shap.summary_plot(
#             shap_feat_df.values,                        # numpy array (n_samples, n_features)
#             X_test[shap_feat_df.columns],               # DataFrame of the same original features
#             plot_type="dot",                            # the classic beeswarm
#             max_display=max_display,                    # how many features to show
#             color_bar=True                              # show the color legend
#         )

def evaluate_shap_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    trained_models: dict,
    encoders: dict,
    feature_mappings: dict,
    max_display: int = 10
):
    """
    For each model:
      - If no encoders: compute per‐class SHAP on raw features
      - Otherwise:
          1. compute per‐class SHAP values on the latent group encodings
          2. decompose them back to individual features
          3. plot the top `max_display` features by mean absolute SHAP for each class
    """
    # —— RAW FEATURES CASE ——
    if not encoders:
        for name, model in trained_models.items():
            print(f"\nSHAP (raw) → {name}")
            bg_k = min(200, X_train.shape[0])
            bg   = shap.kmeans(X_train, bg_k)
            explainer = shap.KernelExplainer(model.predict_proba, bg)
            shap_out = explainer(X_test)
            vals = shap_out.values if hasattr(shap_out, "values") else shap_out

            # multiclass?
            if vals.ndim == 3:
                n_classes = vals.shape[2]
                for c in range(n_classes):
                    class_vals = vals[:, :, c]
                    shap.summary_plot(
                        class_vals,
                        X_test,
                        feature_names=X_test.columns.tolist(),
                        plot_type="dot",
                        max_display=max_display,
                        show=False
                    )
                    plt.title(f"{name} – SHAP (raw) for class {c}")
                    plt.tight_layout()
                    plt.show()
            else:
                # binary or single‐output
                shap.summary_plot(
                    vals,
                    X_test,
                    feature_names=X_test.columns.tolist(),
                    plot_type="dot",
                    max_display=max_display,
                    show=False
                )
                plt.title(f"{name} – SHAP (raw)")
                plt.tight_layout()
                plt.show()
        return

    # —— GROUPED FEATURES CASE ——
    # first compute the latent encodings for X_test
    latent_cols = []
    X_test_latent = pd.DataFrame(index=X_test.index)
    for gid, (scaler, ae, _) in encoders.items():
        feats   = feature_mappings[gid].columns.tolist()
        enc_dim = ae.coefs_[1].shape[0]
        Xs = scaler.transform(X_test[feats].to_numpy())
        Z  = Xs.dot(ae.coefs_[0]) + ae.intercepts_[0]
        if ae.activation == 'relu':
            codes = np.maximum(0, Z)
        elif ae.activation == 'identity':
            codes = Z
        elif ae.activation == 'logistic':
            codes = 1 / (1 + np.exp(-Z))
        elif ae.activation == 'tanh':
            codes = np.tanh(Z)
        else:
            raise ValueError(f"Unsupported activation: {ae.activation}")

        if enc_dim == 1:
            col = f'group_{gid}'
            X_test_latent[col] = codes.ravel()
            latent_cols.append(col)
        else:
            for d in range(enc_dim):
                col = f'group_{gid}_dim{d}'
                X_test_latent[col] = codes[:, d]
                latent_cols.append(col)

    # now run SHAP and decompose per‐class
    for name, model in trained_models.items():
        print(f"\nSHAP → individual features: {name}")

        # pick explainer
        if isinstance(model, (LogisticRegression, SVC, MLPClassifier)):
            bg_size = min(50, X_train.shape[0])
            bg = shap.kmeans(X_train, bg_size)
            explainer = shap.KernelExplainer(model.predict_proba, bg)
        else:
            explainer = get_shap_explainer(model, X_train)

        shap_out = explainer(X_test_latent)
        raw = shap_out.values if hasattr(shap_out, 'values') else shap_out

        # multiclass?
        if raw.ndim == 3:
            n_classes = raw.shape[2]
            for c in range(n_classes):
                raw_c = raw[:, :, c]                          # (n_samples, n_latent)
                shap_feat_df = decompose_shap_to_features(
                    raw_c, latent_cols, feature_mappings
                )
                # summary plot on original features for class c
                shap.summary_plot(
                    shap_feat_df.values,
                    X_test[shap_feat_df.columns],
                    plot_type="dot",
                    max_display=max_display,
                    color_bar=True,
                    show=False
                )
                plt.title(f"{name} – SHAP decomposed for class {c}")
                plt.tight_layout()
                plt.show()

        else:
            # binary or single‐output
            shap_feat_df = decompose_shap_to_features(
                raw, latent_cols, feature_mappings
            )
            shap.summary_plot(
                shap_feat_df.values,
                X_test[shap_feat_df.columns],
                plot_type="dot",
                max_display=max_display,
                color_bar=True,
                show=False
            )
            plt.title(f"{name} – SHAP decomposed")
            plt.tight_layout()
            plt.show()
