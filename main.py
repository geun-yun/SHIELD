from preprocess.preprocess_main import preprocess_all
from train.feature_group import evaluate_k_range, plot_k_selection, group_dissimilar, bicriterion_anticlustering, k_plus_anticlustering, embed_feature_groups
from train.models import get_models
from evaluate.evaluate import evaluate_metrics, run_kfold_cv
from evaluate.SHAP import evaluate_shap_features
from evaluate.fairness import fairness_metrics
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.base import clone
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run feature grouping and modeling pipeline")
    parser.add_argument("--grouping", type=str, default="ungroup", choices=["ungroup", "gd", "bicriterion", "kplus"],
                        help="Grouping method: 'ungroup', 'gd' (greedy dissimilar), 'bicriterion', 'kplus'")
    parser.add_argument("--model", type=str, default="LogisticRegression",
                        help="Model to train (must match name in get_models)")
    parser.add_argument("--dataset", type=str, default="default_dataset",
                        help="Dataset to use (must match available datasets in preprocess_all)")
    parser.add_argument("--k", type=int, default=5, help="Number of groups")
    parser.add_argument("--encoding_dim", type=int, default=5, help="Latent dimension size")
    parser.add_argument("--SHAP", type=str, default="aggregated", choices=["aggregated", "each_label"],
                        help="SHAP analysis: 'aggregated' returns one plot, 'each_label' returns a plot for each label")
    return parser.parse_args()
        
def run_full_pipeline():
    models = get_models()
    for model in models.keys():
        datasets = list(preprocess_all(model))
        for train_data, test_data in datasets:
            train_and_test(model, train_data, test_data)

def train_and_test(model, train_data, test_data, num_groups=4, encoding_dim=5, test_size=0.2, random_state=42):
    """
    Complete pipeline that:
    1. Preprocesses the dataset
    2. Groups dissimilar features using CMI
    3. Reduces each group using autoencoders
    4. Splits into training/testing
    5. Evaluates models for performance and SHAP

    Parameters:
    - dataset_name: str, one of the predefined dataset names
    - num_groups: int, number of dissimilar feature groups
    - encoding_dim: int, number of dimensions per group's latent representation
    - test_size: float, test split ratio
    - random_state: int, seed for reproducibility
    """
    print(f"\n==== Running pipeline for: {model} ====")
    train_X = train_data.iloc[:, :-1]
    train_Y = train_data.iloc[:, -1]
    test_X = test_data.iloc[:, :-1]
    test_Y = test_data.iloc[:, -1]
    # Group features and encode
    assert not train_data.isnull().any().any(), "NaNs detected in feature data before KMeans!"
    # groups = group_dissimilar(train_data, num_groups=num_groups)
    groups2 = bicriterion_anticlustering(train_data, num_groups)
    # # dissimilarity_matrix, feature_names = compute_dissimilarity(train_data)
    # # groups3 = anticluster_features(dissimilarity_matrix, feature_names, num_groups)
    # groups3 = k_plus_anticlustering(train_X)

    # results_kplus = evaluate_k_range(train_data, method_name='k_plus')
    # plot_k_selection(results_kplus, method_name='k_plus')

    # results_gd = evaluate_k_range(train_data, method_name='group_dissimilar')
    # plot_k_selection(results_gd, method_name='group_dissimilar')

    # results_bi = evaluate_k_range(train_data, method_name='bicriterion')
    # plot_k_selection(results_bi, method_name='bicriterion')

    # print("Iterative grouping")
    # print(groups)
    # print("Bicriterion")
    # print(groups2)
    # print("K plus anticlustering")
    # print(groups3)
    # encoded_train, encoders = embed_feature_groups(train_data, groups2, encoding_dim=encoding_dim, fit=True)

    train_emb, encoders, feature_mappings = embed_feature_groups(
        train_data, groups2, encoding_dim=encoding_dim, fit=True
        )
    test_emb = embed_feature_groups(test_data, groups2, encoding_dim=encoding_dim, encoders=encoders, fit=False)
    # encoded_train["target"] = train_Y.values
    # encoded_test["target"] = test_Y.values

    test_emb = pd.DataFrame(index=test_data.index)
    for gid, (scaler, ae, _) in encoders.items():
        feats   = feature_mappings[gid].columns.tolist()
        enc_dim = ae.coefs_[1].shape[0]

        Xs = scaler.transform(test_data[feats].to_numpy())
        W0, b0 = ae.coefs_[0], ae.intercepts_[0]
        Z_test = Xs.dot(W0) + b0
        if ae.activation == 'relu':
            codes = np.maximum(0, Z_test)
        elif ae.activation == 'identity':
            codes = Z_test
        elif ae.activation == 'logistic':
            codes = 1 / (1 + np.exp(-Z_test))
        elif ae.activation == 'tanh':
            codes = np.tanh(Z_test)
        else:
            raise ValueError(f"Unsupported activation: {ae.activation}")
        # …etc for other activations…

        if enc_dim == 1:
            test_emb[f'group_{gid}'] = codes.ravel()
        else:
            for d in range(enc_dim):
                test_emb[f'group_{gid}_dim{d}'] = codes[:, d]
    
    train_emb["target"] = train_Y.values
    test_emb["target"]  = test_Y.values
    X_train = train_emb.drop(columns=["target"])
    y_train = train_emb["target"]
    X_test  = test_emb.drop(columns=["target"])
    y_test  = test_emb ["target"]


    # Separate again for clarity
    # X_train = encoded_train.drop(columns=["target"])
    # y_train = encoded_train["target"]
    # X_test = encoded_test.drop(columns=["target"])
    # y_test = encoded_test["target"]

    # Evaluate
    print("\nEvaluating Model Performance")
    results, trained_models  = evaluate_metrics(X_train, y_train, X_test, y_test, task_name=model)
    print(results)
    # model = get_models()
    # model = model["LogisticRegression"]
    # model.fit(X_train, y_train)
    # y_pred = model.predict(X_test)
    # y_true = y_test
    # sensitive = test_data['sex']
    # print("\nFairness Metrics")
    # results = fairness_metrics(y_true, y_pred, sensitive, privileged_value=1)

    # Optional: SHAP evaluation
    print("\nExplaining with SHAP")
    # new
    evaluate_shap_features(
        X_train, test_X,
        trained_models,
        encoders,
        feature_mappings,
        max_display=10
    )

    # Add the target column back
    # encoded_data["target"] = data.iloc[:, -1].values

    # # Split data
    # X = encoded_data.drop(columns=["target"])
    # y = encoded_data["target"]
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # # Evaluate performance
    # results = evaluate_metrics(X_train, y_train, X_test, y_test, task_name=dataset_name)
    # print(results)

    # # Evaluate SHAP
    # evaluate_shap(X_train, y_train, X_test, dataset_name)
    print("UNGROUPED DATA")
    encoders= {}
    feature_mappings = {}
    X_train, y_train = train_X, train_Y
    X_test,  y_test  = test_X,  test_Y
    print("\nEvaluating Model Performance")
    results, trained_models  = evaluate_metrics(X_train, y_train, X_test, y_test, task_name=model)
    print(results)
    # model = get_models()
    # model = model["LogisticRegression"]
    # model.fit(X_train, y_train)
    # y_pred = model.predict(X_test)
    # y_true = y_test
    # sensitive = test_data['sex']
    # print("\nFairness Metrics")
    # results = fairness_metrics(y_true, y_pred, sensitive, privileged_value=1)
    # print(results)
    evaluate_shap_features(
        X_train, test_X,
        trained_models,
        encoders,
        feature_mappings,
        max_display=10
    )

def compare_grouping_methods(data, base_model, k=4, repeats=3):
    print("\nStarting fairness-aware evaluation via K-fold CV")

    y = data.iloc[:, -1]
    X_raw = data.iloc[:, :-1]

    # Performance on raw features
    print("Evaluating on raw features...")
    raw_results = run_kfold_cv(X_raw, y, clone(base_model), k=k, repeats=repeats)

    # Performance on grouped + embedded features
    print("Evaluating on grouped + embedded features...")
    groups = bicriterion_anticlustering(data, k=4)
    X_emb, encoders, feature_mappings = embed_feature_groups(data, groups, encoding_dim=5, fit=True)
    y_emb = y
    emb_results = run_kfold_cv(X_emb, y_emb, clone(base_model), k=k, repeats=repeats)

    # Print averaged performance
    print("\nAveraged Results (Raw vs Grouped)")
    for metric in raw_results.keys():
        raw_vals = [v for v in raw_results[metric] if v is not None]
        emb_vals = [v for v in emb_results[metric] if v is not None]
        print(f"{metric:>10}: Raw = {np.mean(raw_vals):.4f}, Grouped = {np.mean(emb_vals):.4f}")

    # Single hold-out for SHAP & Fairness
    print("\nHoldout comparison for fairness & SHAP")
    train_df, test_df = train_test_split(data, test_size=0.2,
                                         random_state=42,
                                         stratify=y)

    # Raw-model
    Xtr_raw, ytr_raw = train_df.iloc[:, :-1], train_df.iloc[:, -1]
    Xte_raw, yte_raw = test_df.iloc[:, :-1], test_df.iloc[:, -1]
    clf_raw = clone(base_model).fit(Xtr_raw, ytr_raw)
    ypred_raw = clf_raw.predict(Xte_raw)
    sens_raw = test_df['Gender_Male']      # or your sensitive col
    print("\nRaw fairness metrics:")
    print(fairness_metrics(yte_raw, ypred_raw, sens_raw, privileged_value=1))

    print("\nRaw SHAP top features:")
    evaluate_shap_features(
        Xtr_raw, Xte_raw,
        {"Raw": clf_raw},
        encoders={},                 # no encoders for raw
        feature_mappings={},         # no group mapping
        max_display=10
    )

    # Grouped-model
    # group & embed train and test separately
    groups_ho = bicriterion_anticlustering(train_df, k=4)
    Xtr_emb, encs_ho, maps_ho = embed_feature_groups(train_df, groups_ho, encoding_dim=5, fit=True)
    # apply same encoders to test_df:
    Xte_emb = embed_feature_groups(test_df, groups_ho, encoding_dim=5, encoders=encs_ho, fit=False)
    ytr_emb, yte_emb = train_df.iloc[:, -1], test_df.iloc[:, -1]

    clf_grp = clone(base_model).fit(Xtr_emb, ytr_emb)
    ypred_grp = clf_grp.predict(Xte_emb)
    sens_grp = test_df['Gender_Male']
    print("\nGrouped fairness metrics:")
    print(fairness_metrics(yte_emb, ypred_grp, sens_grp, privileged_value=1))

    print("\nGrouped SHAP top features:")
    evaluate_shap_features(
        Xtr_emb, test_df.iloc[:, :-1],   # pass original test_X so it can re-embed internally
        {"Grouped": clf_grp},
        encs_ho,
        maps_ho,
        max_display=10
    )

    return raw_results, emb_results


def run_comparison_pipeline():
    models = get_models()
    for model_name, model in models.items():
        datasets = list(preprocess_all(model_name))
        for dataset_id, (train_data, test_data) in enumerate(datasets):
            print(f"\n=== Dataset {dataset_id + 1} for model: {model_name} ===")
            full_data = pd.concat([train_data, test_data], axis=0).reset_index(drop=True)
            compare_grouping_methods(full_data, base_model=model)

# run_comparison_pipeline()
run_full_pipeline()