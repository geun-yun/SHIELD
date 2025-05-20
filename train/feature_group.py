import numpy as np
import pandas as pd
from pyitlib import discrete_random_variable as drv
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from sklearn.manifold import MDS


def group_similar(data: pd.DataFrame):
    raise NotImplementedError

def group_dissimilar(data: pd.DataFrame, num_groups: int = 5):
    """
    Groups features into sets with maximal dissimilarity (diversity maximization) using conditional mutual information (CMI).
    
    Parameters:
    - data: pd.DataFrame, preprocessed dataset where the last column is the target
    - num_groups: int, number of dissimilar groups to form
    
    Returns:
    - groups: dict, keys are group indices and values are lists of feature names
    """
    features = data.columns[:-1]
    target = data.columns[-1]
    n = len(features)

    # Step 1: Compute pairwise normalized CMI as similarity, then invert to get dissimilarity
    dissimilarity = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            x = data[features[i]].values
            y = data[features[j]].values
            z = data[target].values

            cmi = drv.information_mutual_conditional(x, y, z)
            h_x = drv.entropy(x)
            h_y = drv.entropy(y)
            normalized_cmi = cmi / (h_x + h_y + 1e-9)
            dissim = normalized_cmi
            dissimilarity[i, j] = dissim
            dissimilarity[j, i] = dissim
    print(dissimilarity)
    sns.heatmap(dissimilarity, xticklabels=features, yticklabels=features)
    plt.title("Conditional Mutual Information (CMI) Heatmap")
    plt.xticks(rotation=90)  # Optional: rotate x-axis labels for readability
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
    
    ####################################################
    is_psd, G, eigenvals = is_euclidean_distance_matrix(dissimilarity)

    if is_psd:
        print("✅ The distance matrix is Euclidean — it can be embedded in 2D (possibly with low error).")
    else:
        print("⚠️ The distance matrix is not strictly Euclidean — 2D embedding will involve distortion.")

    mds = MDS(n_components=3, dissimilarity='precomputed', random_state=42)
    coords = mds.fit_transform(dissimilarity)

    # Plotting
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], s=100)

    for i, label in enumerate(features):  # replace with your labels
        ax.text(coords[i, 0], coords[i, 1], coords[i, 2], label, fontsize=10)

    ax.set_title("3D MDS Projection of Distance Matrix")
    plt.tight_layout()
    plt.show()
    ####################################################
    # Step 2: Greedy grouping to maximize diversity
    groups = {i: [] for i in range(num_groups)}
    assigned = set()
    remaining = set(range(n))

    # Seed with one feature per group, choosing the most mutually dissimilar
    seeds = np.argsort(dissimilarity.sum(axis=1))[-num_groups:]
    for group_id, seed_idx in enumerate(seeds):
        groups[group_id].append(features[seed_idx])
        assigned.add(seed_idx)
        remaining.discard(seed_idx)

    # Greedily assign the most dissimilar feature to each group
    while remaining:
        for group_id in range(num_groups):
            if not remaining:
                break
            # Compute average dissimilarity of each unassigned feature to current group
            max_score = -1
            best_feature_idx = None
            for idx in remaining:
                group_idxs = [features.get_loc(f) for f in groups[group_id]]
                avg_dissim = np.mean([dissimilarity[idx, gidx] for gidx in group_idxs])
                if avg_dissim > max_score:
                    max_score = avg_dissim
                    best_feature_idx = idx
            if best_feature_idx is not None:
                groups[group_id].append(features[best_feature_idx])
                assigned.add(best_feature_idx)
                remaining.discard(best_feature_idx)

    # Print grouped features for tracking
    print("\nGrouped features (maximally dissimilar):")
    for group_id, feature_list in groups.items():
        print(f" Group {group_id}: {feature_list}")
        
    return groups


def create_group_autoencoders(data: pd.DataFrame, groups: dict, encoding_dim: int = 1):
    """
    Create and train one hidden-layer autoencoders per feature group.
    Parameters:
    - data: pd.DataFrame, dataset with original features
    - groups: dict, grouping of features from `group_dissimilar_diverse`
    - encoding_dim: int, dimension of latent space for each group

    Returns:
    - group_embeddings: pd.DataFrame, new dataset with one column per group (encoded features)
    """
    group_embeddings = pd.DataFrame(index=data.index)
    scaler = StandardScaler()

    for group_id, features in groups.items():
        X = scaler.fit_transform(data[features])
        X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

        model = MLPRegressor(hidden_layer_sizes=(encoding_dim,), activation='relu', max_iter=1000, random_state=42)
        model.fit(X_train, X_train)

        encoded = model.predict(X)
        if encoding_dim == 1:
            group_embeddings[f'group_{group_id}'] = encoded.ravel()
        else:
            for dim in range(encoded.shape[1]):
                group_embeddings[f'group_{group_id}_dim{dim}'] = encoded[:, dim]


    return group_embeddings


def is_euclidean_distance_matrix(D):
    """
    Check whether a distance matrix D is Euclidean by computing its Gram matrix,
    and plot the cumulative explained variance from its eigenvalues in descending order.
    """
    # Double centering: G = -0.5 * H D^2 H
    n = D.shape[0]
    H = np.eye(n) - np.ones((n, n)) / n
    D_squared = D ** 2
    G = -0.5 * H @ D_squared @ H

    # Compute eigenvalues
    eigenvals = eigh(G, eigvals_only=True)
    print("Eigenvalues of Gram matrix:", eigenvals)

    # Filter and sort positive eigenvalues in descending order
    eigenvals_pos = np.sort(eigenvals[eigenvals > 0])[::-1]
    explained_top3 = np.sum(eigenvals_pos[:3]) / np.sum(eigenvals_pos)
    print(f"Top 3 dimensions explain {explained_top3:.2%} of total positive variance.")

    # Explained variance
    explained_variance_ratio = eigenvals_pos / np.sum(eigenvals_pos)
    cumulative_variance = np.cumsum(explained_variance_ratio)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(1, len(explained_variance_ratio) + 1),
           explained_variance_ratio,
           alpha=0.6,
           color='coral',
           label='Individual Component Variance')

    ax.plot(range(1, len(cumulative_variance) + 1),
            cumulative_variance,
            marker='o',
            color='deepskyblue',
            label='Cumulative Variance')

    for i, val in enumerate(cumulative_variance):
        ax.text(i + 1, val + 0.015, f"{int(val * 100)}%", ha='center', fontsize=9)

    ax.set_xticks(range(1, len(explained_variance_ratio) + 1))
    ax.set_xlabel("Principal Component Number (Most to Least)")
    ax.set_ylabel("Explained Variance")
    ax.set_title("Cumulative Explained Variance Plot (Descending Order)")
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()

    is_psd = np.all(eigenvals >= -1e-8)
    return is_psd, G, eigenvals


