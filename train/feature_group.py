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
from itertools import combinations
from scipy.spatial.distance import pdist, squareform
from scipy.stats import skew, kurtosis
from sklearn.cluster import KMeans
import numpy as np

def group_similar(data: pd.DataFrame):
    raise NotImplementedError

def dissimilarity_matrix(data: pd.DataFrame):
    features = data.columns[:-1]
    target = data.columns[-1]
    n = len(features)

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
            dissim = 1 - normalized_cmi
            dissimilarity[i, j] = dissim
            dissimilarity[j, i] = dissim
    return dissimilarity

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
    dissimilarity = dissimilarity_matrix(data)
    # print(dissimilarity)
    sns.heatmap(dissimilarity, xticklabels=features, yticklabels=features)
    plt.title("Conditional Mutual Information (CMI) Heatmap")
    plt.xticks(rotation=90)  # Optional: rotate x-axis labels for readability
    plt.yticks(rotation=0)
    plt.tight_layout()
    # plt.show()
    
    ####################################################
    # is_psd, G, eigenvals = is_euclidean_distance_matrix(dissimilarity)

    # if is_psd:
    #     print("✅ The distance matrix is Euclidean — it can be embedded in 2D (possibly with low error).")
    # else:
    #     print("⚠️ The distance matrix is not strictly Euclidean — 2D embedding will involve distortion.")

    # mds = MDS(n_components=3, dissimilarity='precomputed', random_state=42)
    # coords = mds.fit_transform(dissimilarity)

    # # Plotting
    # fig = plt.figure(figsize=(10, 7))
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], s=100)

    # for i, label in enumerate(features):  # replace with your labels
    #     ax.text(coords[i, 0], coords[i, 1], coords[i, 2], label, fontsize=10)

    # ax.set_title("3D MDS Projection of Distance Matrix")
    # plt.tight_layout()
    # plt.show()
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
    
    evaluate_grouping(groups, features.tolist(), dissimilarity, method_name="group_dissimilar")
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



def diversity(group, dist_matrix):
    if len(group) < 2:
        return np.nan
    return sum(dist_matrix[i, j] for i, j in combinations(group, 2))

def dispersion(group, dist_matrix):
    if len(group) < 2:
        return np.nan
    return min(dist_matrix[i, j] for i, j in combinations(group, 2))

def score(groups, dist_matrix, x1, x2):
    f1 = sum(diversity(group, dist_matrix) for group in groups)
    f2 = min(dispersion(group, dist_matrix) for group in groups)
    return x1 * f1 + x2 * f2


def evaluate_grouping(groups: dict, feature_list: list, dist_matrix: np.ndarray, method_name=""):
    group_indices = [[feature_list.index(f) for f in group] for group in groups.values()]
    
    diversity_scores = [diversity(g, dist_matrix) for g in group_indices]
    dispersion_scores = [dispersion(g, dist_matrix) for g in group_indices]

    avg_div = np.nansum(diversity_scores) / len(feature_list)
    min_disp = np.nanmin(dispersion_scores)

    print(f"\n[{method_name}] Evaluation:")
    print(f" - Average Diversity:  {avg_div:.4f}")
    print(f" - Minimum Dispersion: {min_disp:.4f}")
    
    return avg_div, min_disp

def bicriterion_anticlustering(data: pd.DataFrame, k: int = 5, restarts: int = 10):
    """
    Perform bicriterion anticlustering on features to form k groups that are internally diverse.
    Uses a weighted combination of:
      - diversity: total pairwise dissimilarity within groups (maximize)
      - dispersion: minimum pairwise dissimilarity within groups (maximize)

    Parameters:
    - data: pd.DataFrame, where rows are samples and the last column is the target
    - k: int, number of desired feature groups
    - restarts: int, number of random restarts to improve robustness

    Returns:
    - best_groups: dict mapping group index to list of feature names
    """
    # Step 1: Compute dissimilarity matrix between features (using CMI)
    dissim = dissimilarity_matrix(data)
    features = data.columns[:-1].tolist()  # exclude target
    n = len(features)

    # Step 2: Convert feature index list to named groups
    best_score = -np.inf
    best_partition = None

    for _ in range(restarts):
        x1 = np.random.choice([.0001, .001, .01, .1, .5, .99])
        x2 = 1 - x1

        indices = np.random.permutation(n)
        groups = [indices[i::k].tolist() for i in range(k)]  # k roughly equal groups

        improved = True
        while improved:
            improved = False
            for i in range(k):
                for j in range(i + 1, k):
                    for a in groups[i]:
                        for b in groups[j]:
                            if a == b:
                                continue
                            # Create new swapped groups
                            new_groups = [g.copy() for g in groups]
                            if a in new_groups[i] and b in new_groups[j]:
                                new_groups[i] = [b if x == a else x for x in new_groups[i]]
                                new_groups[j] = [a if x == b else x for x in new_groups[j]]
                            else:
                                continue

                            new_score = score(new_groups, dissim, x1, x2)
                            current_score = score(groups, dissim, x1, x2)
                            if new_score > current_score:
                                groups = new_groups
                                improved = True

        final_score = score(groups, dissim, x1, x2)
        if final_score > best_score:
            best_score = final_score
            best_partition = groups

    # After all restarts
    if best_partition is None:
        print(f"[Warning] No valid partition found for k = {k}. Returning last attempted groups.")
        best_partition = groups  # fallback to last tried groups

    # Convert best_partition (indices) to feature names
    group_dict = {i: [features[idx] for idx in group] for i, group in enumerate(best_partition)}
    print("\nBicriterion anticlustering - groups:")
    for group_id, feats in group_dict.items():
        print(f" Group {group_id}: {feats}")
    # Evaluate result
    evaluate_grouping(group_dict, features, dissim, method_name="bicriterion_anticlustering")
    return group_dict


def evaluate_k_range(data: pd.DataFrame, method_name: str, k_range=range(2, 11)):
    """
    Evaluate the anticlustering quality over different k values using diversity and dispersion.

    Parameters:
    - data: pd.DataFrame with features and target
    - method_name: 'group_dissimilar', 'k_plus', or 'bicriterion'
    - k_range: range of k values to test

    Returns:
    - results: pd.DataFrame with columns ['k', 'avg_diversity', 'min_dispersion']
    """
    results = []

    for k in k_range:
        if method_name == 'group_dissimilar':
            groups = group_dissimilar(data.copy(), num_groups=k)
        elif method_name == 'k_plus':
            groups = k_plus_anticlustering(data.copy(), k=k, verbose=False)
        elif method_name == 'bicriterion':
            groups = bicriterion_anticlustering(data.copy(), k=k, restarts=5)
        else:
            raise ValueError("Unknown method")

        features = data.columns[:-1].tolist()
        diss = dissimilarity_matrix(data)
        avg_div, min_disp = evaluate_grouping(groups, features, diss, method_name=f"{method_name}_k={k}")

        results.append({'k': k, 'avg_diversity': avg_div, 'min_dispersion': min_disp})

    return pd.DataFrame(results)

def plot_k_selection(results: pd.DataFrame, method_name: str):
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.plot(results['k'], results['avg_diversity'], marker='o', label='Avg Diversity', color='teal')
    ax1.set_xlabel('k')
    ax1.set_ylabel('Average Diversity', color='teal')
    ax1.tick_params(axis='y', labelcolor='teal')

    ax2 = ax1.twinx()
    ax2.plot(results['k'], results['min_dispersion'], marker='s', label='Min Dispersion', color='orange')
    ax2.set_ylabel('Minimum Dispersion', color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')

    plt.title(f'Group Evaluation vs k ({method_name})')
    fig.tight_layout()
    plt.grid(True)
    plt.show()


def k_plus_anticlustering(data, k=5, verbose=True):
    """
    Perform K-Plus anticlustering on features by augmenting with higher-order statistics
    and clustering the transposed matrix, using CMI-based dissimilarity.
    """
    # Separate features and target
    feature_cols = data.columns[:-1]
    target_col = data.columns[-1]

    # Drop constant features
    variances = data[feature_cols].var(axis=0)
    constant_columns = variances[variances == 0].index.tolist()
    non_constant_columns = variances[variances > 0].index

    data = data[non_constant_columns.tolist() + [target_col]]  # include target again

    if verbose and constant_columns:
        print(f"\n[Info] Excluded {len(constant_columns)} constant feature(s):")
        print(" ", constant_columns)

    # Compute dissimilarity AFTER filtering
    dissimilarity = dissimilarity_matrix(data)

    # Convert DataFrame to NumPy array (features only)
    feature_cols = data.columns[:-1]
    data_array = data[feature_cols].values
    data_T = data_array.T  # Shape: (n_features, n_samples)

    # Augment with skewness and kurtosis
    sk = skew(data_array, axis=0)
    kurt = kurtosis(data_array, axis=0)
    augmented_data = np.hstack([data_T, sk[:, np.newaxis], kurt[:, np.newaxis]])

    # Apply k-means
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
    labels = kmeans.fit_predict(augmented_data)

    features = feature_cols.tolist()
    group_dict = {i: [] for i in range(k)}
    for idx, label in enumerate(labels):
        group_dict[label].append(features[idx])

    if verbose:
        print("\nK-Plus Anticlustering - Feature Groups:")
        for group_id, feats in group_dict.items():
            print(f" Group {group_id}: {feats}")

    evaluate_grouping(group_dict, features, dissimilarity, method_name="k_plus_anticlustering")
    return group_dict


# Step 3: Create embeddings using autoencoders
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np

def embed_feature_groups(data: pd.DataFrame, groups: dict, loss_threshold: float = 1e-2, encoding_dim: int = 5,
                         fit: bool = True, encoders: dict = None):
    embeddings = pd.DataFrame(index=data.index)
    trained_encoders = {}

    for group_id, group_feats in groups.items():
        # 🛡️ Validate presence
        missing = [feat for feat in group_feats if feat not in data.columns]
        if missing:
            raise ValueError(f"[Group {group_id}] Missing features in data: {missing}")

        # ✳️ Ensure 2D shape: (samples, features)
        X_df = data[group_feats]
        X = X_df.to_numpy()
        if X.ndim != 2:
            raise ValueError(f"[Group {group_id}] Feature matrix must be 2D, got shape {X.shape}")

        if fit:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            X_train, X_val = train_test_split(X_scaled, test_size=0.2, random_state=42)

            best_encoding = None
            for dim in range(1, encoding_dim + 1):
                model = MLPRegressor(hidden_layer_sizes=(dim,), activation='relu', max_iter=2000, random_state=42)
                model.fit(X_train, X_train)
                reconstructed = model.predict(X_val)
                loss = mean_squared_error(X_val, reconstructed)
                if loss <= loss_threshold:
                    best_encoding = dim
                    break
            if best_encoding is None:
                best_encoding = encoding_dim

            # Refit on full scaled data
            model = MLPRegressor(hidden_layer_sizes=(best_encoding,), activation='relu', max_iter=2000, random_state=42)
            model.fit(X_scaled, X_scaled)
            encoded = model.predict(X_scaled)

            # 🧠 Save encoder
            trained_encoders[group_id] = (scaler, model, best_encoding)

        else:
            scaler, model, best_encoding = encoders[group_id]
            X_scaled = scaler.transform(X)
            encoded = model.predict(X_scaled)

        # 🔒 Validate output shape
        if encoded.shape[0] != X.shape[0]:
            raise ValueError(f"[Group {group_id}] Encoded output has shape {encoded.shape}, expected {X.shape[0]} rows")

        # 📦 Store embedding
        if encoded.ndim == 1 or encoded.shape[1] == 1:
            embeddings[f"group_{group_id}"] = encoded.ravel()
        else:
            for d in range(encoded.shape[1]):
                embeddings[f"group_{group_id}_dim{d}"] = encoded[:, d]


        if fit:
            print(f"[Group {group_id}] Selected encoding dim: {best_encoding}")

    return (embeddings, trained_encoders) if fit else embeddings


