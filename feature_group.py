# Re-running the code after environment reset

import numpy as np
import pandas as pd
from pyitlib import discrete_random_variable as drv
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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
            dissim = 1 - normalized_cmi
            dissimilarity[i, j] = dissim
            dissimilarity[j, i] = dissim

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