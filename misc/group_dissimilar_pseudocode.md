### Algorithm: Grouping Dissimilar Features using Conditional Mutual Information (CMI)

**Input:**
- `data`: Preprocessed dataset (DataFrame), where the last column is the target variable
- `num_groups`: Number of dissimilar groups to form

**Output:**
- `groups`: A dictionary mapping group indices to lists of feature names

**Steps:**

1. **Extract Features and Target:**
   - Let `features` = all columns except the last one.
   - Let `target` = the last column.

2. **Compute Dissimilarity Matrix:**
   - For each pair of features `(i, j)`:
     - Compute `CMI(X_i; X_j | target)` using `pyitlib`.
     - Compute entropies `H(X_i)` and `H(X_j)`.
     - Normalize CMI as `normalized_cmi = CMI / (H(X_i) + H(X_j) + ε)`.
     - Set `dissimilarity[i][j] = 1 - normalized_cmi`.

3. **Initialize Groups:**
   - Create `num_groups` empty groups.
   - Identify `num_groups` seed features with the highest average dissimilarity.
   - Assign one seed to each group.

4. **Greedy Assignment:**
   - While there are unassigned features:
     - For each group:
       - For each unassigned feature:
         - Compute average dissimilarity between the feature and all features in the group.
       - Select the feature with the maximum average dissimilarity.
       - Assign it to the current group.

5. **Return `groups`.**
