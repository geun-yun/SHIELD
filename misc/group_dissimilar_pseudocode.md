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


## Example (K = 2)
**Suppose that the Dissimilarity Matrix has been computed as below.**
|     |A    |B    |C    |D    |
|-----|-----|-----|-----|-----|
|A    |0    |0.2  |0.9  |0.8  |
|B    |0.2  |0    |0.85 |0.75 |
|C    |0.9  |0.85 |0    |0.1  |
|D    |0.8  |0.75 |0.1  |0    |

**Compute row-wise sum:**
- A: 0 + 0.2 + 0.9 + 0.8 = **1.9**
- B: 0.2  + 0 + 0.85 + 0.75 = 1.8
- C: 0.9 + 0.85 + 0 + 0.1 = **1.85**
- D: 0.8 + 0.75 + 0.1 + 0 = 1.65

**Pick top K sum to be the first element for each group.**
- Group 0: [A]
- Group 1: [C]
- Remaining: B, D

**Compare remaining features with each group via average dissimilarity.**
- B against Group 0 &rarr; B against A &rarr; 0.2
- B against Group 1 &rarr; B against C &rarr; **0.85**

**Allocate features to the group with highest average dissimilarity.**
- Group 0: [A]
- Group 1: [C, B]
- Remaining: D

**Repeat the allocation process until all features have been allocated.**
- D against Group 0 &rarr; D against A &rarr; **0.8**
- D against Group 1 &rarr; D against C and D against B &rarr; avg(0.1,0.75) = 0.425

**Results**
- Group 0: [A, D]
- Group 1: [C, B]



