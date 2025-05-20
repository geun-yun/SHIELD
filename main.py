from preprocess.preprocess_main import preprocess_all
from train.feature_group import group_dissimilar, create_group_autoencoders
from train.models import get_models
from evaluate import evaluate_metrics, evaluate_shap

from sklearn.model_selection import train_test_split

def run_full_pipeline():
    models = get_models()
    for model in models.keys():
        datasets = list(preprocess_all(model))
        for train_data, test_data in datasets:
            train_and_test(model, train_data, test_data)

def train_and_test(model, train_data, test_data, num_groups=5, encoding_dim=1, test_size=0.2, random_state=42):
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
    train_X = train_data[:-1]
    train_Y = train_data[-1]
    test_X = test_data[:-1]
    test_Y = test_data[-1]
    # Group features and encode
    groups = group_dissimilar(train_data, num_groups=num_groups)
    # encoded_data = create_group_autoencoders(data, groups, encoding_dim=encoding_dim)

    # # # Add the target column back
    # # encoded_data["target"] = data.iloc[:, -1].values

    # # # Split data
    # # X = encoded_data.drop(columns=["target"])
    # # y = encoded_data["target"]
    # # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # # # Evaluate performance
    # # results = evaluate_metrics(X_train, y_train, X_test, y_test, task_name=dataset_name)
    # # print(results)

    # # # Evaluate SHAP
    # # evaluate_shap(X_train, y_train, X_test, dataset_name)


run_pipeline("Obesity")
run_pipeline("Breast_cancer")
# run_full_pipeline("Heart_disease")
run_pipeline("Lung_cancer")