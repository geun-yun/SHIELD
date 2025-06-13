from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

def get_models(random_state=42):
    """
    Returns a dictionary of initialized ML models.
    """
    return {
        # "LinearRegression": LinearRegression(),
        "LogisticRegression": LogisticRegression(max_iter=2000, solver='lbfgs', random_state=random_state),
        # "SVM": SVC(kernel='rbf', probability=True, random_state=random_state),
        # "MLP": MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=random_state),
        # "RandomForest": RandomForestClassifier(n_estimators=100, random_state=random_state),
        # "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=random_state)
    }
