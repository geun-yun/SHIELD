from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical

def get_models(random_state=42):
    """
    Returns a dictionary of initialized ML models.
    """
    return {
        "LogisticRegression": LogisticRegression(max_iter=2000, solver='lbfgs', random_state=random_state),
        "SVM": SVC(kernel='rbf', probability=True, random_state=random_state),
        "MLP": MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, random_state=random_state),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=random_state),
        "XGBoost": XGBClassifier(eval_metric='logloss', random_state=random_state)
    }


def get_bayes_spaces():
    return {
      "LogisticRegression": {
        "model__C": Real(1e-2, 1e2, prior="log-uniform")
      },
      "SVC": {
        "model__C":     Real(1e-2, 1e2, prior="log-uniform"),
        "model__gamma": Real(1e-3, 1.0, prior="log-uniform")
      },
      "MLP": {
        "model__hidden_layer_sizes": Categorical([(50,), (100,), (200,)]),
        "model__alpha":              Real(1e-4, 1e-1, prior="log-uniform"),
        "model__learning_rate_init": Real(1e-3, 3e-1, prior="log-uniform")
      },
      "RandomForest": {
        "model__n_estimators": Integer(50, 500),
        "model__max_depth":    Integer(3, 15),
        "model__max_features": Categorical(["sqrt","log2", None])
      },
      "XGBoost": {
        "model__n_estimators":   Integer(50, 500),
        "model__max_depth":      Integer(3, 15),
        "model__learning_rate":  Real(1e-3, 3e-1, prior="log-uniform"),
        "model__subsample":      Real(0.6, 1.0),
        "model__colsample_bytree": Real(0.6, 1.0)
      }
    }


def bayes_tune_models(X_train, y_train, model_name, cv_splits=5, n_iter=30, random_state=42):
    """
    Runs Bayesian hyperparameter search for the selected model only.
    """
    base_models = get_models(random_state)
    if model_name not in base_models:
        raise ValueError(f"Model '{model_name}' not found in get_models()")

    model = base_models[model_name]
    search_space = get_bayes_spaces()[model_name]

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    print(f"Bayesian tuning for {model_name}")
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", model)
    ])

    bayes = BayesSearchCV(
        estimator=pipe,
        search_spaces=search_space,
        n_iter=n_iter,
        scoring="accuracy",
        cv=cv,
        random_state=random_state,
        n_jobs=-1,
        verbose=0
    )

    bayes.fit(X_train, y_train)
    print(f"best params: {bayes.best_params_}")
    print(f"best CV score: {bayes.best_score_:.4f}")

    return bayes

