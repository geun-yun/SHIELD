import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from models import get_models

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

def bayes_tune_models(X_train, y_train, cv_splits=5, n_iter=30, random_state=42):
    base_models = get_models(random_state)
    search_spaces = get_bayes_spaces()

    tuned_searches = {}
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    for name, model in base_models.items():
        print(f"\n→ Bayesian tuning for {name}")
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model",  model)
        ])
        bayes = BayesSearchCV(
            estimator   = pipe,
            search_spaces = search_spaces[name],
            n_iter      = n_iter,
            scoring     = "roc_auc",
            cv          = cv,
            random_state= random_state,
            n_jobs      = -1,
            verbose     = 0
        )
        bayes.fit(X_train, y_train)
        print(f"   • best params: {bayes.best_params_}")
        print(f"   • best CV ROC AUC: {bayes.best_score_:.4f}")
        tuned_searches[name] = bayes

    return tuned_searches
