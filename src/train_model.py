from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV


def train_xgb_model(X_train, y_train, param_dist, n_iter=10, scoring='f1'):
    """
    Train an XGBoost classifier using RandomizedSearchCV.

    Parameters:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.
        param_dist (dict): Hyperparameter search space.
        n_iter (int): Number of search iterations.
        scoring (str): Metric for model evaluation.

    Returns:
        tuple: (best_model, best_params)
    """
    model = XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring=scoring,
        cv=3,
        verbose=1,
        random_state=42,
        n_jobs=-1
    )

    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_