"""
Universal Model Selector — Phase 1 + Phase 2
Implements the "clean -> validate -> predict" pipeline described in the
business plan.

Phase 1 (regression): Linear Regression, Decision Tree, Random Forest
Phase 1 (classification): Logistic Regression, Decision Tree, Random Forest
Phase 2 (regression): Ridge, Lasso, ElasticNet (all with built-in CV alpha
    tuning), K-Nearest Neighbors Regressor
Phase 2 (classification): K-Nearest Neighbors Classifier, Naive Bayes (Gaussian)

Train/Test split validation and K-Fold cross-validation applied to every model.

Phase 3: Gradient Boosting (regression and classification). Time-series
forecasting lives in forecasting.py.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import (
    LinearRegression, LogisticRegression, RidgeCV, LassoCV, ElasticNetCV
)
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier,
)
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)


def detect_problem_type(target: pd.Series) -> str:
    """
    Decide whether the target column calls for regression or classification.
    - Few unique values (<=20, or <5% of rows) and non-continuous -> classification
    - Otherwise numeric and continuous -> regression
    """
    if pd.api.types.is_numeric_dtype(target):
        nunique = target.nunique(dropna=True)
        # Small number of distinct integer-like values -> likely a category (e.g. risk tier 0/1/2)
        if nunique <= max(10, int(0.05 * len(target))) and (target.dropna() % 1 == 0).all():
            return "classification"
        return "regression"
    else:
        return "classification"


def prepare_features(df: pd.DataFrame, target_col: str, profile: dict):
    """Encode categorical features numerically and separate X / y."""
    df = df.copy()
    y = df[target_col]
    X = df.drop(columns=[target_col])

    # Drop free-text, datetime, and any leftover identifier columns from X for this
    # simple MVP (identifiers should already be dropped during cleaning, but this
    # is a safety net in case a column slips through).
    drop_cols = [c for c in X.columns if profile.get(c) in ("text", "datetime", "identifier")]
    X = X.drop(columns=drop_cols)

    # Encode categorical columns
    encoders = {}
    for col in X.columns:
        if profile.get(col) == "categorical":
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            encoders[col] = le

    # Encode target if classification and non-numeric
    target_encoder = None
    if detect_problem_type(y) == "classification" and not pd.api.types.is_numeric_dtype(y):
        target_encoder = LabelEncoder()
        y = pd.Series(target_encoder.fit_transform(y.astype(str)), index=y.index)

    return X, y, drop_cols, encoders, target_encoder


def run_regression_models(X, y):
    """Train/test split, fit candidate regression models, score on held-out test set."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": RidgeCV(alphas=np.logspace(-3, 3, 20)),
        "Lasso Regression": LassoCV(alphas=np.logspace(-3, 3, 20), max_iter=5000, random_state=42),
        "ElasticNet Regression": ElasticNetCV(alphas=np.logspace(-3, 3, 20), l1_ratio=[.1, .5, .7, .9, .95, 1], max_iter=5000, random_state=42),
        "KNN Regressor": KNeighborsRegressor(n_neighbors=min(5, max(2, len(X)//10))),
        "Decision Tree Regressor": DecisionTreeRegressor(random_state=42, max_depth=8),
        "Random Forest Regressor": RandomForestRegressor(random_state=42, n_estimators=200, max_depth=10),
        "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=42, n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)

        rmse = mean_squared_error(y_test, preds) ** 0.5
        mae = mean_absolute_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        # 5-fold cross validation on RMSE for stability check
        cv_scores = cross_val_score(model, scaler.fit_transform(X), y, cv=5, scoring="neg_root_mean_squared_error")
        cv_rmse_mean = -cv_scores.mean()
        cv_rmse_std = cv_scores.std()

        results[name] = {
            "model": model,
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2,
            "CV_RMSE_mean": cv_rmse_mean,
            "CV_RMSE_std": cv_rmse_std,
        }

    best_name = min(results, key=lambda k: results[k]["RMSE"])
    return results, best_name, (X_test, y_test)


def run_classification_models(X, y):
    """Train/test split, fit candidate classification models, score on held-out test set."""
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

    # Stratified split requires every class to have at least 2 members (so both
    # train and test can get at least one). Fall back to a plain (non-stratified)
    # split if that's not possible -- this matters for small or imbalanced datasets.
    class_counts = y.value_counts()
    can_stratify = y.nunique() > 1 and (class_counts.min() >= 2)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y if can_stratify else None
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    n_neighbors = min(5, max(2, len(X) // 10))
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "KNN Classifier": KNeighborsClassifier(n_neighbors=n_neighbors),
        "Naive Bayes": GaussianNB(),
        "Decision Tree Classifier": DecisionTreeClassifier(random_state=42, max_depth=8),
        "Random Forest Classifier": RandomForestClassifier(random_state=42, n_estimators=200, max_depth=10),
        "Gradient Boosting Classifier": GradientBoostingClassifier(random_state=42, n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)

        avg_method = "binary" if y.nunique() == 2 else "weighted"
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, average=avg_method, zero_division=0)
        rec = recall_score(y_test, preds, average=avg_method, zero_division=0)
        f1 = f1_score(y_test, preds, average=avg_method, zero_division=0)
        cm = confusion_matrix(y_test, preds)

        try:
            cv_folds = max(2, min(5, class_counts.min()))
            cv_scores = cross_val_score(model, scaler.fit_transform(X), y, cv=cv_folds, scoring="accuracy")
            cv_mean, cv_std = cv_scores.mean(), cv_scores.std()
        except ValueError:
            # Extremely small/imbalanced classes (e.g. a class with only 1 member)
            # can make even 2-fold stratified CV impossible. Skip CV rather than crash.
            cv_mean, cv_std = None, None

        results[name] = {
            "model": model,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1": f1,
            "ConfusionMatrix": cm,
            "CV_Accuracy_mean": cv_mean,
            "CV_Accuracy_std": cv_std,
        }

    best_name = max(results, key=lambda k: results[k]["F1"])
    return results, best_name, (X_test, y_test)


def run_pipeline(df: pd.DataFrame, target_col: str, profile: dict):
    """Full Step 2 + Step 3 pipeline: prep features, detect problem type, run models."""
    X, y, dropped_cols, encoders, target_encoder = prepare_features(df, target_col, profile)
    problem_type = detect_problem_type(df[target_col])

    if X.shape[1] == 0:
        raise ValueError("No usable feature columns remain after dropping text/datetime columns.")

    if problem_type == "regression":
        results, best_name, test_split = run_regression_models(X, y)
    else:
        results, best_name, test_split = run_classification_models(X, y)

    return {
        "problem_type": problem_type,
        "dropped_cols": dropped_cols,
        "results": results,
        "best_model": best_name,
        "feature_cols": list(X.columns),
    }
