# Loads the train/test split produced by prep.py (downloaded here as the
# workflow artifact from the previous job), builds a preprocessing +
# XGBoost pipeline, tunes it with GridSearchCV, logs every tuned parameter
# and the evaluation metrics to MLflow, evaluates the best model on the
# held-out test set, and saves the full pipeline so the workflow can commit
# it into tourism_project/deployment/ for the Streamlit app to load.

import pandas as pd
import mlflow
import xgboost as xgb
import joblib
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

NUMERIC_FEATURES = [
    "Age", "CityTier", "DurationOfPitch", "NumberOfPersonVisiting",
    "NumberOfFollowups", "PreferredPropertyStar", "NumberOfTrips",
    "Passport", "PitchSatisfactionScore", "OwnCar",
    "NumberOfChildrenVisiting", "MonthlyIncome",
]

CATEGORICAL_FEATURES = [
    "TypeofContact", "Occupation", "Gender", "ProductPitched",
    "MaritalStatus", "Designation",
]

MODEL_PATH = "tourism_project/deployment/model.joblib"
RANDOM_STATE = 42

# The artifact from the data-prep job lands in the current working
# directory, so the split files are read from there, not from
# tourism_project/data/.
X_train = pd.read_csv("Xtrain.csv")
X_test = pd.read_csv("Xtest.csv")
y_train = pd.read_csv("ytrain.csv").iloc[:, 0]
y_test = pd.read_csv("ytest.csv").iloc[:, 0]

# Some columns have missing values; fill them before the pipeline runs so
# the scaler/encoder/model never see a NaN.
for col in NUMERIC_FEATURES:
    X_train[col] = X_train[col].fillna(X_train[col].median())
    X_test[col] = X_test[col].fillna(X_train[col].median())
for col in CATEGORICAL_FEATURES:
    X_train[col] = X_train[col].fillna(X_train[col].mode()[0])
    X_test[col] = X_test[col].fillna(X_train[col].mode()[0])

preprocessor = make_column_transformer(
    (StandardScaler(), NUMERIC_FEATURES),
    (OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
)

pipeline = make_pipeline(
    preprocessor,
    xgb.XGBClassifier(
        random_state=RANDOM_STATE,
        eval_metric="logloss",
    ),
)

PARAM_GRID = {
    "xgbclassifier__n_estimators": [100, 200],
    "xgbclassifier__max_depth": [3, 5],
    "xgbclassifier__learning_rate": [0.05, 0.1],
}

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("tourism-wellness-package")

with mlflow.start_run():
    search = GridSearchCV(
        pipeline,
        param_grid=PARAM_GRID,
        scoring="roc_auc",
        cv=5,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)

    best_model = search.best_estimator_

    # Log every tuned hyperparameter
    for param, value in search.best_params_.items():
        mlflow.log_param(param, value)
    mlflow.log_metric("best_cv_roc_auc", search.best_score_)

    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }
    for name, value in metrics.items():
        mlflow.log_metric(name, value)

    print(f"Best params: {search.best_params_}")
    print(f"Best CV ROC-AUC: {search.best_score_:.4f}")
    print("\nTest set performance:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred))

    joblib.dump(best_model, MODEL_PATH)
    mlflow.log_artifact(MODEL_PATH)
    print(f"\nSaved best model pipeline -> {MODEL_PATH}")
