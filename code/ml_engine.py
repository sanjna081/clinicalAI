import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report,
    confusion_matrix, f1_score
)
from sklearn.impute import SimpleImputer
import xgboost as xgb
import hdbscan
import warnings
warnings.filterwarnings("ignore")


# preprocessing

def preprocess(df, target_col=None, scale=False):
    """
    Full preprocessing pipeline:
    1. Drop high-nullity columns (>40% missing)
    2. Strip ID columns (unique ratio > 90%)
    3. Separate and encode target
    4. Impute missing values (median for numeric, mode for categorical)
    5. Encode categoricals (one-hot <= 5 unique, label encode > 5)
    6. Optional Z-score scaling
    """
    df = df.copy()

    # Step 1: Drop high-nullity columns
    null_threshold = 0.4 * len(df)
    df = df.dropna(thresh=null_threshold, axis=1)

    # Step 2: Strip ID columns
    id_cols = [
        col for col in df.columns
        if col != target_col and df[col].nunique() / len(df) > 0.9
    ]
    df = df.drop(columns=id_cols)

    # Step 3: Separate target
    y = None
    if target_col and target_col in df.columns:
        y = df[target_col].copy()
        df = df.drop(columns=[target_col])

    # Step 4: Separate column types
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # Step 5: Impute missing values
    if numeric_cols:
        num_imputer = SimpleImputer(strategy="median")
        df[numeric_cols] = num_imputer.fit_transform(df[numeric_cols])

    for col in categorical_cols:
        if df[col].isnull().any():
            mode_val = df[col].mode()
            fill_val = mode_val[0] if not mode_val.empty else "Unknown"
            df[col] = df[col].fillna(fill_val)

    # Step 6: Encode categoricals
    le = LabelEncoder()
    ohe_cols = [c for c in categorical_cols if df[c].nunique() <= 5]
    label_cols = [c for c in categorical_cols if df[c].nunique() > 5]

    if ohe_cols:
        df = pd.get_dummies(df, columns=ohe_cols, drop_first=True)

    for col in label_cols:
        df[col] = le.fit_transform(df[col].astype(str))

    # Step 7: Encode target
    if y is not None:
        y = le.fit_transform(y.astype(str))

    # Step 8: Keep only numeric columns
    df = df.select_dtypes(include=[np.number, bool]).astype(float)

    # Step 9: Optional Z-score scaling
    if scale:
        scaler = StandardScaler()
        df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

    return df, y


# shared metrics

def _classification_metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)

    try:
        if len(np.unique(y_test)) == 2:
            auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        else:
            auc = roc_auc_score(
                y_test,
                model.predict_proba(X_test),
                multi_class="ovr",
                average="weighted"
            )
        auc = round(auc, 4)
    except Exception:
        auc = "N/A"

    return {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "auc": auc,
        "f1": round(f1_score(y_test, y_pred, average="weighted"), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred),
    }


# Classification Models

def run_logistic_regression(df, target_col):
    X, y = preprocess(df, target_col, scale=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    feature_importances = dict(
        sorted(zip(X.columns, abs(model.coef_[0])),
               key=lambda x: x[1], reverse=True)[:10]
    )
    metrics = _classification_metrics(model, X_test, y_test)
    return {"model": "Logistic Regression", "feature_importances": feature_importances, **metrics}


def run_random_forest(df, target_col):
    X, y = preprocess(df, target_col, scale=False)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    feature_importances = dict(
        sorted(zip(X.columns, model.feature_importances_),
               key=lambda x: x[1], reverse=True)[:10]
    )
    metrics = _classification_metrics(model, X_test, y_test)
    return {"model": "Random Forest", "feature_importances": feature_importances, **metrics}


def run_xgboost(df, target_col):
    X, y = preprocess(df, target_col, scale=False)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = xgb.XGBClassifier(
        n_estimators=100,
        random_state=42,
        eval_metric="mlogloss",
        verbosity=0
    )
    model.fit(X_train, y_train)

    feature_importances = dict(
        sorted(zip(X.columns, model.feature_importances_),
               key=lambda x: x[1], reverse=True)[:10]
    )
    metrics = _classification_metrics(model, X_test, y_test)
    return {"model": "XGBoost", "feature_importances": feature_importances, **metrics}


# Clustering

def run_kmeans(df, target_col=None, n_clusters=4):
    X_scaled, _ = preprocess(df, target_col, scale=True)

    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)

    # PCA for visualization — reduces all features to 2 dimensions
    pca = PCA(n_components=2, random_state=42)
    X_pca = pd.DataFrame(
        pca.fit_transform(X_scaled),
        columns=["PCA Component 1", "PCA Component 2"]
    )

    df_result = df.copy()
    df_result["cluster"] = labels
    numeric_cols = df_result.select_dtypes(include=[np.number]).columns.tolist()
    cluster_profiles = df_result.groupby("cluster")[numeric_cols].mean().round(2)

    return {
        "model": "KMeans",
        "n_clusters": n_clusters,
        "cluster_labels": labels.tolist(),
        "cluster_sizes": df_result["cluster"].value_counts().sort_index().to_dict(),
        "cluster_profiles": cluster_profiles.to_dict(),
        "X_pca": X_pca,
    }


def run_hdbscan(df, target_col=None):
    X_scaled, _ = preprocess(df, target_col, scale=True)

    model = hdbscan.HDBSCAN(min_cluster_size=50)
    labels = model.fit_predict(X_scaled)

    # PCA for visualization
    pca = PCA(n_components=2, random_state=42)
    X_pca = pd.DataFrame(
        pca.fit_transform(X_scaled),
        columns=["PCA Component 1", "PCA Component 2"]
    )

    df_result = df.copy()
    df_result["cluster"] = labels
    numeric_cols = df_result.select_dtypes(include=[np.number]).columns.tolist()
    cluster_profiles = df_result.groupby("cluster")[numeric_cols].mean().round(2)

    return {
        "model": "HDBSCAN",
        "n_clusters": len(set(labels)) - (1 if -1 in labels else 0),
        "cluster_labels": labels.tolist(),
        "cluster_sizes": df_result["cluster"].value_counts().sort_index().to_dict(),
        "cluster_profiles": cluster_profiles.to_dict(),
        "X_pca": X_pca,
    }


# Router

def run_model(model_name, df, target_col=None, n_clusters=4):
    model_name = model_name.lower().strip()

    routes = {
        "logistic_regression": lambda: run_logistic_regression(df, target_col),
        "random_forest":       lambda: run_random_forest(df, target_col),
        "xgboost":             lambda: run_xgboost(df, target_col),
        "kmeans":              lambda: run_kmeans(df, target_col, n_clusters),
        "hdbscan":             lambda: run_hdbscan(df, target_col),
    }

    if model_name not in routes:
        return run_random_forest(df, target_col)

    return routes[model_name]()