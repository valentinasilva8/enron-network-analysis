from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


FEATURE_COLS = [
    "in_degree",
    "out_degree",
    "weighted_degree",
    "clustering_coefficient",
    "closeness_centrality",
]

# Community size is derived at runtime if "community" column is present.
COMMUNITY_SIZE_COL = "community_size"


def _prepare_data(nodes_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Build feature matrix X and binary target y from nodes.csv data."""
    df = nodes_df.copy()

    # Binary target: top 20% PageRank OR top 20% betweenness
    pr_threshold = df["pagerank"].quantile(0.80)
    bw_threshold = df["betweenness_centrality"].quantile(0.80)
    df["high_influence"] = (
        (df["pagerank"] >= pr_threshold) | (df["betweenness_centrality"] >= bw_threshold)
    ).astype(int)

    # Derive community_size feature if community column exists
    feature_cols = list(FEATURE_COLS)
    if "community" in df.columns:
        comm_sizes = df["community"].value_counts().to_dict()
        df[COMMUNITY_SIZE_COL] = df["community"].map(comm_sizes)
        feature_cols.append(COMMUNITY_SIZE_COL)

    df[feature_cols] = df[feature_cols].fillna(0.0)
    return df, df["high_influence"], feature_cols


def run_neural_network_classification(
    nodes_csv_path: Path,
    output_dir: Path,
    figures_dir: Path,
) -> dict:
    """Train MLP and baseline RF; save metrics and confusion-matrix plot."""
    nodes_df = pd.read_csv(nodes_csv_path)
    df, y, feature_cols = _prepare_data(nodes_df)
    X = df[feature_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y,
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    # --- MLP classifier ---
    mlp = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.15,
    )
    mlp.fit(X_train_sc, y_train)
    y_pred_mlp = mlp.predict(X_test_sc)

    # --- Baseline: Random Forest ---
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_sc, y_train)
    y_pred_rf = rf.predict(X_test_sc)

    def _metrics(y_true, y_pred, model_name: str) -> dict:
        return {
            "model": model_name,
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        }

    mlp_metrics = _metrics(y_test, y_pred_mlp, "MLPClassifier")
    rf_metrics = _metrics(y_test, y_pred_rf, "RandomForest_Baseline")

    results = {
        "label_definition": "high_influence = 1 if PageRank >= top 20% OR betweenness >= top 20%",
        "features_used": feature_cols,
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "class_distribution_train": {
            "high_influence_1": int(y_train.sum()),
            "not_high_influence_0": int(len(y_train) - y_train.sum()),
        },
        "mlp": mlp_metrics,
        "baseline_rf": rf_metrics,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "neural_network_metrics.json").write_text(json.dumps(results, indent=2))
    print("=== NEURAL NETWORK CLASSIFICATION RESULTS ===")
    print(json.dumps(results, indent=2))

    # --- Confusion-matrix plot (MLP) ---
    figures_dir.mkdir(parents=True, exist_ok=True)
    cm = np.array(mlp_metrics["confusion_matrix"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
                xticklabels=["Not High", "High"], yticklabels=["Not High", "High"])
    axes[0].set_title("MLP Classifier")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")

    cm_rf = np.array(rf_metrics["confusion_matrix"])
    sns.heatmap(cm_rf, annot=True, fmt="d", cmap="Oranges", ax=axes[1],
                xticklabels=["Not High", "High"], yticklabels=["Not High", "High"])
    axes[1].set_title("Random Forest Baseline")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")

    plt.suptitle("Confusion Matrices: High-Influence Node Prediction")
    plt.tight_layout()
    plt.savefig(figures_dir / "result_neural_network_confusion_matrix.png", dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Confusion matrix saved to {figures_dir / 'result_neural_network_confusion_matrix.png'}")
    return results
