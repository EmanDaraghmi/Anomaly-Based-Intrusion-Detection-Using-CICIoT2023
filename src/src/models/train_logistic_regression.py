import json
import time
from pathlib import Path

import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

from data.load_data import load_ciciot2023
from features.preprocess import (
    clean_dataset,
    create_binary_target,
    prepare_features,
    split_and_scale
)


RANDOM_STATE = 42


def main():
    print("=" * 60)
    print("Logistic Regression Baseline")
    print("=" * 60)

    # Load a manageable sample first
    data = load_ciciot2023(
        data_directory="data/raw",
        max_files=10,
        rows_per_file=50000
    )

    # Preprocessing
    data = clean_dataset(data)
    data = create_binary_target(data)

    print("\nClass distribution:")
    print(data["binary_label"].value_counts())

    features, target = prepare_features(data)

    (
        X_train,
        X_test,
        y_train,
        y_test,
        X_train_scaled,
        X_test_scaled,
        scaler
    ) = split_and_scale(
        features,
        target,
        test_size=0.20,
        random_state=RANDOM_STATE
    )

    # Model
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    print("\nTraining Logistic Regression...")

    start_time = time.time()
    model.fit(X_train_scaled, y_train)
    training_time = time.time() - start_time

    start_time = time.time()
    predictions = model.predict(X_test_scaled)
    prediction_time = time.time() - start_time

    probabilities = model.predict_proba(X_test_scaled)[:, 1]

    # Metrics
    results = {
        "model": "Logistic Regression",
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(
            y_test,
            predictions,
            zero_division=0
        ),
        "recall": recall_score(
            y_test,
            predictions,
            zero_division=0
        ),
        "f1_score": f1_score(
            y_test,
            predictions,
            zero_division=0
        ),
        "roc_auc": roc_auc_score(
            y_test,
            probabilities
        ),
        "training_time_seconds": training_time,
        "prediction_time_seconds": prediction_time
    }

    print("\nResults:")
    for metric_name, metric_value in results.items():
        print(f"{metric_name}: {metric_value}")

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=["Benign", "Attack"],
            zero_division=0
        )
    )

    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, predictions))

    # Create output directories
    metrics_directory = Path("results/metrics")
    models_directory = Path("results/models")

    metrics_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    models_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save metrics
    metrics_path = (
        metrics_directory /
        "logistic_regression_results.json"
    )

    with open(
        metrics_path,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            results,
            file,
            indent=4
        )

    # Save model and scaler
    joblib.dump(
        model,
        models_directory /
        "logistic_regression.joblib"
    )

    joblib.dump(
        scaler,
        models_directory /
        "robust_scaler.joblib"
    )

    print("\nSaved results to:")
    print(metrics_path)

    print("\nTraining completed successfully.")


if __name__ == "__main__":
    main()
