from data.load_data import load_ciciot2023
from features.preprocess import (
    clean_dataset,
    create_binary_target,
    prepare_features,
    split_and_scale
)


def main():
    """
    Test the complete CICIoT2023 loading and preprocessing pipeline.
    """

    print("=" * 60)
    print("CICIoT2023 data pipeline test")
    print("=" * 60)

    # Load a small sample while testing
    data = load_ciciot2023(
        data_directory="data/raw",
        max_files=2,
        rows_per_file=10000
    )

    print("\nRaw dataset shape:")
    print(data.shape)

    # Clean the dataset
    cleaned_data = clean_dataset(data)

    print("\nShape after cleaning:")
    print(cleaned_data.shape)

    # Create binary labels
    labelled_data = create_binary_target(cleaned_data)

    print("\nBinary class distribution:")
    print(labelled_data["binary_label"].value_counts())

    print("\nBinary class percentages:")
    print(
        labelled_data["binary_label"]
        .value_counts(normalize=True)
        .mul(100)
        .round(2)
    )

    # Prepare features and target
    features, target = prepare_features(labelled_data)

    print("\nFeature matrix shape:")
    print(features.shape)

    print("\nTarget shape:")
    print(target.shape)

    print("\nFeature columns:")
    print(features.columns.tolist())

    # Split and scale
    (
        X_train,
        X_test,
        y_train,
        y_test,
        X_train_scaled,
        X_test_scaled,
        scaler
    ) = split_and_scale(
        features=features,
        target=target,
        test_size=0.20,
        random_state=42
    )

    print("\nTraining data shape:")
    print(X_train.shape)

    print("\nTesting data shape:")
    print(X_test.shape)

    print("\nScaled training data shape:")
    print(X_train_scaled.shape)

    print("\nScaled testing data shape:")
    print(X_test_scaled.shape)

    print("\nTraining class distribution:")
    print(y_train.value_counts(normalize=True).round(4))

    print("\nTesting class distribution:")
    print(y_test.value_counts(normalize=True).round(4))

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()
