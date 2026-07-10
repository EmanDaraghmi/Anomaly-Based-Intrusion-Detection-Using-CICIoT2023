import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler


BENIGN_LABELS = {
    "BenignTraffic",
    "Benign",
    "Normal",
    "normal"
}


def clean_dataset(data: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the CICIoT2023 dataset.

    Parameters
    ----------
    data : pd.DataFrame
        Raw CICIoT2023 dataset.

    Returns
    -------
    pd.DataFrame
        Cleaned dataset.
    """
    cleaned_data = data.copy()

    cleaned_data.columns = cleaned_data.columns.str.strip()

    if "label" not in cleaned_data.columns:
        raise KeyError(
            "The dataset must contain a 'label' column."
        )

    cleaned_data.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True
    )

    cleaned_data.drop_duplicates(inplace=True)
    cleaned_data.dropna(subset=["label"], inplace=True)

    return cleaned_data


def create_binary_target(data: pd.DataFrame) -> pd.DataFrame:
    """
    Create a binary target column.

    0 = benign traffic
    1 = attack traffic
    """
    labelled_data = data.copy()

    labelled_data["binary_label"] = (
        ~labelled_data["label"]
        .astype(str)
        .str.strip()
        .isin(BENIGN_LABELS)
    ).astype(np.int8)

    return labelled_data


def prepare_features(
    data: pd.DataFrame
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separate numeric features and binary target.
    """
    if "binary_label" not in data.columns:
        raise KeyError(
            "The dataset must contain 'binary_label'."
        )

    features = data.drop(
        columns=["label", "binary_label"]
    )

    target = data["binary_label"].astype(int)

    features = features.apply(
        pd.to_numeric,
        errors="coerce"
    )

    features.dropna(
        axis=1,
        how="all",
        inplace=True
    )

    features.fillna(
        features.median(numeric_only=True),
        inplace=True
    )

    constant_columns = [
        column
        for column in features.columns
        if features[column].nunique(dropna=False) <= 1
    ]

    features.drop(
        columns=constant_columns,
        inplace=True
    )

    return features, target


def split_and_scale(
    features: pd.DataFrame,
    target: pd.Series,
    test_size: float = 0.20,
    random_state: int = 42
):
    """
    Split the dataset before fitting the scaler.

    This avoids data leakage.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
        stratify=target
    )

    scaler = RobustScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        X_train_scaled,
        X_test_scaled,
        scaler
    )
