from pathlib import Path
import pandas as pd


def find_csv_files(data_directory: str) -> list[Path]:
    """
    Find all CSV files inside the raw dataset directory.

    Parameters
    ----------
    data_directory : str
        Path to the directory containing CICIoT2023 CSV files.

    Returns
    -------
    list[Path]
        Sorted list of CSV file paths.
    """
    directory = Path(data_directory)

    if not directory.exists():
        raise FileNotFoundError(
            f"Dataset directory does not exist: {directory}"
        )

    csv_files = sorted(directory.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files were found in: {directory}"
        )

    return csv_files


def load_ciciot2023(
    data_directory: str = "data/raw",
    max_files: int | None = None,
    rows_per_file: int | None = None
) -> pd.DataFrame:
    """
    Load and combine CICIoT2023 CSV files.

    Parameters
    ----------
    data_directory : str
        Directory containing the CSV files.
    max_files : int or None
        Maximum number of files to load.
    rows_per_file : int or None
        Maximum number of rows loaded from each file.

    Returns
    -------
    pd.DataFrame
        Combined CICIoT2023 dataset.
    """
    csv_files = find_csv_files(data_directory)

    if max_files is not None:
        csv_files = csv_files[:max_files]

    dataframes = []

    for file_path in csv_files:
        print(f"Loading: {file_path.name}")

        dataframe = pd.read_csv(
            file_path,
            nrows=rows_per_file,
            low_memory=False
        )

        dataframes.append(dataframe)

    combined_data = pd.concat(
        dataframes,
        ignore_index=True
    )

    print(f"\nFiles loaded: {len(csv_files)}")
    print(f"Dataset shape: {combined_data.shape}")

    return combined_data


if __name__ == "__main__":
    dataset = load_ciciot2023(
        data_directory="data/raw",
        max_files=2,
        rows_per_file=10000
    )

    print("\nColumns:")
    print(dataset.columns.tolist())

    if "label" in dataset.columns:
        print("\nLabel distribution:")
        print(dataset["label"].value_counts())
