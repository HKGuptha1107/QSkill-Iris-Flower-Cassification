

import pandas as pd
from pathlib import Path
from sklearn.datasets import load_iris


def process_iris(output_path=None):
    """Process the Iris dataset and save it as a CSV file."""
    iris = load_iris()
    processed_data = pd.DataFrame(iris.data, columns=iris.feature_names)
    processed_data["target"] = iris.target
    processed_data["species"] = [iris.target_names[target] for target in iris.target]

    if output_path is None:
        output_path = Path(__file__).resolve().parents[1] / "data" / "iris_processed.csv"
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed_data.to_csv(output_path, index=False)
    return processed_data, output_path


def display_data_summary(data):
    """Display the main structure and statistics of the processed data."""
    print("\nFirst five rows:")
    print(data.head())

    print("\nDataset shape:")
    print(f"Rows: {data.shape[0]}, Columns: {data.shape[1]}")

    print("\nColumn data types:")
    print(data.dtypes)

    print("\nMissing values per column:")
    print(data.isnull().sum())

    print("\nStatistical summary:")
    print(data.describe())

    print("\nSpecies counts:")
    print(data["species"].value_counts())


if __name__ == "__main__":
    data, output_path = process_iris()
    print(f"Saved {len(data)} rows to {output_path}")
    display_data_summary(data)