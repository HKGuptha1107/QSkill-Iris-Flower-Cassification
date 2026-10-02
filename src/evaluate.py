from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "iris_processed.csv"
MODEL_DIR = BASE_DIR / "models"

FEATURE_COLUMNS = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
]

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "K-Nearest Neighbors": "KNN.pkl",
    "Decision Tree": "decision_tree.pkl",
}


def load_test_data() -> tuple[pd.DataFrame, pd.Series]:
    """Load the same reproducible test split used when training the models."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    data = pd.read_csv(DATA_PATH)
    missing_columns = set(FEATURE_COLUMNS + ["species"]) - set(data.columns)
    if missing_columns:
        columns = ", ".join(sorted(missing_columns))
        raise ValueError(f"Dataset is missing required columns: {columns}")

    features = data[FEATURE_COLUMNS]
    labels = data["species"]
    _, test_features, _, test_labels = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )
    return test_features, test_labels


def evaluate_model(
    model_name: str,
    model_file: str,
    test_features: pd.DataFrame,
    test_labels: pd.Series,
) -> dict[str, float]:
    """Evaluate one saved model and print its detailed classification results."""
    model_path = MODEL_DIR / model_file
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = joblib.load(model_path)
    predictions = model.predict(test_features)

    scores = {
        "accuracy": accuracy_score(test_labels, predictions),
        "precision": precision_score(
            test_labels, predictions, average="weighted", zero_division=0
        ),
        "recall": recall_score(
            test_labels, predictions, average="weighted", zero_division=0
        ),
        "f1_score": f1_score(
            test_labels, predictions, average="weighted", zero_division=0
        ),
    }

    print(f"\n{'=' * 60}")
    print(model_name)
    print(f"{'=' * 60}")
    print(classification_report(test_labels, predictions, zero_division=0))
    print("Confusion matrix:")
    print(confusion_matrix(test_labels, predictions))

    return scores


def main() -> None:
    """Evaluate all saved models and print a ranked comparison."""
    test_features, test_labels = load_test_data()
    results = {}

    for model_name, model_file in MODEL_FILES.items():
        results[model_name] = evaluate_model(
            model_name,
            model_file,
            test_features,
            test_labels,
        )

    ranking = sorted(
        results.items(),
        key=lambda item: item[1]["f1_score"],
        reverse=True,
    )

    print(f"\n{'=' * 78}")
    print("MODEL COMPARISON")
    print(f"{'=' * 78}")
    print(
        f"{'Model':<24} {'Accuracy':>10} {'Precision':>10} "
        f"{'Recall':>10} {'F1 score':>10}"
    )
    print("-" * 78)
    for model_name, scores in ranking:
        print(
            f"{model_name:<24} {scores['accuracy']:>10.3f} "
            f"{scores['precision']:>10.3f} {scores['recall']:>10.3f} "
            f"{scores['f1_score']:>10.3f}"
        )

    print(f"\nBest model by weighted F1 score: {ranking[0][0]}")


if __name__ == "__main__":
    main()