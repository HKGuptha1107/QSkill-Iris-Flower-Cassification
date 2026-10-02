import joblib
import pandas as pd 
from pathlib import Path

MODEL_DIR = Path("models")

FEATURES = [
    "sepal length (cm)",
    "sepal width (cm)",
    "petal length (cm)",
    "petal width (cm)",
]

new_data = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]],columns=FEATURES)

model_files = [i for i in MODEL_DIR.iterdir() if i.is_file() and i.suffix == ".pkl"]

for file_path in model_files:
    model_name = file_path.stem
    model = joblib.load(file_path)
    prediction = model.predict(new_data)

    print(f"{model_name} : {prediction[0]}")