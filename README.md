## Iris Flower Prediction API

The project includes a browser interface for the trained Logistic Regression model.

### Run the app

From the project root:

```powershell
C:\Users\HP\AppData\Local\Programs\Python\Python314\python.exe -m uvicorn src.app:app --reload
```

Open <http://127.0.0.1:8000/> to use the prediction form. The form sends the four measurements to `POST /predict` and displays the returned species. Interactive API documentation is available at <http://127.0.0.1:8000/docs>.

### Evaluate the trained models

Run the evaluation script from the project root:

```powershell
C:\Users\HP\AppData\Local\Programs\Python\Python314\python.exe src\evaluate.py
```

It evaluates Logistic Regression, K-Nearest Neighbors, and Decision Tree on the same reproducible test split and reports accuracy, precision, recall, F1 score, classification reports, and confusion matrices.