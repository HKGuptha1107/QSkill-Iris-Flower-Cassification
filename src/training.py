import pandas as pd 
import joblib as jb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

class TrainModels:
    def __init__(self,path):
        self.data = pd.read_csv(path)
        self.model_path = "models"

    def splitData(self):
        X = self.data.drop(columns=["species", "target"])
        y = self.data["species"]

        X_train,X_test,y_train,y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        return X_train,X_test,y_train,y_test

    def save_model(self,model,filename):
        path = f"{self.model_path}/{filename}"
        jb.dump(model,path)
        print("Model Saved : ",path)

    
    def logisticRegression(self):
        X_train,X_test,y_train,y_test = self.splitData()

        model = Pipeline([("scaler",StandardScaler()),("classifier",LogisticRegression(max_iter=200))])

        model.fit(X_train,y_train)

        y_pred = model.predict(X_test)

        print("Accuracy Score : ",accuracy_score(y_test,y_pred))
        print("\nClassification Report : ",classification_report(y_test,y_pred))
        print("\nConfusion Matrix : ",confusion_matrix(y_test,y_pred))

        self.save_model(model,"logistic_regression.pkl")

    def KNearestNeighborhood(self):
        X_train,X_test,y_train,y_test = self.splitData()

        model = Pipeline([("scaler",StandardScaler()),("classifier",KNeighborsClassifier(n_neighbors=5))])

        model.fit(X_train,y_train)

        y_pred = model.predict(X_test)
        
        print("Accuracy Score : ",accuracy_score(y_test,y_pred))
        print("\nClassification Report : ",classification_report(y_test,y_pred))
        print("\nConfusion Matrix : ",confusion_matrix(y_test,y_pred))

        self.save_model(model,"KNN.pkl")

    def DecisionTree(self):
        X_train,X_test,y_train,y_test = self.splitData()

        model = DecisionTreeClassifier(random_state=42)

        model.fit(X_train,y_train)

        y_pred = model.predict(X_test)

        print("Accuracy Score : ",accuracy_score(y_test,y_pred))
        print("\nClassification Report : ",classification_report(y_test,y_pred))
        print("\nConfusion Matrix : ",confusion_matrix(y_test,y_pred))

        self.save_model(model,"decision_tree.pkl") 

    def main(self):
        print("Now Start the Train Models : ")
        print("\n\nLogistic Regression Model Trainig Start ...")
        self.logisticRegression()
        print("\n\nCompleted Logistic Regression .. ")
        print("\n\nKNN Training Model Start ...")
        self.KNearestNeighborhood()
        print("\n\nComplete KNN Training model..")
        print("\n\nDecisionTree Model Training Start ..")
        self.DecisionTree()
        print("\n\nDecision Tree completed ...")

    def test(self):
        print("Shape:", self.data.shape)
        print("Columns:", self.data.columns.tolist())
        print("Missing values:\n", self.data.isnull().sum())
        print("Duplicate rows:", self.data.duplicated().sum())
        print("Class counts:\n", self.data["species"].value_counts())

if __name__ == "__main__":
    PATH = "data/iris_processed.csv"
    obj = TrainModels(PATH)
    obj.main()
    #obj.test()


"""
OutPut :: 
Now Start the Train Models : 


Logistic Regression Model Trainig Start ...
Accuracy Score :  0.9333333333333333

Classification Report :                precision    recall  f1-score   support

      setosa       1.00      1.00      1.00        10
  versicolor       0.90      0.90      0.90        10
   virginica       0.90      0.90      0.90        10

    accuracy                           0.93        30
   macro avg       0.93      0.93      0.93        30
weighted avg       0.93      0.93      0.93        30


Confusion Matrix :  [[10  0  0]
 [ 0  9  1]
 [ 0  1  9]]
Model Saved :  models/logistic_regression.pkl


Completed Logistic Regression .. 


KNN Training Model Start ...
Accuracy Score :  0.9333333333333333

Classification Report :                precision    recall  f1-score   support

      setosa       1.00      1.00      1.00        10
  versicolor       0.83      1.00      0.91        10
   virginica       1.00      0.80      0.89        10

    accuracy                           0.93        30
   macro avg       0.94      0.93      0.93        30
weighted avg       0.94      0.93      0.93        30


Confusion Matrix :  [[10  0  0]
 [ 0 10  0]
 [ 0  2  8]]
Model Saved :  models/KNN.pkl


Complete KNN Training model..


DecisionTree Model Training Start ..
Accuracy Score :  0.9333333333333333

Classification Report :                precision    recall  f1-score   support

      setosa       1.00      1.00      1.00        10
  versicolor       0.90      0.90      0.90        10
   virginica       0.90      0.90      0.90        10

    accuracy                           0.93        30
   macro avg       0.93      0.93      0.93        30
weighted avg       0.93      0.93      0.93        30


Confusion Matrix :  [[10  0  0]
 [ 0  9  1]
 [ 0  1  9]]
Model Saved :  models/decision_tree.pkl


Decision Tree completed ...
"""