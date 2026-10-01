
import matplotlib.pyplot as plt 
import seaborn as sns
import pandas as pd
import numpy as np

class EdaAnalysis:
    def __init__(self,path):
        self.path = path
        self.data = pd.read_csv(self.path)

    def feature_dist(self):
        self.data.hist(figsize=(10,10))
        plt.title("Features Distribution")
        plt.savefig("images/feature_dist.png")
        plt.show()

    def compare_features(self):
        self.data.boxplot(figsize=(10,10))
        plt.title("Outliers vs Spread vs Median")
        plt.savefig("images/feature_compare.png")
        plt.show()

    def relation_btw_features(self):
        plt.scatter(self.data["petal length (cm)"],
    self.data["petal width (cm)"])
        plt.xlabel("Petal Length")
        plt.ylabel("Petal Width")
        plt.title("Relation between features")
        plt.savefig("images/relation_features.png")
        plt.show()

    def corrleation_data(self):
        corr = self.data.select_dtypes(include=np.number).corr()

        sns.heatmap(corr,cmap="coolwarm",annot=True)
        plt.title("Corrleation Data")
        plt.savefig("images/corrleation_data.png")
        plt.show()

    def main(self):
        self.feature_dist()
        self.compare_features()
        self.relation_btw_features()
        self.corrleation_data()

if __name__ == "__main__":
    path = "data/iris_processed.csv"
    obj = EdaAnalysis(path)
    obj.main()
