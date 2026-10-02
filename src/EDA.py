
import matplotlib.pyplot as plt 
import seaborn as sns
import pandas as pd
import numpy as np

class EdaAnalysis:
    def __init__(self,path):
        # Load the processed dataset and keep only numeric feature columns.
        self.path = path
        self.data = pd.read_csv(self.path)
        self.features = self.data.select_dtypes(
        include=np.number
        ).drop(
            columns=["target", "Species"],
            errors="ignore"
        )

    def feature_dist(self):
        # Create histograms to show the distribution of each feature.
        self.features.hist(figsize=(10,10))
        plt.title("Distribution of Iris Features")
        plt.tight_layout()
        plt.savefig("images/feature_dist.png")
        plt.show()

    def compare_features(self):
        # Use box plots to compare feature ranges and spot outliers.
        self.features.boxplot(figsize=(10,10))
        plt.title("Box Plot of Iris Features")
        plt.tight_layout()
        plt.savefig("images/feature_compare.png")
        plt.show()

    def relation_btw_features(self):
        # Show how petal length and width change for each iris species.
        sns.scatterplot(data = self.data,
                    x = "petal length (cm)",
                    y ="petal width (cm)",
                    hue="species")
        plt.title("Relation between features")
        plt.tight_layout()
        plt.savefig("images/relation_features.png")
        plt.show()

    def corrleation_data(self):
        # Calculate and display the correlation between numeric columns.
        corr = self.data.select_dtypes(include=np.number).corr()

        sns.heatmap(corr,cmap="coolwarm",annot=True,vmin=-1,vmax=1)
        plt.title("Corrleation Data")
        plt.tight_layout()
        plt.savefig("images/corrleation_data.png")
        plt.show()

    def main(self):
        # Run all exploratory data analysis plots.
        self.feature_dist()
        self.compare_features()
        self.relation_btw_features()
        self.corrleation_data()

if __name__ == "__main__":
    path = "data/iris_processed.csv"
    obj = EdaAnalysis(path)
    obj.main()
