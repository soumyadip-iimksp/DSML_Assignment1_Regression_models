__author__ = "Soumyadip Majumder"
__version__ = "1.0.0"
__maintainer__ = "Soumyadip Majumder"
__status__ = "Test"
__date__ = "29 Jan 2023"

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
import pickle


class DataPrep:


    def pattern_visualizations(self, data:pd.DataFrame, PATH:str):
        if not os.path.exists(PATH):
            os.makedirs(PATH)

        for col in list(data.columns)[:-1]:
            sns.scatterplot(data=data, x=col, y="MEDV")
            plt.title(f"HOUSEPRICE: {col} vs MDEV")
            plt.xlabel(col)
            plt.ylabel("MDEV")
            plt.savefig(f"{PATH}/{col}_MDEV_scatter.png", bbox_inches = 'tight')
            print(f"Scatter Plot of {col} vs MDEV is saved")

            sns.displot(data=data, x=col, kind="kde")
            plt.title(f"Distribution of {col}")
            plt.xlabel(col)
            plt.savefig(f"{PATH}/{col}_distribution.png", bbox_inches = 'tight')
            print(f"Distribution Plot of {col}  is saved")

        sns.pairplot(data)
        plt.title(f"Pairplot")
        plt.savefig(f"{PATH}/Pairplot.png", bbox_inches = 'tight')
        print("Pairplot is saved")


    def log_log_data(self, data:pd.DataFrame):
        """
        features: log transformed
        target:   log transformed
        """
        data.replace(0, 0.0000001, inplace=True)
        self.data_out = data.apply(lambda x: np.log10(x))
        return self.data_out


    def log_lin_data(self, data:pd.DataFrame):
        """
        features: Linear
        target:   log transformed
        """
        self.data_out = data.copy()
        self.data_out.replace(0, 0.0000001, inplace=True)
        self.data_out["MEDV"] = np.log10(self.data_out["MEDV"].values)
        return self.data_out


    def lin_log_data(self, data:pd.DataFrame):
        """
        features: Log Transformed
        target:   Linear
        """
        data.replace(0, 0.0000001, inplace=True)
        self.data_out = data.apply(lambda x: np.log10(x))
        self.data_out["MEDV"] = data["MEDV"]
        return self.data_out


    def train_val_test_split(self, data:pd.DataFrame, split_ratios:list, dv_name:str):

        DV_PATH = "./models"

        self.X = data.drop(["MEDV"], axis=1)
        self.y = data["MEDV"].values

        self.train_X, self.extra_X, self.train_y, self.extra_y = train_test_split(self.X, self.y, 
                                                                                    test_size=(1-split_ratios[0]), 
                                                                                    shuffle=True, random_state=143)
        self.val_X, self.test_X, self.val_y, self.test_y = train_test_split(self.extra_X, self.extra_y, 
                                                                            test_size=(split_ratios[2]/sum(split_ratios[1:])), 
                                                                            shuffle=True, random_state=143)
        del self.extra_X, self.extra_y

        self.train_X = self.train_X.to_dict(orient="records")
        self.val_X = self.val_X.to_dict(orient="records")
        self.test_X = self.test_X.to_dict(orient="records")

        self.dv = DictVectorizer(sparse=False)
        self.dv.fit(self.train_X)
        self.train_X = self.dv.transform(self.train_X)
        self.val_X = self.dv.transform(self.val_X)
        self.test_X = self.dv.transform(self.test_X)

        if not os.path.exists(DV_PATH):
            os.makedirs(DV_PATH)
        with open(f"{DV_PATH}/dv_{dv_name}.bin", "wb") as f_out:
            pickle.dump(self.dv, f_out)
        return self.train_X, self.train_y, self.val_X, self.val_y, self.test_X, self.test_y
        




if __name__ == "__main__":
    df = pd.read_csv("Housing_Price_data.csv")
    dp = DataPrep()
   
    df_lin_log = dp.lin_log_data(df)
    df_log_lin = dp.log_lin_data(df)
    df_log_log = dp.log_log_data(df)
    print("original")
    print(df.head(2))
    print("lin log")
    print(df_lin_log.head(2))
    print("log lin")
    print(df_log_lin.head(2))
    print("log log")
    print(df_log_log.head(2))
    dp.pattern_visualizations(df_lin_log, "./lin_log")
    dp.pattern_visualizations(df_log_lin, "./log_lin")
    dp.pattern_visualizations(df_log_log, "./log_log")
    train_X, train_y, val_X, val_y, test_X, test_y = dp.train_val_test_split(df, [0.7, 0.15, 0.15], "original")
    print(f"shape: train_X: {train_X.shape}")
    print(f"shape: train_y: {train_y.shape}")
    print(f"shape: val_X: {val_X.shape}")
    print(f"shape: val_y: {val_y.shape}")
    print(f"shape: test_X: {test_X.shape}")
    print(f"shape: test_y: {test_y.shape}")

    train_X, train_y, val_X, val_y, test_X, test_y = dp.train_val_test_split(df_lin_log, [0.7, 0.15, 0.15], "original")
    print(f"shape: train_X: {train_X.shape}")
    print(f"shape: train_y: {train_y.shape}")
    print(f"shape: val_X: {val_X.shape}")
    print(f"shape: val_y: {val_y.shape}")
    print(f"shape: test_X: {test_X.shape}")
    print(f"shape: test_y: {test_y.shape}")

    train_X, train_y, val_X, val_y, test_X, test_y = dp.train_val_test_split(df_log_log, [0.7, 0.15, 0.15], "original")
    print(f"shape: train_X: {train_X.shape}")
    print(f"shape: train_y: {train_y.shape}")
    print(f"shape: val_X: {val_X.shape}")
    print(f"shape: val_y: {val_y.shape}")
    print(f"shape: test_X: {test_X.shape}")
    print(f"shape: test_y: {test_y.shape}")

    train_X, train_y, val_X, val_y, test_X, test_y = dp.train_val_test_split(df_log_lin, [0.7, 0.15, 0.15], "original")
    print(f"shape: train_X: {train_X.shape}")
    print(f"shape: train_y: {train_y.shape}")
    print(f"shape: val_X: {val_X.shape}")
    print(f"shape: val_y: {val_y.shape}")
    print(f"shape: test_X: {test_X.shape}")
    print(f"shape: test_y: {test_y.shape}")