__author__ = "Soumyadip Majumder"
__version__ = "1.0.0"
__maintainer__ = "Soumyadip Majumder"
__status__ = "Test"
__date__ = "29 Jan 2023"

from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import r2_score, mean_squared_error
from prepare_data import DataPrep as dp
import pandas as pd
import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings("ignore")

class Model:

    def __init__(self, data:pd.DataFrame, split_ratio:list, dv_name:str):
        self.df = data
        self.split_ratio = split_ratio
        self.dv_name = dv_name

        self.train_X, self.train_y, self.val_X, self.val_y, self.test_X, self.test_y = dp.train_val_test_split(self, data=self.df,
                                                                                                                    split_ratios=self.split_ratio,
                                                                                                                    dv_name=self.dv_name)

    def build_eval_lin_reg_model(self):
        MODEL_PATH = "./models"
        self.lin_reg = LinearRegression()
        self.lin_reg.fit(self.train_X, self.train_y)

        if self.dv_name in  ["log_log", "log_lin"]:

            self.val_pred = self.lin_reg.predict(self.val_X)
            self.val_rmse = np.sqrt(mean_squared_error(np.exp(self.val_y), np.exp(self.val_pred)))
            self.val_r_squared = r2_score(np.exp(self.val_y), np.exp(self.val_pred))
        
        else:
            self.val_pred = self.lin_reg.predict(self.val_X)
            self.val_rmse = np.sqrt(mean_squared_error(self.val_y, self.val_pred))
            self.val_r_squared = r2_score(self.val_y,self.val_pred)

        print(f"Val RMSE: {round(self.val_rmse, 4)}")
        print(f"Val R Squared: {round(self.val_r_squared, 4)}")

        if not os.path.exists(MODEL_PATH):
            os.makedirs(MODEL_PATH)
        with open(f"{MODEL_PATH}/lin_reg_{self.dv_name}.bin", "wb") as f_out:
            pickle.dump(self.lin_reg, f_out)

    def evaluate_lin_reg(self):
        if self.dv_name == "log_log":
            with open(f"./models/lin_reg_{self.dv_name}.bin", "rb") as f_in:
                self.model = pickle.load(f_in)
            self.test_pred = self.model.predict(self.test_X)
            self.test_rmse = np.sqrt(mean_squared_error(np.exp(self.test_y), np.exp(self.test_pred)))
            self.test_r_squared = r2_score(np.exp(self.test_y), np.exp(self.test_pred))

            print(f"Test RMSE: {round(self.test_rmse, 4)}")
            print(f"Test R Squared: {round(self.test_r_squared, 4)}")

        elif self.dv_name == "log_lin":
            with open(f"./models/lin_reg_{self.dv_name}.bin", "rb") as f_in:
                self.model = pickle.load(f_in)
            self.test_pred = self.model.predict(self.test_X)
            self.test_rmse = np.sqrt(mean_squared_error(np.exp(self.test_y), np.exp(self.test_pred)))
            self.test_r_squared = r2_score(np.exp(self.test_y), np.exp(self.test_pred))
        
            print(f"Test RMSE: {round(self.test_rmse, 4)}")
            print(f"Test R Squared: {round(self.test_r_squared, 4)}")
        
        elif self.dv_name == "lin_log":
            with open(f"./models/lin_reg_{self.dv_name}.bin", "rb") as f_in:
                self.model = pickle.load(f_in)
            self.test_pred = self.model.predict(self.test_X)
            self.test_rmse = np.sqrt(mean_squared_error(self.test_y, self.test_pred))
            self.test_r_squared = r2_score(self.test_y, self.test_pred)
            
            print(f"Test RMSE: {round(self.test_rmse, 4)}")
            print(f"Test R Squared: {round(self.test_r_squared, 4)}")
        
        else:
            with open(f"./models/lin_reg_{self.dv_name}.bin", "rb") as f_in:
                self.model = pickle.load(f_in)
            self.test_pred = self.model.predict(self.test_X)
            self.test_rmse = np.sqrt(mean_squared_error(self.test_y, self.test_pred))
            self.test_r_squared = r2_score(self.test_y, self.test_pred)

            print(f"Test RMSE: {round(self.test_rmse, 4)}")
            print(f"Test R Squared: {round(self.test_r_squared, 4)}")

    def build_ridge_model(self, alpha=0.5):
        MODEL_PATH = "./models"
        self.lin_reg = Ridge(alpha=alpha)
        self.lin_reg.fit(self.train_X, self.train_y)

        if self.dv_name in  ["log_log", "log_lin"]:

            self.val_pred = self.lin_reg.predict(self.val_X)
            self.val_rmse = np.sqrt(mean_squared_error(np.exp(self.val_y), np.exp(self.val_pred)))
            self.val_r_squared = r2_score(np.exp(self.val_y), np.exp(self.val_pred))
        
        else:
            self.val_pred = self.lin_reg.predict(self.val_X)
            self.val_rmse = np.sqrt(mean_squared_error(self.val_y, self.val_pred))
            self.val_r_squared = r2_score(self.val_y,self.val_pred)

        print(f"Val RMSE: {round(self.val_rmse, 4)}")
        print(f"Val R Squared: {round(self.val_r_squared, 4)}")

        if not os.path.exists(MODEL_PATH):
            os.makedirs(MODEL_PATH)
        with open(f"{MODEL_PATH}/ridge_reg_{self.dv_name}.bin", "wb") as f_out:
            pickle.dump(self.lin_reg, f_out)

    def evaluate_ridge_reg(self):
        if self.dv_name == "log_log":
            with open(f"./models/ridge_reg_{self.dv_name}.bin", "rb") as f_in:
                self.model = pickle.load(f_in)
            self.test_pred = self.model.predict(self.test_X)
            self.test_rmse = np.sqrt(mean_squared_error(np.exp(self.test_y), np.exp(self.test_pred)))
            self.test_r_squared = r2_score(np.exp(self.test_y), np.exp(self.test_pred))

            print(f"Test RMSE: {round(self.test_rmse, 4)}")
            print(f"Test R Squared: {round(self.test_r_squared, 4)}")

        elif self.dv_name == "log_lin":
            with open(f"./models/ridge_reg_{self.dv_name}.bin", "rb") as f_in:
                self.model = pickle.load(f_in)
            self.test_pred = self.model.predict(self.test_X)
            self.test_rmse = np.sqrt(mean_squared_error(np.exp(self.test_y), np.exp(self.test_pred)))
            self.test_r_squared = r2_score(np.exp(self.test_y), np.exp(self.test_pred))
        
            print(f"Test RMSE: {round(self.test_rmse, 4)}")
            print(f"Test R Squared: {round(self.test_r_squared, 4)}")
        
        elif self.dv_name == "lin_log":
            with open(f"./models/ridge_reg_{self.dv_name}.bin", "rb") as f_in:
                self.model = pickle.load(f_in)
            self.test_pred = self.model.predict(self.test_X)
            self.test_rmse = np.sqrt(mean_squared_error(self.test_y, self.test_pred))
            self.test_r_squared = r2_score(self.test_y, self.test_pred)
            
            print(f"Test RMSE: {round(self.test_rmse, 4)}")
            print(f"Test R Squared: {round(self.test_r_squared, 4)}")
        
        else:
            with open(f"./models/ridge_reg_{self.dv_name}.bin", "rb") as f_in:
                self.model = pickle.load(f_in)
            self.test_pred = self.model.predict(self.test_X)
            self.test_rmse = np.sqrt(mean_squared_error(self.test_y, self.test_pred))
            self.test_r_squared = r2_score(self.test_y, self.test_pred)

            print(f"Test RMSE: {round(self.test_rmse, 4)}")
            print(f"Test R Squared: {round(self.test_r_squared, 4)}")

    def build_lasso_model(self, alpha=0.5):
        MODEL_PATH = "./models"
        self.lin_reg = Lasso(alpha=alpha)
        self.lin_reg.fit(self.train_X, self.train_y)

        if self.dv_name in  ["log_log", "log_lin"]:

            self.val_pred = self.lin_reg.predict(self.val_X)
            self.val_rmse = np.sqrt(mean_squared_error(np.exp(self.val_y), np.exp(self.val_pred)))
            self.val_r_squared = r2_score(np.exp(self.val_y), np.exp(self.val_pred))
        
        else:
            self.val_pred = self.lin_reg.predict(self.val_X)
            self.val_rmse = np.sqrt(mean_squared_error(self.val_y, self.val_pred))
            self.val_r_squared = r2_score(self.val_y,self.val_pred)

        print(f"Val RMSE: {round(self.val_rmse, 4)}")
        print(f"Val R Squared: {round(self.val_r_squared, 4)}")

        if not os.path.exists(MODEL_PATH):
            os.makedirs(MODEL_PATH)
        with open(f"{MODEL_PATH}/lasso_reg_{self.dv_name}.bin", "wb") as f_out:
            pickle.dump(self.lin_reg, f_out)

    def evaluate_lasso_reg(self):
        if self.dv_name == "log_log":
            with open(f"./models/ridge_reg_{self.dv_name}.bin", "rb") as f_in:
                self.model = pickle.load(f_in)
            self.test_pred = self.model.predict(self.test_X)
            self.test_rmse = np.sqrt(mean_squared_error(np.exp(self.test_y), np.exp(self.test_pred)))
            self.test_r_squared = r2_score(np.exp(self.test_y), np.exp(self.test_pred))

            print(f"Test RMSE: {round(self.test_rmse, 4)}")
            print(f"Test R Squared: {round(self.test_r_squared, 4)}")

        elif self.dv_name == "log_lin":
            with open(f"./models/lasso_reg_{self.dv_name}.bin", "rb") as f_in:
                self.model = pickle.load(f_in)
            self.test_pred = self.model.predict(self.test_X)
            self.test_rmse = np.sqrt(mean_squared_error(np.exp(self.test_y), np.exp(self.test_pred)))
            self.test_r_squared = r2_score(np.exp(self.test_y), np.exp(self.test_pred))
        
            print(f"Test RMSE: {round(self.test_rmse, 4)}")
            print(f"Test R Squared: {round(self.test_r_squared, 4)}")
        
        elif self.dv_name == "lin_log":
            with open(f"./models/lasso_reg_{self.dv_name}.bin", "rb") as f_in:
                self.model = pickle.load(f_in)
            self.test_pred = self.model.predict(self.test_X)
            self.test_rmse = np.sqrt(mean_squared_error(self.test_y, self.test_pred))
            self.test_r_squared = r2_score(self.test_y, self.test_pred)
            
            print(f"Test RMSE: {round(self.test_rmse, 4)}")
            print(f"Test R Squared: {round(self.test_r_squared, 4)}")
        
        else:
            with open(f"./models/lasso_reg_{self.dv_name}.bin", "rb") as f_in:
                self.model = pickle.load(f_in)
            self.test_pred = self.model.predict(self.test_X)
            self.test_rmse = np.sqrt(mean_squared_error(self.test_y, self.test_pred))
            self.test_r_squared = r2_score(self.test_y, self.test_pred)

            print(f"Test RMSE: {round(self.test_rmse, 4)}")
            print(f"Test R Squared: {round(self.test_r_squared, 4)}")

    def hyperparameter_tuning(self, model_name:str):
        self.actual = self.test_y
        self.alphas = []
        self.r2_scores = []
        self.rmse_scores =[]
        for self.alpha in np.linspace(0,1,11):
            if model_name == "Ridge":
                self.model = Ridge(alpha=self.alpha, max_iter= 10000, random_state=22232)
            elif model_name == "Lasso":
                self.model = Lasso(alpha=self.alpha, random_state=22323)
            else:
                print("Wrong Selection")

            self.model.fit(self.train_X, self.train_y)
            self.val_pred = self.model.predict(self.val_X)
            self.val_rmse = np.sqrt(mean_squared_error(self.val_y, self.val_pred))
            self.val_r_squared = r2_score(self.val_y, self.val_pred)

            self.alphas.append(self.alpha)
            self.rmse_scores.append(self.val_rmse)
            self.r2_scores.append(self.val_r_squared)
          
        self.metrics = pd.DataFrame(list(zip(np.round(self.alphas, 2), np.round(self.r2_scores, 3), np.round(self.rmse_scores, 3))), columns=["alpha", "R2", "rmse"])
        print(self.metrics)
        self.best_alpha = max(self.metrics["alpha"][self.metrics["R2"] == max(self.metrics["R2"])])
        print(f"alpha= {self.best_alpha}")
        if model_name == "Ridge":
            self.best_model = Ridge(alpha=self.best_alpha)
        elif model_name == "Lasso":
            self.best_model = Lasso(alpha=self.best_alpha)

        
        self.best_model.fit(self.train_X, self.train_y)

        with open(f"./models/{model_name}_best_{self.dv_name}.bin", "wb") as f_out:
            pickle.dump(self.best_model, f_out)
        self.test_pred = self.best_model.predict(self.test_X)
        self.test_rmse = np.sqrt(mean_squared_error(self.test_y, self.test_pred))
        self.test_r_squared = r2_score(self.test_y, self.test_pred)

        print(f"Test RMSE: {round(self.test_rmse, 4)}")
        print(f"Test R Squared: {round(self.test_r_squared, 4)}")
        
        return self.best_alpha

if __name__ == "__main__":
    
    df = pd.read_csv("Housing_Price_data.csv")
    df_lin_log = dp().lin_log_data(df)
    df_log_lin = dp().log_lin_data(df)
    df_log_log = dp().log_log_data(df)
    
    print("Linear")
    print("original")
    mdl = Model(df, [0.7, 0.15, 0.15], "original")
    mdl.build_eval_lin_reg_model()
    mdl.evaluate_lin_reg()

    print("Lin_Log")
    mdl = Model(df_lin_log, [0.7, 0.15, 0.15], "lin_log")
    mdl.build_eval_lin_reg_model()
    mdl.evaluate_lin_reg()

    print("Log_Lin")
    mdl = Model(df_log_lin, [0.7, 0.15, 0.15], "log_lin")
    mdl.build_eval_lin_reg_model()
    mdl.evaluate_lin_reg()

    print("Log_Log")
    mdl = Model(df_log_log, [0.7, 0.15, 0.15], "log_log")
    mdl.build_eval_lin_reg_model()
    mdl.evaluate_lin_reg()

    print("")
    print("Ridge")
    print("original")
    mdl = Model(df, [0.7, 0.15, 0.15], "original")
    mdl.build_ridge_model()
    mdl.evaluate_ridge_reg()

    print("Lin_Log")
    mdl = Model(df_lin_log, [0.7, 0.15, 0.15], "lin_log")
    mdl.build_ridge_model()
    mdl.evaluate_ridge_reg()

    print("Log_Lin")
    mdl = Model(df_log_lin, [0.7, 0.15, 0.15], "log_lin")
    mdl.build_ridge_model()
    mdl.evaluate_ridge_reg()

    print("Log_Log")
    mdl = Model(df_log_log, [0.7, 0.15, 0.15], "log_log")
    mdl.build_ridge_model()
    mdl.evaluate_ridge_reg()

    print("")
    print("Lasso")
    print("original")
    mdl = Model(df, [0.7, 0.15, 0.15], "original")
    mdl.build_lasso_model()
    mdl.evaluate_lasso_reg()

    print("Lin_Log")
    mdl = Model(df_lin_log, [0.7, 0.15, 0.15], "lin_log")
    mdl.build_lasso_model()
    mdl.evaluate_lasso_reg()

    print("Log_Lin")
    mdl = Model(df_log_lin, [0.7, 0.15, 0.15], "log_lin")
    mdl.build_lasso_model()
    mdl.evaluate_lasso_reg()

    print("Log_Log")
    mdl = Model(df_log_log, [0.7, 0.15, 0.15], "log_log")
    mdl.build_lasso_model()
    mdl.evaluate_lasso_reg()

    print(mdl.hyperparameter_tuning("Ridge"))

    mdl.hyperparameter_tuning("Lasso")