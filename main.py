__author__ = "Soumyadip Majumder"
__version__ = "1.0.0"
__maintainer__ = "Soumyadip Majumder"
__status__ = "Test"
__date__ = "29 Jan 2023"

import pandas as pd
from prepare_data import DataPrep as dp
from build_models import Model
from sklearn.linear_model import Ridge



    
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

# Choosing the best transformed dataq for models: Lin-log Transformation 

print("")
print("Ridge")

print("Lin_Log")
mdl = Model(df_lin_log, [0.7, 0.15, 0.15], "lin_log")
mdl.build_ridge_model()
mdl.evaluate_ridge_reg()


print("")
print("Lasso")

print("Lin_Log")
mdl = Model(df_lin_log, [0.7, 0.15, 0.15], "lin_log")
mdl.build_lasso_model()
mdl.evaluate_lasso_reg()

mdl = Model(df_lin_log, [0.7, 0.15, 0.15], "lin_log")
print("Ridge")
ridge_best_alpha = mdl.hyperparameter_tuning("Ridge")
print(ridge_best_alpha)
print("Lasso")
lasso_best_alpha = mdl.hyperparameter_tuning("Lasso")
print(lasso_best_alpha)