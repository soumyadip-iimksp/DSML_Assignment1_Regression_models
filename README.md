# DSML_Assignment1_Regression_models

## Goal: 
In this assignment, you will implement Linear Regression, Ridge Regression and Lasso Regression. The goal of this assignment is to give you experience in model selection and analyze the result with hyperparameter tunning. 
Data sets: 

The dataset that you use in this project is “Boston Housing Data”. The independent variables are “CRIM”, “ZN”, “INDUS”, “CHAS”, “NOX”, “RM”, “AGE”, “DIS”, “RAD”, “TAX”, “TRATIO”, “B”, “LSTAT” and dependent variable “MEDV”. The detail about data is given in the following links. 
Divide the data into 70%, 15% and 15% training, validation and test set. 
 
## Links:
[Data Description](https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.names)

[Data](https://raw.githubusercontent.com/jbrownlee/Datasets/master/housing.csv)

## Tasks:
1.	Implementing Linear Regression and do some model selection on validation set. Find the best model for Linear Regression using validation set and evaluate on test set.
2.	Use the selected best model from Linear Regression, develop Ridge Regression and Lasso Regression. 
3.	Do hyperparameter tuning for Ridge Regression and Lasso Regression on validation set and find the best hyperparameter using validation set and evaluate on test set. 
4.	Do comparative result analysis for Linear Regression, Ridge Regression and Lasso Regression.


### To run the code
1. Download/clone the code
1. install pipenv using `pip install pipenv`
2. Install all the dependent libraries using `pipenv install`
3. Run the code `pipenv run python main.py`

### Outputs

**Linear Regression**
*Original data*
Val RMSE: 5.5598
Val R Squared: 0.7008
Test RMSE: 6.1341
Test R Squared: 0.6189
*Lin_Log*
Val RMSE: 5.2077
Val R Squared: 0.7375
Test RMSE: `4.9554`
Test R Squared: `0.7513`
*Log_Lin*
Val RMSE: 0.3472
Val R Squared: 0.7651
Test RMSE: 0.3822
Test R Squared: 0.6756
*Log_Log*
Val RMSE: 0.3829
Val R Squared: 0.7143
Test RMSE: 0.3565
Test R Squared: 0.7177

Choosing the best transformation **Lin-Log** for *Ridge* and *Lasso* Regression

**Ridge**
*Lin_Log*
Val RMSE: 5.0037
Val R Squared: 0.7577
Test RMSE: `5.1454`
Test R Squared: `0.7319`

**Lasso**
*Lin_Log*
Val RMSE: 6.5309
Val R Squared: 0.5871
Test RMSE: `7.1149`
Test R Squared: `0.4873`

Hyperparameter Tuning with *alpha*

**Ridge Regression**

   | alpha | R2 | rmse |
   | --- | --- | --- |
     0.0 | 0.737 | 5.208
     0.1 | 0.752 | 5.064
     0.2 | 0.757 | 5.012
     0.3 | 0.758 | 4.996
     0.4 | 0.758 | 4.996
     0.5 | 0.758 | 5.004
     0.6 | 0.756 | 5.016
     0.7 | 0.755 | 5.030
     0.8 | 0.754 | 5.046
     0.9 | 0.752 | 5.062
     1.0 | 0.750 | 5.078
**best alpha=** `0.5`
**Test RMSE:** `5.1454`
**Test R Squared:** `0.7319`

**Lasso regression**
     | alpha | R2 | rmse |
     | --- | --- | --- |
      0.0 | 0.737 | 5.208
      0.1 | 0.682 | 5.733
      0.2 | 0.640 | 6.094
      0.3 | 0.627 | 6.212
      0.4 | 0.609 | 6.357
      0.5 | 0.587 | 6.531
      0.6 | 0.561 | 6.731
      0.7 | 0.532 | 6.956
      0.8 | 0.498 | 7.202
      0.9 | 0.460 | 7.467
      1.0 | 0.418 | 7.751
**best alpha=** `0.0`
**Test RMSE:** `4.9554`
**Test R Squared:** `0.7513`


**Observation:** Linear regression with R-square value of `0.7513` has few features which as not significant (testes separately with OLS models) and leading to inflated R2 value with some overfitting to noise. Ridge Regression with the `L2 Regularizer` penalises those not significant features and has reduces the R-square value minimaly to `0.7319`. Lasso regression at `alpha=0`, it is acting as a Linear regression model. WIth increaase in *alpha*, it is suppressing the features and at `alpha=1.0`, the coefficient of some of the unwanted features has completely turned to *Zero* thereby affecting the R-square value and increase in `RMSE`

Further improvemnts can be tried with k-Fold Cross Validation