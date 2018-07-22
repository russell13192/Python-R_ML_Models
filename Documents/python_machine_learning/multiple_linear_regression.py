# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 23:51:10 2018

@author: Russell Murray
"""

# Multiple Linear Regression
# Importing the Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# Declaring and instantiating LabelEncoder 
labelencoder_X = LabelEncoder()
# Selecting column index we want to encode
# labelEncoder only converts categorical column to numerical categorical values???
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
# OneHotEncoder provides numerical mathematics to the categorical column to properly train the model???
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the Dummy Variable Trap
# Removing the first column from X -> Ensures dataset will not contain redundant dependencies (variables??)
# Most Python ML Libraries for encoding categorical data normally take care of this for you!
X = X[:, 1:]

# Splitting the dataset into the Training set and the Testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting the Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
# Creating the Vector of predictions
y_pred = regressor.predict(X_test)

# Building the optimal model using Backward Elimination
import statsmodels.formula.api as sm
# We need to add a column of 1's to our matrix of Independent Varaibles so our model can correctly incorporate the Constant when fitting the Training set
# To ensure that our Multiple-Linear Regression equation is in fact y = b0x0 + b1x1 +b2x2 + ... + bnxn
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)

# Starting Backward Elimination
# Creating a new Matrix of features that will be our optimal Matrix of features
# Matrix will only contain features that are statistically significant for our Dependent variable Profit
# Backward Elimination starts by including all the features in the beginning
# And then removing them one by the Independent variables that are not statistically significant
X_opt = X[:, [0, 1, 2, 3, 4, 5]] # X_opt initialized as a Matrix of all the Independent variables that we have in our Dataset
# First step for Backward Elimination is to select a significance model in order for a feature to stay in the model
# We will use a significance level of 0.05 for this model
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit() # Model is not fitted with all possible predictors -> Step 2 in Backward Elimination
regressor_OLS.summary()

# After first iteration of calculations from regressor_OLS.summary(), column index 2 has the highest P-value of .990 and was above the Significance level of .05
# So we have to remove it from the model

X_opt = X[:, [0, 1, 3, 4, 5]] # X_opt is now initialized minus column index 2 as it had the highest P-value of .990 which was also greater that the Significance Level
# Of .05 we declared in order for a feature to stay in the model

regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit() # Model is not fitted with all possible predictors -> Step 2 in Backward Elimination
regressor_OLS.summary()

# After second iteration of calculations from regressor_OLS.summary(), column index 1 has the highest P-value of .940 and was above the Significance level of .05
# So we have to remove it from the model


X_opt = X[:, [0, 3, 4, 5]] # X_opt is now initialized minus column index 1 as it had the highest P-value of .940 which was also greater that the Significance Level
# Of .05 we declared in order for a feature to stay in the model

regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit() # Model is not fitted with all possible predictors -> Step 2 in Backward Elimination
regressor_OLS.summary()

# After third iteration of calculations from regressor_OLS.summary(), column index 2 has the highest P-value of .602 and was above the Significance level of .05
# So we have to remove it from the model

X_opt = X[:, [0, 3, 5]] # X_opt is now initialized minus column index 1 as it had the highest P-value of .940 which was also greater that the Significance Level
# Of .05 we declared in order for a feature to stay in the model

regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit() # Model is not fitted with all possible predictors -> Step 2 in Backward Elimination
regressor_OLS.summary()

# After fourth iteration of calculations from regressor_OLS.summary(), column index 2 has the highest P-value of .006 and was above the Significance level of .05
# So we have to remove it from the model

X_opt = X[:, [0, 3]] # X_opt is now initialized minus column index 1 as it had the highest P-value of .940 which was also greater that the Significance Level
# Of .05 we declared in order for a feature to stay in the model

regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit() # Model is not fitted with all possible predictors -> Step 2 in Backward Elimination
regressor_OLS.summary()

### Automatic implementation of Bacward Elimination using P-value only
"""import statsmodels.formula.api as sm
def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    regressor_OLS.summary()
    return x
 
SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)"""


### Automatic implementation of Backward Elimination using P-value and Adjusted R Squared
"""import statsmodels.formula.api as sm
def backwardElimination(x, SL):
    numVars = len(x[0])
    temp = np.zeros((50,6)).astype(int)
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        adjR_before = regressor_OLS.rsquared_adj.astype(float)
        if maxVar > SL:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    temp[:,j] = x[:, j]
                    x = np.delete(x, j, 1)
                    tmp_regressor = sm.OLS(y, x).fit()
                    adjR_after = tmp_regressor.rsquared_adj.astype(float)
                    if (adjR_before >= adjR_after):
                        x_rollback = np.hstack((x, temp[:,[0,j]]))
                        x_rollback = np.delete(x_rollback, j, 1)
                        print (regressor_OLS.summary())
                        return x_rollback
                    else:
                        continue
    regressor_OLS.summary()
    return x
 
SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)"""
# ----------- Model is now ready ---> the highest features P-value is less than the significance level required to stay in the model



