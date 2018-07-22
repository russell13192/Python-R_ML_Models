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