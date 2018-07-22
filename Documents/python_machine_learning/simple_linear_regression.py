# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 19:44:10 2018

@author: Russell Murray
"""

# Simple Linear Regression

# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
# Extracting the Indepent variables and storing into matrix
X = dataset.iloc[:, :-1].values
# Extracting the Dependent variables and creating vector
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting Simple Linear Regression to The Training set
from sklearn.linear_model import LinearRegression
# Declaring and instantiating Linear Regressor object
regressor = LinearRegression()
# Fitting training data to the model
regressor.fit(X_train, y_train)

# Predicting the Test set results (Dependent Variable Vector)
y_pred = regressor.predict(X_test)

# Visualizing the Training set results
plt.scatter(X_train, y_train, color = 'red')
# Plot regression line by fitting Linear Regressor to the Training set
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Year of Experience')
plt.ylabel('Salary')
plt.show()

# Visualizing the Test set results
plt.scatter(X_test, y_test, color = 'red')
# Plot regression line by fitting Linear Regressor to the Training set
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Year of Experience')
plt.ylabel('Salary')
plt.show()