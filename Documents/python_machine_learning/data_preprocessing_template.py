# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 17:08:03 2018

@author: Russell Murray
"""
import numpy as np #Library that contains mathematical tools
import matplotlib.pyplot as plt #Library used for plotting 
import pandas as pd #Library used to import and manage datasets

# Import the dataset
dataset = pd.read_csv('Data.csv')

# Create Matrix of Features
# Extracting pertinent data for generating the matrix features and
# : means all, :-1 means all but last (in row, column format)
X = dataset.iloc[:, :-1].values 

# Create the Dependent Variable Vector
y = dataset.iloc[:, 3].values

# Taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
# Fitting imputer to Matrix
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
# Declaring and instantiating LabelEncoder object
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()

labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Splitting the dataset into the Training set and the Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)