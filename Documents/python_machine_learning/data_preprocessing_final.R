# Data Preprocessing

# Import the dataset
dataset = read.csv('Data.csv')

# Splitting the dataset into the Training and Test set
library(caTools)
set.seed(123)
split = sample.split(dataset$Purchased, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)
