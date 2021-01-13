# Support Vector Regression (SVR)

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
# Change y to 2D array because StandardScaler.fit_transform() only takes a 2D array input
y = y.reshape(len(y), 1) 

# Feature Scaling
# We have to apply feature scaling with SVR because there is no explicit equation of the dependent variable in respect to the features 
# There aren't coefficients multiplying each feature which means SVR does not compensate for higher feature values with lower coefficients 
# There is an implicit equation of the depedent variable in respect to the features 
from sklearn.preprocessing import StandardScaler 
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)
print(X)
print(y)

# Training the SVR model on the whole dataset

# Predicting a new result

# Visualising the SVR results

# Visualising the SVR results (for higher resolution and smoother curve)
