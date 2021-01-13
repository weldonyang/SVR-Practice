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
from sklearn.svm import SVR 
regressor = SVR(kernel='rbf')
regressor.fit(X, y)

# Predicting a new result
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform([[6.5]])))

# Visualising the SVR results
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color='red')                      
plt.plot(sc_X.inverse_transform(X), sc_y.inverse_transform(regressor.predict(X)), color='blue')    
plt.title('Salary vs Position Level (SVR)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Visualising the SVR results (for higher resolution and smoother curve)
# NOTE: if you have a large outlier, SVR will NOT catch it properly 
X_grid = np.arange(min(sc_X.inverse_transform(X)), max(sc_X.inverse_transform(X)), 0.1)                                               
X_grid = X_grid.reshape((len(X_grid), 1))   
plt.scatter(sc_X.inverse_transform(X), sc_y.inverse_transform(y), color='red')                       
plt.plot(X_grid, sc_y.inverse_transform(regressor.predict(sc_X.transform(X_grid))), color='blue')   # transforms X_grid into the correct matrix of degrees 
plt.title('Salary vs Position Level (SVR)')
plt.xlabel('Postion Level')
plt.ylabel('Salary')
plt.show()