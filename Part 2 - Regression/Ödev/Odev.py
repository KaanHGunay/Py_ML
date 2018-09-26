import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('maaslar_yeni.csv')
X = dataset.iloc[:, 2:5].values
y = dataset.iloc[:, 5].values

# Multiple Linear Regression
from sklearn.linear_model import LinearRegression
regressor_mlr = LinearRegression()
regressor_mlr.fit(X, y)
y_pred_mlr = regressor_mlr.predict(X)
# 32861.59416921

# Polynomial Regression
from sklearn.preprocessing import PolynomialFeatures
regressor_poly = PolynomialFeatures(degree = 4)
X_poly = regressor_poly.fit_transform(X)
lin_reg = LinearRegression()
lin_reg.fit(X_poly,y)
y_pred_poly = lin_reg.predict(regressor_poly.fit_transform(X))
# 97031.03443134

# SVR
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X_sc = sc_X.fit_transform(X)
y_ = dataset.iloc[:, 5:6]
y_sc = sc_y.fit_transform(y_)

from sklearn.svm import SVR
regressor_svm = SVR(kernel = 'rbf')
regressor_svm.fit(X_sc, y_sc)

y_pred_svr = regressor_svm.predict(X_sc)
y_pred_svr = sc_y.inverse_transform(y_pred_svr)
# 5351.13649417

# Decision Tree
from sklearn.tree import DecisionTreeRegressor
regressor_tree = DecisionTreeRegressor(random_state=0)
regressor_tree.fit(X, y)
y_pred_dt = regressor_tree.predict(X)
# 60000

# Random Forest
from sklearn.ensemble import RandomForestRegressor
regressor_rf = RandomForestRegressor(n_estimators = 300, random_state = 0)
regressor_rf.fit(X, y)
y_pred_rf = regressor_rf.predict(X)
# 53513.33333333
