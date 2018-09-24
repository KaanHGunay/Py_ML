import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing dataset
dataset = pd.read_csv('odev_tenis.csv')

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_X = LabelEncoder()
X[:, 0] = labelEncoder_X.fit_transform(X[:, 0])
oneHotEncoder = OneHotEncoder(categorical_features=[0])
X = oneHotEncoder.fit_transform(X).toarray()
labelEncode_y = LabelEncoder()
y = labelEncode_y.fit_transform(y)

# Avoiding the dummy variable trap
X = X[:, 1:] # Kütüphane tarafından yapılamaktadır. Elle yapmaya gerek yok

# Splitting the dataset into the training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)

# Fitting MLR to Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set Result
y_pred = regressor.predict(X_test)

# Building the optimal model using Backward Elimination (Manuel)
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((14, 1)).astype(int), values = X, axis = 1)
X_opt = X[:, [0, 2]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

