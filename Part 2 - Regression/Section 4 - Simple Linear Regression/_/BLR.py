# Basit Doğrusal Regresyon

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('satislar.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 0:1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_Y = StandardScaler()
y_train = sc_Y.fit_transform(y_train)
y_test = sc_Y.transform(y_test)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Aylar vs Satışlar (Eğitim Seti)')
plt.xlabel('Aylar')
plt.ylabel('Satış')
plt.show()

plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Aylar vs Satışlar (Test Seti)')
plt.xlabel('Aylar')
plt.ylabel('Satış')
plt.show()