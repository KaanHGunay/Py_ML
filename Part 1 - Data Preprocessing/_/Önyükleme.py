# Verilerin Ön Yüklenmesi

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

veriler = pd.read_csv('eksikveriler.csv')

# Eksik verilerin düzenlenmesi
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
Yas = veriler.iloc[:, 1:4].values
imputer = imputer.fit(Yas[:, 1:4])
Yas[:, 1:4] = imputer.transform(Yas[:, 1:4])

# Kategorik veriler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
ulke = veriler.iloc[:, 0:1].values
le = LabelEncoder()
ulke[:, 0] = le.fit_transform(ulke[:, 0])
ohe = OneHotEncoder(categorical_features = 'all')
ulke = ohe.fit_transform(ulke).toarray()

# Data Frame 
sonuc = pd.DataFrame(data = ulke, index = range(22), columns = ['fr', 'tr', 'us'])
sonuc2 = pd.DataFrame(data = Yas, index = range(22), columns = ['boy', 'kilo', 'yas'])

cinsiyet = veriler.iloc[:, -1:].values
sonuc3 = pd.DataFrame(data = cinsiyet, index = range(22), columns = ['cinsiyet'])

veri = pd.concat([sonuc, sonuc2], axis = 1)
c_veri = pd.concat([veri, sonuc3], axis = 1)

# Veri Kümesini eğitim ve test kümesi
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(veri, sonuc3, test_size = 1/3, random_state = 0)

# Öznitelik ölçekleme
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)
