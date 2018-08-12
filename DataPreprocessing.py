# Data preprocessing

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing dataset
dataset = pd.read_csv('Data.csv')

# Elimizdeki datasetini masrise çeviriyoruz. : tüm veriyi al :-1 sondaki kolonu alma anlamında
X = dataset.iloc[:, :-1].values

# Sonuç vektörünü oluşturuyoruz
Y = dataset.iloc[:, 3].values