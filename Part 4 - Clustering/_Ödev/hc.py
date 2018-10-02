import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

veriler = pd.read_csv('musteriler.csv')
X = veriler.iloc[:,3:].values

#kmeans
from sklearn.cluster import KMeans
kmeans = KMeans (n_clusters = 4, init = 'k-means++')
kmeans.fit(X)

print(kmeans.cluster_centers_)
sonuclar = []
for i in range(1,11):
    kmeans = KMeans (n_clusters = i, init='k-means++', random_state= 123)
    kmeans.fit(X)
    sonuclar.append(kmeans.inertia_)

plt.plot(range(1,11),sonuclar)
plt.show()

kmeans = KMeans (n_clusters = 4, init='k-means++', random_state= 123)
y_kmeans= kmeans.fit_predict(X) 
plt.scatter(X[y_kmeans==0,0],X[y_kmeans==0,1],s=100, c='red')
plt.scatter(X[y_kmeans==1,0],X[y_kmeans==1,1],s=100, c='blue')
plt.scatter(X[y_kmeans==2,0],X[y_kmeans==2,1],s=100, c='green')
plt.scatter(X[y_kmeans==3,0],X[y_kmeans==3,1],s=100, c='yellow')
plt.title('KMeans')
plt.show()

#HC
from sklearn.cluster import AgglomerativeClustering
ac = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')
Y_tahmin = ac.fit_predict(X)

plt.scatter(X[Y_tahmin==0,0],X[Y_tahmin==0,1],s=100, c='red')
plt.scatter(X[Y_tahmin==1,0],X[Y_tahmin==1,1],s=100, c='blue')
plt.scatter(X[Y_tahmin==2,0],X[Y_tahmin==2,1],s=100, c='green')
plt.scatter(X[Y_tahmin==3,0],X[Y_tahmin==3,1],s=100, c='yellow')
plt.title('HC')
plt.show()

import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))
plt.show()

from sklearn.cluster import AffinityPropagation
ap = AffinityPropagation()
y_predict = ap.fit_predict(X)

plt.scatter(X[y_predict==0,0],X[y_predict==0,1],s=100, c='red')
plt.scatter(X[y_predict==1,0],X[y_predict==1,1],s=100, c='blue')
plt.scatter(X[y_predict==2,0],X[y_predict==2,1],s=100, c='green')
plt.scatter(X[y_predict==3,0],X[y_predict==3,1],s=100, c='yellow')
plt.scatter(X[y_predict==4,0],X[y_predict==4,1],s=100, c='cyan')
plt.scatter(X[y_predict==5,0],X[y_predict==5,1],s=100, c='black')
plt.scatter(X[y_predict==6,0],X[y_predict==6,1],s=100, c='magenta')
plt.scatter(X[y_predict==7,0],X[y_predict==7,1],s=100, c='#eeefff')
plt.scatter(X[y_predict==8,0],X[y_predict==8,1],s=100, c='#C0C0C0')
plt.title('AffinityPropagation')
plt.show()

from sklearn.cluster import MeanShift
ms = MeanShift()
y_mc = ms.fit_predict(X)

plt.scatter(X[y_mc==0,0],X[y_mc==0,1],s=100, c='red')
plt.scatter(X[y_mc==1,0],X[y_mc==1,1],s=100, c='blue')
plt.scatter(X[y_mc==2,0],X[y_mc==2,1],s=100, c='green')
plt.title('MeanShift')
plt.show()

from sklearn.cluster import SpectralClustering
spec = SpectralClustering()
y_spec = spec.fit_predict(X)

plt.scatter(X[y_spec==3,0],X[y_spec==3,1],s=100, c='yellow')
plt.scatter(X[y_spec==4,0],X[y_spec==4,1],s=100, c='cyan')
plt.scatter(X[y_spec==5,0],X[y_spec==5,1],s=100, c='black')
plt.scatter(X[y_spec==6,0],X[y_spec==6,1],s=100, c='magenta')
plt.scatter(X[y_spec==7,0],X[y_spec==7,1],s=100, c='#eeefff')
plt.scatter(X[y_spec==8,0],X[y_spec==8,1],s=100, c='#C0C0C0')
plt.title('SpectralClustering')
plt.show()
