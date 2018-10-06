# Natural Language Processing

# Importing the library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3) 
# tsv olması nedenitle tab ile ayrılacağını delimeter parametresi ile belirttik
# quoting ile " işaretinin oluşturacağı sorunları görmezden gelmesi sağlandı

# Cleaning the texts
import re # Yazıları düzenleme kütüphanesi
import nltk
nltk.download('stopwords') # Olumlu veya olumsuz anlam içermeyen kelimelerin silinmesi için
from nltk.corpus import stopwords # gereksiz kelimeler için
from nltk.stem.porter import PorterStemmer # İngilizcedeki kelimelerin geçmiş/gelecek/şimdiki hallerini eşitleyecek 
# Yazılarda bulunan işaret ve sayıların temizlenmesi
# sub metotuna verilen paramtreler silinmeyecek olanlardır
review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][0]) # a dan z ye harfler ve boşluk silinmeyecek
review = review.lower() # bütün harflerin küçültülmesi
review = review.split() # kelimleri liste haline getirme
ps = PorterStemmer() 
review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))] # Tüm kelimelerin düzenlenmesi
review = ' '.join(review) # Kelimelerin tekrar birleştirilmesi

corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower() 
    review = review.split() 
    ps = PorterStemmer() 
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))] 
    review = ' '.join(review)
    corpus.append(review)

# Creating the Bag of Words Model
from sklearn.feature_extraction.text import CountVectorizer
# Count Vectorizer kendi içerisinde küçük harfe dünüştürme ve gereksiz kelimleri atma özelliğine sahiptir.
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting classifier to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)