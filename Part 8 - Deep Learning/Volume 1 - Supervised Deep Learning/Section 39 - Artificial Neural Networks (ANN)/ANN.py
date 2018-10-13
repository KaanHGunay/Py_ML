# Artificial Neural Networks

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_X_1 = LabelEncoder()
X[:, 1] = labelEncoder_X_1.fit_transform(X[:, 1]) # Ülkeleri sayısal veriye dönüştürme
labelEncoder_X_2 = LabelEncoder()
X[:, 2] = labelEncoder_X_2.fit_transform(X[:, 2]) # Cinsiyeti sayısal veriye dönüştürme
oneHotEncoder = OneHotEncoder(categorical_features=[1])
X = oneHotEncoder.fit_transform(X).toarray()
X = X[:, 1:] # Dummy Var. Trap

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Make ANN
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout # Rastgele bazı katmanları çalıştırmaz ve aşırı öğrenmenin önüne geçer
# Genellikle 1. ve 2. katmana uygulanması daha doğru olur (Aşırı öğrenmenin genellikle kaynağı bu katmanlardır.)

# Initializing ANN
classifier = Sequential()

# Adding the input layer and first hidden layer
classifier.add(Dense(kernel_initializer = 'uniform', input_dim = 11, activation = 'relu', units = 6)) # relu = rectifier algoritması
classifier.add(Dropout(p = 0.1))

# Adding second hidden layer
classifier.add(Dense(kernel_initializer = 'uniform', activation = 'relu', units = 6))
classifier.add(Dropout(p = 0.1))

# Adding the output layer
classifier.add(Dense(kernel_initializer = 'uniform', activation = 'sigmoid', units = 1)) # sigmoid = sigmoid alogritması
# Output layer ikili sonuçtan fazla ise Activation Softmax kullanılır

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy']) # stochastic algritmalardan adam algoritması kullanılacak
# İki seçenekli output layer olduğu için loss = binary_crossentropy
# İkiden fazla olduğu durumlarda categorical_crossentropy

# Fitting the ANN to the Trainig set
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100) 

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Tek bir örnekten predict alma 
new_prediction = classifier.predict(sc.transform(np.array([[0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))

# Evaluating ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from keras.models import Sequential
from keras.layers import Dense
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(kernel_initializer = 'uniform', input_dim = 11, activation = 'relu', units = 6))
    classifier.add(Dense(kernel_initializer = 'uniform', activation = 'relu', units = 6))
    classifier.add(Dense(kernel_initializer = 'uniform', activation = 'sigmoid', units = 1))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier, 
                             batch_size = 10, 
                             epochs = 100)

accuracies = cross_val_score(estimator = classifier, 
                             X = X_train, 
                             y = y_train,
                             cv = 10
                             # n_jobs = -1  # Windows ile ilgili problem paralel hesaplama yapmıyor
                             )
mean = accuracies.mean()
variance = accuracies.std()

# Parameter Tuning
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
def build_classifier(optimizer, kernel_initializer, activation):
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = kernel_initializer, activation = activation, input_dim = 11))
    classifier.add(Dense(units = 6, kernel_initializer = kernel_initializer, activation = activation))
    classifier.add(Dense(units = 1, kernel_initializer = kernel_initializer, activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn = build_classifier)

parameters = {'batch_size': [25, 32],
              'epochs': [50, 60],
              'kernel_initializer' : ['glorot_uniform', 'uniform'],
              'activation' : ['elu', 'relu'],
              'optimizer': ['adam', 'rmsprop']}

grid_search = GridSearchCV(estimator = classifier,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)

grid_search = grid_search.fit(X_train, y_train)
best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_