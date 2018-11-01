# Sentiment Alalysis with RNN

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, GRU, Embedding, CuDNNGRU
from keras.optimizers import Adam
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

dataset = pd.read_csv('hepsiburada.csv')
target = dataset['Rating'].values.tolist()
data = dataset['Review'].values.tolist()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size = 0.2, random_state = 0)

num_words = 10000 # 10.000 kelime ile çalışılacak
tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(data)

X_train_tokens = tokenizer.texts_to_sequences(X_train)
X_test_tokens = tokenizer.texts_to_sequences(X_test)

# Ne kadar token olduğu
num_tokens = [len(tokens) for tokens in X_train_tokens + X_test_tokens]
num_tokens = np.array(num_tokens)

# Yorumları eşit token sayısına eşitlemek için kullanıyoruz
max_tokens = np.mean(num_tokens) + 2 * np.std(num_tokens)
max_tokens = int(max_tokens)

# Bütün yorumları aynı seviyeye getiriyoruz
X_train_pad = pad_sequences(X_train_tokens, maxlen = max_tokens)
X_test_pad = pad_sequences(X_test_tokens, maxlen = max_tokens)

# Tokenleri geri yorum halini görebilmek için
idx = tokenizer.word_index
inverse_map = dict(zip(idx.values(), idx.keys()))
def tokens_to_string(tokens):
    words = [inverse_map[token] for token in tokens if token != 0]
    text = ' '.join(words)
    return text

# Model oluşturma ve eğitim
model = Sequential()
embedding_size = 50
model.add(Embedding(input_dim = num_words,
                    output_dim = embedding_size,
                    input_length = max_tokens,
                    name = 'embedding_layer'))

model.add(CuDNNGRU(units = 16, return_sequences = True))
model.add(CuDNNGRU(units = 8, return_sequences = True))
model.add(CuDNNGRU(units = 4, return_sequences = False))
model.add(Dense(1, activation = 'sigmoid'))
optimizer = Adam(lr = 1e-3)
model.compile(loss = 'binary_crossentropy',
              optimizer = optimizer,
              metrics = ['accuracy'])

model.fit(X_train_pad, y_train, epochs = 30, batch_size = 256)
model.save('model2.h5')

# Ne kadar başarılı oldupunu ölçüyoruz
result = model.evaluate(X_test_pad, y_test)

# Hatalı sonuçları inceleme
y_pred = model.predict(x = X_test_pad[0:1000])
y_pred = y_pred.T[0]
cls_pred = np.array([1.0 if p > 0.5 else 0.0 for p in y_pred])
cls_true = np.array(y_test[0:1000])
incorrect = np.where(cls_pred != cls_true)
incorrect = incorrect[0]

idx = incorrect[0]
text = X_test[idx]

text1 = "bu ürün çok iyi herkese tavsiye ederim"
text2 = "kargo çok hızlı aynı gün elime geçti"
text3 = "büyük bir hayal kırıklığı yaşadım bu ürün bu markaya yakışmamış"
text4 = "mükemmel"
text5 = "tasarımı harika ancak kargo çok geç geldi ve ürün açılmıştı tavsiye etmem"
text6 = "hiç resimde gösterildiği gibi değil"
text7 = "kötü yorumlar gözümü korkutmuştu ancak hiçbir sorun yaşamadım teşekkürler"
text8 = "hiç bu kadar kötü bir satıcıya denk gelmemiştim ürünü geri iade ediyorum"
text9 = "tam bir fiyat performans ürünü"
text10 = "beklediğim gibi çıkmadı"
texts = [text1, text2, text3, text4, text5, text6, text7, text8, text9, text10]
tokens = tokenizer.texts_to_sequences(texts)
tokens_pad = pad_sequences(tokens, maxlen=max_tokens)
model.predict(tokens_pad)