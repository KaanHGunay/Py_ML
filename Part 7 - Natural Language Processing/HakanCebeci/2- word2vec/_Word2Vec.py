# Word2Vec Algoritm

import numpy as np
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

f = open('hurriyet.txt', 'r', encoding = 'utf-8')
text = f.read()
t_list = text.split('\n')
corpus = []

for cumle in t_list:
    corpus.append(cumle.split())
    
model = Word2Vec(corpus, # Training set
                 size = 100, # 100 lük bir vektör oluşturulacak
                 window = 5,  # Çevredeki 5 kelimeye bakılacak
                 min_count = 5, # En az 5 defa geçen kelimeler dikkate alınacak
                 sg = 1 # Skip-Gram algoritması kullanılacak
                 )

# Kelime vektörüne ulaşma
print(model.wv['ankara'])

# Benze kelimeleri çıkarma
print(model.wv.most_similar('youtube'))

# Model kaydetme
model.save('word2vec.model')

# Modeli kullanma
model = Word2Vec.load('word2vec.model')

# Yakın kelimeleri görselleştirme
def closestwords_tsneplot(model, word):
    word_vectors = np.empty((0,100))
    word_labels = [word]    
    close_words = model.wv.most_similar(word)    
    word_vectors = np.append(word_vectors, np.array([model.wv[word]]), axis=0)
    
    for w, _ in close_words:
        word_labels.append(w)
        word_vectors = np.append(word_vectors, np.array([model.wv[w]]), axis=0)
        
    tsne = TSNE(random_state=0)
    Y = tsne.fit_transform(word_vectors)    
    x_coords = Y[:, 0]
    y_coords = Y[:, 1]    
    plt.scatter(x_coords, y_coords)    
    for label, x, y in zip(word_labels, x_coords, y_coords):
        plt.annotate(label, xy=(x, y), xytext=(5, -2), textcoords='offset points')        
    plt.show()
    
closestwords_tsneplot(model, 'pkk')