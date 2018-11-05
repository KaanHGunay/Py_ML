# Image Captioning

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
import sys
from PIL import Image

from keras import backend as K
from keras.models import Model
from keras.layers import Input, Dense, GRU, CuDNNGRU, Embedding
from keras.applications import VGG16
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import coco
from coco import cache

# Verilerin indirilmesi
coco.maybe_download_and_extract()

_, filenames, captions = coco.load_records(train = True)
num_images = len(filenames)

def load_image(path, size = None):
    img = Image.open(path)
    
    if not size is None:
        img = img.resize(size = size, resample = Image.LANCZOS)
    return img

def show_image(idx):
    dir = coco.train_dir
    filename = filenames[idx]
    caption = captions[idx]    
    path = os.path.join(dir, filename)
    
    for cap in caption:
        print(cap)
        
    img = load_image(path)
    plt.imshow(img)
    plt.show()

image_model = VGG16() # VGG16 modeli kullanılacak
transfer_layer = image_model.get_layer('fc2') # Son katman çıkarılacağı için bir öncekii katmanın ismi alınıyor
image_model_transfer = Model(inputs = image_model.input,
                             output = transfer_layer.output) # Model ilk katmanından sondan bir önceki katmana kadar alıyoruz
img_size = K.int_shape(image_model.input)[1:3] # Hangi boyutta resim alınacağı
transfer_values_size = K.int_shape(transfer_layer.output)[1] # Düşünce vektörüne bağlanacak nöron sayısı

# Eğitimi hızlandırma amacıyla:
# Her resmi bir kere VGG Modelinden geçirip sonuçları kaydedeceğiz
# Sonraki adımda eğitim esnasında VGG Modeli tekrar kullanılmayacak
# Fotoğrafların ne kadarının işlendiğini gösteren fonksiyon
def print_progress(count, max_count):
    pct_complete = count / max_count
    msg = '\r- Progress: {0:.1%}'.format(pct_complete)
    sys.stdout.write(msg)
    sys.stdout.flush()
    
# Fotoğrafların işlenmesi 
def process_images(data_dir, filenames, batch_size = 32):
    num_images = len(filenames)
    # Fotoğrafların matrislerinin tanıumlanması
    shape = (batch_size,) + img_size + (3,)
    image_batch = np.zeros(shape = shape, dtype = np.float16)
    # Transfer edilecek işlenmiş matris
    shape = (num_images, transfer_values_size)
    transfer_values = np.zeros(shape = shape, dtype = np.float16)
    start_index = 0
    
    while start_index < num_images:
        print_progress(count = start_index, max_count = num_images)
        end_index = start_index + batch_size
        
        if end_index > num_images:
            end_index = end_index
            
        current_batch_size = end_index - start_index
        
        for i, filename in enumerate(filenames[start_index:end_index]):
            path = os.path.join(data_dir, filename)
            img = load_image(path, size = img_size)
            image_batch[i] = img
            
        transfer_values_batch = image_model_transfer.predict(image_batch[0:current_batch_size])
        transfer_values[start_index:end_index] = transfer_values_batch[0:current_batch_size]
        start_index = end_index
    print()
    return transfer_values

# İşlenen fotoğrafların bilgisayarda kayıtlı ise kullan değilse hesapla
def process_train_images():
    print('Eğitim setindeki {0} resim işleniyor'.format(len(filenames)))
    cache_path = os.path.join(coco.data_dir, 'transfer_values_train.pkl')
    transfer_values = cache(cache_path = cache_path,
                            fn = process_images,
                            data_dir = coco.train_dir,
                            filenames = filenames)
    return transfer_values

transfer_values = process_train_images()
print('Shape:', transfer_values.shape)

mark_start = 'ssss '
mark_end = ' eeee'

def mark_captions(captions_listlist):
    captions_marked = [[mark_start + caption + mark_end
                        for caption in captions_list]
                        for captions_list in captions_listlist]
    return captions_marked

captions_marked = mark_captions(captions)

def flatten(captions_listlist):
    captions_list = [caption
                     for captions_list in captions_listlist
                     for caption in captions_list]
    return captions_list

captions_flat = flatten(captions_marked)
num_words = 10000

class TokenizerWrap(Tokenizer): # CP captions_to_tokens'i kopyalama, yazılacak
    
    def __init__(self, texts, num_words=None):
        Tokenizer.__init__(self, num_words=num_words)        
        self.fit_on_texts(texts)
        self.index_to_word = dict(zip(self.word_index.values(),
                                      self.word_index.keys()))

    def token_to_word(self, token):
        word = " " if token == 0 else self.index_to_word[token]
        return word 

    def tokens_to_string(self, tokens):
        words = [self.index_to_word[token] for token in tokens if token != 0]        
        text = " ".join(words)
        return text
    
    def captions_to_tokens(self, captions_listlist):
        tokens = [self.texts_to_sequences(captions_list)
                  for captions_list in captions_listlist]
        return tokens

tokenizer = TokenizerWrap(texts = captions_flat, num_words = num_words)
tokens_train = tokenizer.captions_to_tokens(captions_marked)
token_start = tokenizer.word_index[mark_start.strip()]
token_end = tokenizer.word_index[mark_end.strip()]

def get_random_caption_tokens(idx):
    result = []
    
    for i in idx:
        j = np.random.choice(len(tokens_train[i]))
        tokens = tokens_train[i][j]
        result.append(tokens)
    
    return result

def batch_generator(bathc_size):
    while True:
        idx = np.random.randint(num_images, size=batch_size)        
        t_values = transfer_values[idx]
        tokens = get_random_caption_tokens(idx)        
        num_tokens = [len(t) for t in tokens]
        max_tokens = np.max(num_tokens)        
        tokens_padded = pad_sequences(tokens,
                                      maxlen=max_tokens,
                                      padding='post',
                                      truncating='post')
        
        decoder_input_data = tokens_padded[:, 0:-1]
        decoder_output_data = tokens_padded[:, 1:]        
        x_data = {'decoder_input': decoder_input_data, 'transfer_values_input': t_values}
        y_data = {'decoder_output': decoder_output_data}        
        yield (x_data, y_data)
        
batch_size = 256
generator = batch_generator(batch_size) 
batch = next(generator)
batch_x = batch[0]
batch_y = batch[1]
num_captions = [len(caption) for caption in captions]
total_num_captions = np.sum(num_captions)
steps_per_epoch = int(total_num_captions / batch_size)
state_size = 256
embedding_size = 100
transfer_values_input = Input(shape=(transfer_values_size,),
                              name='transfer_values_input')

decoder_transfer_map = Dense(state_size,
                             activation='tanh',
                             name='decoder_transfer_map')

decoder_input = Input(shape=(None,), name='decoder_input')
word2vec = {} 
with open('glove.6B.100d.txt', encoding='UTF-8') as f:
  for line in f:
    values = line.split()
    word = values[0]
    vec = np.asarray(values[1:], dtype='float32')
    word2vec[word] = vec

embedding_matrix = np.random.uniform(-1, 1, (num_words, embedding_size)) 
for word, i in tokenizer.word_index.items():
    if i < num_words:
        embedding_vector = word2vec.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

decoder_embedding = Embedding(input_dim=num_words, 
                              output_dim=embedding_size,
                              weights=[embedding_matrix],
                              trainable=False,
                              name='decoder_embedding')

decoder_gru1 = CuDNNGRU(state_size, name='decoder_gru1', return_sequences=True)
decoder_gru2 = CuDNNGRU(state_size, name='decoder_gru2', return_sequences=True)
decoder_gru3 = CuDNNGRU(state_size, name='decoder_gru3', return_sequences=True)

decoder_dense = Dense(num_words,
                      activation='linear',
                      name='decoder_output')

def connect_decoder(transfer_values):
    initial_state = decoder_transfer_map(transfer_values)
    net = decoder_input
    net = decoder_embedding(net)
    net = decoder_gru1(net, initial_state=initial_state)
    net = decoder_gru2(net, initial_state=initial_state)
    net = decoder_gru3(net, initial_state=initial_state)
    decoder_output = decoder_dense(net)    
    return decoder_output

decoder_output = connect_decoder(transfer_values=transfer_values_input)

decoder_model = Model(inputs=[transfer_values_input, decoder_input],
                      outputs=[decoder_output])

def sparse_cross_entropy(y_true, y_pred):
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true,
                                                          logits=y_pred)
    loss_mean = tf.reduce_mean(loss)
    return loss_mean

optimizer = RMSprop(lr=1e-3)
decoder_target = tf.placeholder(dtype='int32', shape=(None, None))
decoder_model.compile(optimizer=optimizer,
                      loss=sparse_cross_entropy,
                      target_tensors=[decoder_target])
path_checkpoint = 'checkpoint.keras'
checkpoint = ModelCheckpoint(filepath=path_checkpoint, save_weights_only=True)
try:
    decoder_model.load_weights(path_checkpoint)
except Exception as error:
    print("Checkpoint yüklenirken hata oluştu. Eğitime sıfırdan başlanıyor.")
    print(error)

decoder_model.fit_generator(generator=generator,
                            steps_per_epoch=steps_per_epoch,
                            epochs=50,
                            callbacks=[checkpoint])

def generate_caption(image_path, max_tokens=30):
    image = load_image(image_path, size=img_size)    
    image_batch = np.expand_dims(image, axis=0)    
    transfer_values = image_model_transfer.predict(image_batch)    
    decoder_input_data = np.zeros(shape=(1, max_tokens), dtype=np.int)
    token_int = token_start
    output_text = ''
    count_tokens = 0

    while token_int != token_end and count_tokens < max_tokens:
        decoder_input_data[0, count_tokens] = token_int
        x_data = {'transfer_values_input': transfer_values, 'decoder_input': decoder_input_data}
        decoder_output = decoder_model.predict(x_data)
        token_onehot = decoder_output[0, count_tokens, :]
        token_int = np.argmax(token_onehot)
        sampled_word = tokenizer.token_to_word(token_int)
        output_text += " " + sampled_word
        count_tokens += 1
        
    plt.imshow(image)
    plt.show()    
    print('Predicted caption:')
    print(output_text)
    print()


def generate_caption_coco(idx):
    data_dir = coco.train_dir
    filename = filenames[idx]
    caption = captions[idx]    
    path = os.path.join(data_dir, filename)    
    generate_caption(image_path=path)    
    print('True captions:')
    for cap in caption:
        print(cap)
