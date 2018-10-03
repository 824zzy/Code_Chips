from __future__ import print_function
from keras.preprocessing import sequence
from keras.datasets import imdb

max_features = 20000
maxlen = 80
batch_size = 32

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

print(len(x_train), 'train sequences')
print(len(y_train), 'test sequences')

x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape', x_train.shape)
print('x_test shape', x_test.shape)

from keras.models import Model
from keras.layers import *
from attention_keras import Attention 

S_inputs = Input(shape=(None,), dtype='int32')
embeddings = Embedding(max_features, 128)(S_inputs)
#  embeddings = Position_Embedding()(embeddings)
O_seq = Attention(8, 16)([embeddings, embeddings, embeddings])
O_seq = GlobalAveragePooling1D()(O_seq)
O_seq = Dropout(0.5)(O_seq)
outputs = Dense(1, activation='sigmoid')(O_seq)

model = Model(inputs=S_inputs, outputs=outputs)

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
print('train...')

model.fit(x_train, y_train, batch_size=batch_size,
          epochs=5,
          validation_data=(x_test, y_test))



