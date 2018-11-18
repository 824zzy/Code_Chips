# -*- coding: utf-8 -*-
import re
import codecs
import numpy as np
from keras.layers import *
from keras.models import Model
from keras import backend as K
from keras.engine.topology import Layer
from keras.callbacks import Callback
import tensorflow as tf

n = 5 # only abstract 5 words poem
latent_dim = 64
hidden_dim = 64

s = codecs.open('poems.txt', encoding='utf-8').read()
s = re.findall(u'　　(.{%s}，.{%s}。.*?)\r\n'%(n,n), s) # find poems via regular expression
poem = []
for i in s:
    for j in i.split(u'。'):
        if j:
            poem.append(j)

poem = [i[:n] + i[n+1:] for i in poem if len(i) == 2*n+1] # drop comma

id2char = dict(enumerate(set(''.join(poem))))
char2id = {j:i for i,j in id2char.items()}

poem2id = [[char2id[j] for j in i] for i in poem]
poem2id = np.array(poem2id)

class GCNN(Layer): # GCNN with residual
    def __init__(self, output_dim=None, residual=False, **kwargs):
        super(GCNN, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.residual = residual
    def build(self, input_shape):
        if self.output_dim == None:
            self.output_dim = input_shape[-1]
        self.kernel = self.add_weight(name='gcnn_kernel',
                                      shape=(3, input_shape[-1],
                                             self.output_dim * 2),
                                      initializer='glorot_uniform',
                                      trainable=True)
    def call(self, x):
        _ = K.conv1d(x, self.kernel, padding='same')
        _ = _[:, :, :self.output_dim] * K.sigmoid(_[:, :, self.output_dim:])
        if self.residual:
            return _ + x
        else:
            return _

input_sentence = Input(shape=(2*n,), dtype='int32')
input_vec = Embedding(len(char2id), hidden_dim)(input_sentence)
h = GCNN(residual=True)(input_vec)
h = GCNN(residual=True)(h)
h = GlobalAveragePooling1D()(h)

z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)

def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim), mean=0, stddev=1)
    return z_mean + K.exp(z_log_var / 2) * epsilon

z = Lambda(sampling)([z_mean, z_log_var])

decoder_hidden = Dense(hidden_dim * (2 * n))
decoder_cnn = GCNN(residual=True)
decoder_dense = Dense(len(char2id), activation='softmax')

h = decoder_hidden(z)
h = Reshape((2*n, hidden_dim))(h)
h = decoder_cnn(h)
output = decoder_dense(h)

# build model
vae = Model(input_sentence, output)

xent_loss = K.sum(K.sparse_categorical_crossentropy(input_sentence, output), 1)
kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
vae_loss = K.mean(xent_loss + kl_loss)

vae.add_loss(vae_loss)
vae.compile(optimizer='adam')
vae.summary()

decoder_input = Input(shape=(latent_dim,))
_ = decoder_hidden(decoder_input)
_ = Reshape((2*n, hidden_dim))(_)
_ = decoder_cnn(_)
_output = decoder_dense(_)
generator = Model(decoder_input, _output)

def gen():
    r = generator.predict(np.random.randn(1, latent_dim))[0]
    r = r.argmax(axis=1)
    return ''.join([id2char[i] for i in r[:n]])\
           + u'，'\
           + ''.join([id2char[i] for i in r[n:]])

class Evalueate(Callback):
    def __init__(self):
        self.log = []
    def on_epoch_end(self, epoch, logs=None):
        self.log.append(gen())
        print (u'          %s'%(self.log[-1])).encode('utf-8')

evaluator = Evalueate()

vae.fit(poem2id,
        shuffle=True,
        epochs=3,
        batch_size=64,
        callbacks=[evaluator])

vae.save_weights('poem.model')

for i in range(20):
    print gen()
