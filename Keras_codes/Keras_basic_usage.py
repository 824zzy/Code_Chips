import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense,Activation
import numpy as np

model = Sequential()
model.add(Dense(units=64, activation='relu', input_dim=100))
model.add(Dense(units=10, activation='softmax'))

model.compile(loss="categorical_crossentropy",
              optimizer="rmsprop",
              metrics=["accuracy"])

x_train = np.random.random((1000, 100))
y_train = np.random.randint(10, size=(1000, 1))
one_hot_train = keras.utils.to_categorical(y_train, num_classes=10)

model.fit(x_train, one_hot_train, epochs=5, batch_size=32)
  

