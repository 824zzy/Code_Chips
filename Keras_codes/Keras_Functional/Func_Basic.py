from keras.layers import Input, Dense
from keras.models import Model

inputs = Input(shape(784,))

x = Dense(64, activation='relu')(inputs)
x = Dense(64, acivation='relu')(x)
predictions = Dense(10, activation='softmax')(x)

model = Model(inputs=inputs, outputs=predictions)
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
# model.fit(data, labels)


