from keras.layers import LSTM,Input,Embedding,Dense
from keras.models import Model

main_input = Input(shape=(100,), dtype='int32', name='main_input')
x = Embedding(output_dim=512, input_dim=10000, input_length=100)(main_input)
lstm_out = LSTM(32)(x)

aux_output = Dense(1, activation='sigmoid', name='aux_output')(lstm_out)

aux_input = Input(shape=(5,), name='aux_input')
x = keras.layers.concatenate([lstm_out, aux_input])

x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)
x = Dense(64, activation='relu')(x)

main_output = Dense(1, activation='simoid', name="main_output")(x)


model = Model(inputs=[main_input, aux_input], outputs=[main_output, aux_output])
model.compile(optimizer='rmsprop', loss='binary_crossentropy', loss_weights=[1., 0.2])

model.fit([headline_data, additional_data], [labels, labels],
          epochs=5, batch_size=32)
