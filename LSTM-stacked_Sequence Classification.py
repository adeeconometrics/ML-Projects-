from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np

data_dim = 16
timesteps = 8
class_num = 10

model = Sequential()
model.add(LSTM(32, return_sequences=True, input_shape=(timesteps, data_dim)))
model.add(LSTM(32, return_sequences=True))
model.add(LSTM(32))
model.add(Dense(10, activation='softmax'))
# note return_sequences is a boolean argument -- Whether to return the last output
# in the output sequence, or the full sequence.

model.compile(loss='categorical_crossentropy',
              oprimizer='rmsprop', metrics=['accuracy'])

# dummy data generation
x_train = np.random.ranodom((1000, timesteps, data_dim))
y_train = np.random.random((1000, class_num))

# dummy data for validation dataset
x_test = np.random.random((100, timesteps, data_dim))
y_test = np.random.random((100, class_num))

model.fit(x_train, y_train, batch_size=64, epochs=5,
          validation_data=(x_test, y_test))
# note: validation_data are a tuple of arguments of which the loss function and any model
# metrics will evaluate at the end of each epoch
