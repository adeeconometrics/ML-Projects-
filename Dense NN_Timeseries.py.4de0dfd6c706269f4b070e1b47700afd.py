import os
import numpy as np
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop


data_dir = ''
fname = os.path.join(data_dir, 'jena_climate_2009_2016.csv')

f = open(fname)  # open a file on a stream
data = f.read()
f.close()

lines = data.split('\n')
header = lines[0].split(',')
# initialization
float_data = np.zeros((len(lines), len(header)-1))
for i, line in enumerate(lines):
    # i'm not sure what this means
    values = [float(x) for x in line.split(',')[1:]]
    float_data[i, :] = values

# normalizing the data -- only for the training set
# computes mean along the specified axis
mean = float_data[:200000].mean(axis=0)
float_data -= mean
std = float_data[:200000].std(axis=0)  # computes std along the specified axis
# what happens to a distribution when it is scaled to a std of its subportion?
float_data /= std

# data generator


def data_generator(data, lookback, delay, min_index, max_index,
                   shuffle=False, batch_size=128, step=6):

    if max_index is None:
        max_index = len(data)-delay-1

    i = min_index + lookback

    while 1:
        if shuffle:
            rows = np.random.randint(
                min_index+lookback, max_index, size=batch_size)
        else:
            if i+batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i+batch_size, max_index))
            i += len(rows)

        samples = np.zeros(len(rows),
                           lookback//step,  # results to a whole number
                           data.shape[-1])

        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][1]

        yield samples, targets


# variable declaration
lookback = 1440
step = 6
delay = 144
batch_size = 128

train_gen = data_generator(float_data,
                           lookback=lookback,
                           delay=delay,
                           min_index=0,
                           max_index=200000,
                           shuffle=True,
                           step=step,
                           batch_size=batch_size)

validation_gen = data_generator(float_data,
                                lookback=lookback,
                                delay=delay,
                                min_index=200001,
                                max_index=300000,
                                step=step,
                                batch_size=batch_size)

test_generator = data_generator(float_data,
                                lookback=lookback,
                                delay=delay,
                                min_index=300001,
                                max_index=None,
                                step=step,
                                batch_size=batch_size)

validation_steps = (300000-200001-lookback)
test_steps = (len(float_data)-300001-lookback)


def evaluate_MAE():
    batch_maes = []
    for step in range(validation_steps):
        samples, targets = next(validation_gen)
        preds = samples[:, -1, 1]
        mae = np.mean(np.abs(preds-targets))
        batch_maes.append(mae)

    print(np.mean(batch_maes))


evaluate_MAE()

# naive NN model

model = keras.Sequential()
model.add(layers.Flatten(input_shape=(lookback//step, float_data.shape[-1])))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(1))

model.compile(optimizers=RMSprop(), loss='mae')

history = model.fit_generator(train_gen,
                              steps_per_epoch=500,
                              epochs=20,
                              validation_data=validation_gen,
                              validation_steps=validation_steps)

# Visualize
# loss = history.history['loss']
# validation_loss = history.history['validation_loss']

# epochs = range(1, len(loss)+1)

# plt.plot(epochs, loss, 'bo', label='Training loss')
# plt.plot(epochs, validation_loss, 'b', label='Validation loss')
# plt.legend()
# plt.show()
