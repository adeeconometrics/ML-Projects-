# word embedding
from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras import preprocessing
from keras.datasets import imdb
from keras import Embedding

embedding_layer = Embedding(1000, 64)
# embedding layer takes at least two arguments:
# the number of possible tokens
# and the dimensionality of the embeddings
# note that embedding_layer can be best understood
# as a dictionaty that maps integer indices to dense vectors

# loading IMDB data for use with an Embedding layer

max_features = 1000
max_lenght = 20

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)

x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=max_length)
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=max_lenght)
# pad_sequences pads sequences to the same length
# Numpy array with shape `(len(sequences), maxlen)`


# using an embedding layer and classifier on the IMDB data

model = Sequential()
model.add(Embedding(max_features, 8, input_length=max_lenght))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
# model.summary()

history = model.fit(x_train,
                    y_train,
                    epochs=10,
                    batch_size=32,
                    validation_split=0.2)  # i'm not sure what validation_split defined at dtype=float32
