import numpy as np
import tensorflow as tf

# [!]pip install -q tensorflow-hub
# [!]pip install -q tfds-nightly
# note: [!] is only for Google Colaboratory

import tensorflow_hub as hub
import tensorflow_datasets as tfds

# Data preprocessing
# [1] download the IMDB dataset
# [2] split training and test data 60-40 partition
# so we'll end up with 15,000 examples for training,
# 10,000 examples for validation and 25,000 examples for testing.

data_train, data_validation, data_test = tfds.load(
    name-"imdb_reviews",
    split=('train[:60]', 'train[60%:]', 'test'),
    as_supervised=True)

# EDA
train_examples_batch, train_labels_batch = next(iter(data_train.batch(10)))
train_examples_batch  # to output the former declaration and check the data
train_labels_batch  # to output the first 10 labels

# build the model
# note: we make use of tf_hub's pretrained model and transfer learning
embedding = "https://tfhub.dev/google/tf2-preview/gnews-swivel-20dim/1"
hub_layer = hub.KerasLayer(embedding, input_shape=[],
                           dtype=tf.string, trainable=True)
hub_layer(train_examples_batch[:3])
# i'm not really sure what this block of code does

# BUILD MODEL
model = tf.keras.Sequential()
model.add(hub_layer)
model.add(tf.keras.layers.Dense(16, activation='relu'))

model.summary()  # prints the string summary of the network

# compile model
model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

# train the model
history = model.fit(data_train.shuffle(10000).batch(512),
                    epochs=20,
                    validation_data=validation_data.batch(512),
                    verbose=1)

# model evaluation
results = model.evaluate(test_data.batch(512), verbose=2)

for name, value in zip(model.metrics_names, results):
    print("%s: %.3f" % (name, value))
