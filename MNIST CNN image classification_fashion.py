# import libraries
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# loading dataset from fashion mnist and splitting dataset for testing and training -- cross validation
fashion_mnist_dataset = keras.datasets.fashion_mnist
(images_train, labels_train), (images_test,
                               labels_test) = fashion_mnist_dataset.load_data()

# note: the images from fashion mnist are 28x28 NumPy arrays with pizel values ranging from 0-255
# the labels are array of integers, ranging from 0-9
class_labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat'
                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# EDA -- optional
# returns the dimensions of your dataset
images_train.shape
# returns the lenght in the labels training set -- Return the length (the number of items) of an object.
len(labels_train)
len(labels_test)

# data preprocessing

plt.figure()  # creates a new figure
plt.imshow(images_train[0])  # display the image on the axes
plt.colorbar()  # add a colorbar to a plot
plt.grid(False)  # turn the axes grids on or off
plt.show()  # display the figure

# scaling the values from 0-255 to 0-1 before feeding into the model
images_train = images_train/255.0
images_test = images_test/255.0

# check to verify data format before feeding in to the model
plt.figure(figsize=(10, 10))  # size of the figure to be projected
for i in range(25):
    # Return a subplot axes positioned by the given grid definition.
    plt.subplot(5, 5, i+1)
    # gets the x-limits of the current tick locaations and labels -- i'm not quite sure how it works [see MATPLOTLIB]
    plt.xticks([])
    # gets the y-limits of the current tick locaations and labels
    plt.yticks([])
    plt.grid(False)
    plt.imshow(images_train[i], cmap=plt.cm.binary)
    plt.xlabel(class_labels[labels_train[i]])
plt.show()

# model specification
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    # transforms the images from 2D to 1D [28*28] array of pixels
    keras.layers.Dense(128, activation='relu'),
    # The first Dense layer has 128 nodes (or neurons).
    keras.layers.Dense(10)
    # The second (and last) layer returns a logits array with length of 10.
])

#  model compilation specifications includes optimization method, loss function, and metrics
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(
                  from_logits=True),
              metrics=['accuracy'])

# train the model
model.fit(images_train, labels_train, epochs=10)

# evaluate using accuracy metrics
test_loss, test_accuracy = model.evaluate(images_test, labels_test, verbose=2)
print('\nTest accuracy: ', test_accuracy)

# note: if the accuracy on the test dataset is a little less than the accuracy of the training dataset
# you might want to check out for overfitting and adjust your model through hyperparemeter tuning
# an over fitted model is not ideal

# make predictions

probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
# since the model's linear outputs logits -- The vector of raw (non-normalized) predictions that a classification model generates,
# which is ordinarily then passed to a normalization function.
# for multiclass classification problems, logits typically become an input of softmax function
predictions = probability_model.predict(images_test)
# Generates output predictions for the input samples.
# note: computation is done in batches

# OPTIONAL -- EDA
predictions[0]
# this will output an array of floating point values which represent the model's confidence
np.argmax(predictions[0])
# returns the indices of the max value along an axis
labels_test[0]
# examining the test label shows that this classification is correct (if they are the same, then prediction is correct)

# for representation and validation -- define new funcitons


def plot_images(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array, true_label[i], img
    plt.grid(False)
    plt.xticks([])
    plt.yticks[]

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{}{:2.0f}% ({})".format(class_labels[predicted_label],
                                100*np.max(predictions_array),
                                class_label[true_label]),
                                color=color))

def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label=predictions_array, true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    this_plot=plt.bar(range(10), predictions_array, color = '#777777')
    plt.ylim([0, 1])
    predicted_label=np.argmax(predictions_array)

    this_plot[predicted_label].set_color('red')
    this_plot[true_label].set_color('blue')

# verify predictions
i=0
plt.figure(figsize = (6, 3))
plt.subplot(1, 2, 1)
plot_images(i, predictions[i], labels_test, images_test)
plt.subplot(1, 2, 2)
plot_value_array(1, predictions[i], labels_test)
plt.show()
