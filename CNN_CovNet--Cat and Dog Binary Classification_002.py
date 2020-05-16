# VGG-CNN Model with data augmentation
# data preprocessing
from keras.preprocessing.image import ImageDataGenerator
from keras import models
from keras import layers
import os
import shutil
# make file and define sets for validation set, and test set
original_dataset_dir = ''  # set the native filepath foe loading your data
base_dir = ' '  # set the base directory
os.mkdir(base_dir)
# joins two or more filenames separated by '/' making a subdirectory
train_dir = os.path.join(base_dir, 'train')
os.mkdr(train_dir)  # create new directory
validation_dir = os.path.join(base_dir, 'validation')
os.mkdir(validation_dir)
test_dir = os.path.join(base_dir, 'test')
os.mkdir(test_dir)
# directory for training set on cat images
cats_train_dir = os.path.join(train_dir, 'cats')
os.mkdir(cats_train_dir)
# directory for training set on dog images
dogs_train_dir = os.path.join(train_dir, 'dogs_train_dir')
os.mkdir(dogs_train_dir)
# directory for validation set on cat images
cats_validation_dir = os.path.join(validation_dir, 'cats_validation_dir')
os.mkdir(cats_validation_dir)
# directory for validation set on dog images
dogs_validation_dir = os.path.join(validation_dir, 'dogs_validation_dir')
os.mkdir(dogs_validation_dir)
# directory for test set on cat images
cats_test_dir = os.path.join(test_dir, 'cats_test_dir')
os.mkdir(cats_test_dir)
# directory for test set on dog images
dogs_test_dir = os.path.join(test_dir, 'dogs_test_dir')
os.mkdir(dogs_test_dir)

# NOTE:
# Perform a string formatting operation. The string on which this method is called can contain literal text or
# replacement fields delimited by braces {}. Each replacement field contains either
# the numeric index of a positional argument, or the name of a keyword argument.

# copies the first 1000 images to cats_train_dir
fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    source = os.path.join(original_dataset_dir, fname)
    destination = os.path.join(cats_train_dir, fname)
    shutil.copyfile(source, destination)
    # Utility functions for copying and archiving files and directory trees.
# copies the next 500 images to cats_validation_dir
fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    source = os.path.join(original_dataset_dir, fname)
    destination = os.path.join(cats_validation_dir, fname)
    shutil.copyfile(source, destination)
# copies the next 500 images to cats_tests_dir
fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    source = os.path.join(original_dataset_dir, fname)
    destination = os.path.join(cats_test_dir, fname)
    shutil.copyfile(source, destination)

fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    source = os.path.join(original_dataset_dir, fname)
    destination = os.path.join(dogs_train_dir, fname)
    shutil.copyfile(source, destination)

fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
for fname in fnames:
    source = os.path.join(original_dataset_dir, fname)
    destination = os.path.join(dogs_validation_dir, fname)
    shutil.copyfile(source, destination)

fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
for fname in fnames:
    source = os.path.join(original_dataset_dir, fname)
    destination = os.path.join(dogs_test_dir, fname)
    shutil.copyfile(source, destination)


train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150)
    batch_size=20,
    class_mode='binary')

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150)
    batch_size=20,
    class_mode='binary'

# setting up data augmentation configuration
# transformations that yeild to believable-looking images
data_generated=ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')


# BUILDING THE NETWORK

model=models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3,), activation='relu')
model.add(layers.MaxPooling2D((2, 2))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# model.summary() -- if you want to visualize the model summary and check for the number of parameters

from keras import optimizers

model.compile(loss='binary_crossentropy',
            optimizer=optimizers.RMSprop(lr=1e-4),
            metrics=['acc'])

# train covnet using data-augmentation generators
# i'm not quite sure why i put it here again
train_datagen=ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
)

history=model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=50
)

# Save your model -- model.save('cats_and_dogs_small_1.h5)

# Displaying curves of loss and accuracy during training

import matplotlib.pyplot as plt

acc=history.history['acc']
val_acc=history.history['val_acc']
loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(1, len(acc)+1)

plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and Validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, loss, 'b', label='Validation loss')
plt.title('Training and Valication loss')
plt.legend()

plt.show()
