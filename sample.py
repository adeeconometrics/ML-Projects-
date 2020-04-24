import tensorflow as tf


def generate_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, filter_size=3, activation='relu'),
        tf.keras.MaxPool2D(pool_size=2, strides=2),

        # second convolutional layer
        tf.keras.layers.Conv2D(62, filter_size=3, activation='relu'),
        tf.keras.MaxPool2D(pool_size=2, strides=2),

        # fully connected classifier
        tf.keras.layers.Flatten(),
        tf.keras.Dense(1024, activation='relu'),
        tf.keras.Dense(10, activation='softmax')
    ])
    return model
