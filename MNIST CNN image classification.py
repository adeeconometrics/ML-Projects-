import tensorflow as tf

# importing dataset
mnist = tf.keras.datasets.mnist

# split data to train_data and test_data -- for model evaluation
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# convert integers to floating-point numbers
x_train, x_test = x_train/255.0, x_test/255.0

# Sequential model specifications
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10)
])

# i'm not sure about what this block of code means
predictions = model(x_train[:1]).numpy()
tf.nn.softmax(predictions).numpy()

# loss function specifications
loss_func = tf.keras.losses.sparse_categorical_crossentropy(from_logits=True)
loss_fucn(y_train[:1], predictions).numpy()

# model compilation specifications includes optimization method, loss function, and metrics
model.compile(optimizer='adam', loss=loss_func, metrics=['accuracy'])

# model training specifications
model.fit(x_train, y_train, epochs=5)

# model evaluation -- drawn from x_test, and y_test dataset
# Returns the loss value & metrics values for the model in test mode.
# Computation is done in batches.

model.evaluate(x_test, y_test, verbose=2)  # i'm not sure what 'verbose' means
