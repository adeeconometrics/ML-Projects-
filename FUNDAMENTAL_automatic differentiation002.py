import tensorflow as tf
import numpy as np

x = tf.constant(3.0)

with tf.GradientTape(persistent=True) as tape:
    tape.watch(x)
    y = x*x
    z = y*y

dz_dx = tape.gradient(z, x)
dz_dy = tape.gradient(z, y)
del tape  # drop the reference to the tape

# recording control flow


def function(x, y):
    output = 1.0
    for i in range(y):
        if i > 1 and i < 5:
            output = tf.multiply(output, x)
        return output


def gradient(x, y):
    with tf.GradientTape() as tape:
        tape.watch(x)
        out = function(x, y)
    return tape.gradient(out, x)


x = tf.convert_to_tensor(2.0)  # converts Python objects to tensors

assert gradient(x, 6).numpy() == 12.0
assert gradient(x, 5).numpy() == 12.0
assert gradient(x, 4).numpy() == 4.0
