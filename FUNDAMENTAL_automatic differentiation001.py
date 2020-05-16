import tensorflow as tf
import numpy as np

x = tf.ones((2, 2))

with tf.GradientTape() as tape:
    tape.watch(x)  # ensures that 'tensors' is being traced by the tape
    y = tf.reduce_sum(x)  # computes the sum of the tensor
    z = tf.multiply(y, y)  # computes the element-wise product of tensors

# computes the gradient using operations recorded in context of this shape
dz_dx = tape.gradient(z, x)
# computes the derivative of z with respect to the original input tensor x

for i in [0, 1]:
    for j in [0, 1]:
        # to traverse indices across the 2x2 matrix
        assert dz_dx[i][j].numpy() == 8.0
        # explicitly converts a tensor to a numpy array
        # note: assert keyword acts as a boolean flag of which the program returns
        # nothing when the condition set is true else raise an AssertionError

# NOTE: only real or complex datatypes (dtypes) are differentiable
