import tensorflow as tf
import numpy as np

x = tf.Variable(1.0)

with tf.GradientTape() as tape_01:
    with tf.GradientTape() as tape_02:
        y = x*x*x

    dy_dx = tape_02.gradient(y, x)

d2y_d2x = tape_01.gradient(dy_dx, x)

assert dy_dx.numpy() == 3.0
assert d2y_d2x.numpy() == 6.0
