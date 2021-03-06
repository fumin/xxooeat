import numpy as np
import tensorflow as tf

data = np.genfromtxt("/Users/mac/fumin/tmp/xxooeat/data", delimiter=",")
x = data[:,:-1]
y = data[:,-1:]

y *= 10

model = tf.keras.Sequential([
  tf.keras.layers.Dense(128, activation="tanh"),
  tf.keras.layers.Dense(128, activation="tanh"),
  tf.keras.layers.Dense(1)
])
model.compile(loss = tf.losses.MeanSquaredError(),
              optimizer = tf.optimizers.Adam())
model.fit(x, y, epochs=100)
