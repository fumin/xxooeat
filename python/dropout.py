import numpy as np
import tensorflow as tf

def getDatum():
	x = np.random.randint(2, size=4)
	y = (bool(x[0]) or bool(x[1])) and (bool(x[2]) != bool(x[3]))
	return x, y

dataSize = 10000
x = np.zeros([dataSize, 4])
y = np.zeros(dataSize)
for i in range(dataSize):
	x[i], y[i] = getDatum()
print(x[:3], y[:3])

model = tf.keras.Sequential([
  tf.keras.layers.Dense(32, activation="relu"),
  tf.keras.layers.Dropout(0.5),
  tf.keras.layers.Dense(32, activation="relu"),
  tf.keras.layers.Dropout(0.5),
  tf.keras.layers.Dense(1)
])
model.compile(loss = tf.losses.MeanSquaredError(),
              optimizer = tf.optimizers.Adam(),
              metrics=[tf.metrics.BinaryAccuracy()])
model.fit(x, y, epochs=100)
