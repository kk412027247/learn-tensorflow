from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

values_x = np.array([-10, 0, 2, 6, 12, 15], dtype=float)
values_y = np.array([10, 30, 34, 42, 54, 60], dtype=float)
for i, x in enumerate(values_x):
    print("X: {} Y: {}".format(x, values_y[i]))

x = np.linspace(-10, 10, 100)
plt.title('Graph of y=2x+30')
plt.plot(x, x * 2 + 30);

model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1])
])
model.compile(loss='mean_squared_error',
              optimizer=tf.keras.optimizers.Adam(0.1))

history = model.fit(values_x, values_y, epochs=500, verbose=False)

plt.xlabel("Epoch Number")
plt.ylabel("Loss Magnidute")
plt.plot(history.history['loss'])

print(model.predict([20.0]))

print("These are the layer variables: {}".format(model.layers[0].get_weights()))
