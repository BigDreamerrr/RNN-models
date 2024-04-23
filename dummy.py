import numpy as np
from RNN import RNN
import tensorflow as tf

X = np.array([
    [[1, 3, 2, 5, -4], [4, 0, 0, -4, 5]],
    [[-4, 15, 6, 5, -4], [4, 2, -5, 8, 10]],
    [[1, 3, 20, -4, 22], [4, 6, 22, 15, 10]],
])

Y = np.array([
    [1.0, 0.0],
    [0.0, 1.0]
])

H_0 = np.zeros((X.shape[1], 5))

model = RNN()

model.fit(X, Y, 'categorical_crossentropy', final_layers=[
    tf.keras.layers.Dense(2, activation=tf.keras.activations.sigmoid, name='last_layer')
    ])

pred = model.predict(X)