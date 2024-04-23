import tensorflow as tf
import numpy as np

def buildLSTM(mem_size, input_size, L=2, final_layers=[]):
    # Time x N x fet

    last_hidden_tensor =  tf.keras.layers.Input(shape=(mem_size,), name='h_0')
    last_mem_tensor = tf.keras.layers.Input(shape=(mem_size,), name='c_0')

    inputs = [None] * (L + 2)
    inputs[0] = last_mem_tensor
    inputs[1] = last_hidden_tensor

    # Four gates
    f_layer = tf.keras.layers.Dense(
            mem_size, 
            activation=tf.keras.activations.sigmoid, name=f'forget')
    
    i_layer = tf.keras.layers.Dense(
            mem_size, 
            activation=tf.keras.activations.sigmoid, name=f'input')
    
    o_layer =  tf.keras.layers.Dense(
            mem_size, 
            activation=tf.keras.activations.sigmoid, name=f'ouput')
    
    content_layer =  tf.keras.layers.Dense(
            mem_size, 
            activation=tf.keras.activations.sigmoid, name=f'content')

    for i in range(2, L + 2):
        input_tensor = tf.keras.layers.Input(shape=(input_size,), name=f'x_{i - 1}')
        inputs[i] = input_tensor

        concat_tensor = tf.keras.layers.Concatenate()([input_tensor, last_hidden_tensor])

        f_tensor = f_layer(concat_tensor)
        i_tensor = i_layer(concat_tensor)
        o_tensor = o_layer(concat_tensor)
        content_tensor = content_layer(concat_tensor)

        new_c_tensor = tf.keras.layers.Add()([
            tf.keras.layers.Multiply()([f_tensor, last_mem_tensor]),
            tf.keras.layers.Multiply()([i_tensor, content_tensor])
        ])
        
        new_h_tensor = tf.keras.layers.Multiply()([
                o_tensor,
                tf.keras.activations.sigmoid(new_c_tensor)
            ])
        
        last_mem_tensor = new_c_tensor
        last_hidden_tensor = new_h_tensor

    for layer in final_layers:
        last_hidden_tensor = layer(last_hidden_tensor)

    return tf.keras.Model(inputs=inputs, outputs=last_hidden_tensor)

# Time x N x fet
X = np.array([
    [[1, 3, 2, 5, -4], [4, 0, 0, -4, 5]],
    [[-4, 15, 6, 5, -4], [4, 2, -5, 8, 10]],
    [[1, 3, 20, -4, 22], [4, 6, 22, 15, 10]],
])

Y = np.array([
    [1.0, 0.0],
    [0.0, 1.0]
])

mem_size = 3

H_0 = np.zeros((X.shape[1], mem_size))
C_0 = np.zeros((X.shape[1], mem_size))

model = buildLSTM(mem_size, 5, L=3, final_layers=[
    tf.keras.layers.Dense(
        10,
        activation=tf.keras.activations.softmax)
])

inputs = [(C_0 if i == 0 else (H_0 if i == 1 else X[i - 2])) for i in range(X.shape[0] + 2)]

pred = model.predict(inputs)
model.fit(inputs, Y)