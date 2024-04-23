import numpy as np
import h5py
import tensorflow as tf

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
            activation=tf.keras.activations.tanh, name=f'content')
    
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
            tf.keras.layers.Multiply()([i_tensor, content_tensor]) #??
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

def buildRNN(hidden_state_size, input_size, L=2, final_layers=[]):
    inputs = [None] * (L + 1)

    # h_0 = zero vector
    inputs[0] = tf.keras.layers.Input(shape=(hidden_state_size,), name='h_0')

    last_output_layer = inputs[0]
    feed_back_layer = tf.keras.layers.Dense(
            hidden_state_size, 
            activation=tf.keras.activations.sigmoid, name=f'feedback')
    
    for i in range(1, L + 1):
        input_layer = tf.keras.layers.Input(shape=(input_size,), name=f'i_{i}')
        inputs[i] = input_layer
        concat_layer = tf.keras.layers.Concatenate()([input_layer, last_output_layer])
        last_output_layer = feed_back_layer(concat_layer)

    for layer in final_layers:
        last_output_layer = layer(last_output_layer)

    return tf.keras.Model(inputs=inputs, outputs=last_output_layer)

class CustomModel:
    def build_model(self, **kwargs):
        pass

    def build_inputs(self, X):
        pass

    def fit(self, X, Y, loss, epochs=10, **kwargs):
        # Time x N x fet
        L = X.shape[0]

        kwargs['L'] = L
        kwargs['input_size'] = X.shape[2]

        model = self.build_model(**kwargs)

        model.compile(
            optimizer='adam',
            loss=loss,
            metrics=['accuracy'])
        
        self.kwargs = kwargs
        inputs = self.build_inputs(X)
        model.fit(inputs, Y, epochs=epochs)

        self.model = model

    def predict(self, X):
        return self.model.predict(self.build_inputs(X))
    
    def save(self, path):
        self.model.save(path)

class RNN(CustomModel):
    def build_model(self, **kwargs):
        hidden_state_size = kwargs.get('hidden_state_size')

        if hidden_state_size== None:
            hidden_state_size = kwargs['input_size']

        return buildRNN(
            hidden_state_size, 
            kwargs['input_size'], 
            kwargs['L'],
            kwargs['final_layers'])
    
    def build_inputs(self, X):
        hidden_state_size = self.kwargs.get('hidden_state_size')

        if hidden_state_size== None:
            hidden_state_size = self.kwargs['input_size']

        H_0 = np.zeros((X.shape[1], hidden_state_size))

        return [(H_0 if i == 0 else X[i - 1]) for i in range(X.shape[0] + 1)]
    
class LSTM(CustomModel):
    def build_model(self, **kwargs):
        return buildLSTM(
            kwargs['mem_size'],
            kwargs['input_size'],
            kwargs['L'],
            kwargs['final_layers'])
    
    def build_inputs(self, X):
        H_0 = np.zeros((X.shape[1], self.kwargs['mem_size']))
        C_0 = np.zeros((X.shape[1], self.kwargs['mem_size']))

        return [(C_0 if i == 0 else (H_0 if i == 1 else X[i - 2])) for i in range(X.shape[0] + 2)]