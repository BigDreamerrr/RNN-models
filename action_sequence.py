from sklearn.preprocessing import LabelEncoder
from RNN import RNN
import h5py
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
import tensorflow as tf
import numpy as np
import pandas as pd

def encode_hf5_file(path):
    hf = h5py.File(path, 'r')

    data_name = None
    for name in hf.keys():
        data_name = name

    data = hf[data_name][:]
    hf.close()
    
    return data

train_data = encode_hf5_file(r"D:\Dataset\cse512springhw5\train_data.h5")
train_labels = encode_hf5_file(r"D:\Dataset\cse512springhw5\train_label.h5")

val_data = encode_hf5_file(r"D:\Dataset\cse512springhw5\val_data.h5")
val_labels = encode_hf5_file(r"D:\Dataset\cse512springhw5\val_label.h5")

test_data = encode_hf5_file(r"D:\Dataset\cse512springhw5\test_data.h5")

train_data = np.swapaxes(train_data, 0, 1)
val_data = np.swapaxes(val_data, 0, 1)
test_data = np.swapaxes(test_data, 0, 1)

model = RNN()

encoder = OneHotEncoder()
train_labels_encoded = encoder.fit_transform(train_labels.reshape(train_labels.shape[0], 1)).toarray()

model.fit(train_data, train_labels_encoded, loss='categorical_crossentropy', 
          final_layers=[
            #   tf.keras.layers.Dense(
            #       32,
            #       activation=tf.keras.activations.relu,
            #   ),
              tf.keras.layers.Dense(
                  10, 
                  activation=tf.keras.activations.softmax, 
                  name='last_layer')
          ], 
          epochs=180, hidden_state_size=256)

pred = model.predict(train_data)
pred = pred.argmax(axis=-1)

print(f"On train: {accuracy_score(pred, train_labels)}")

val_pred = model.predict(val_data).argmax(axis=-1)

print(f"On val: {accuracy_score(val_pred, val_labels)}")

test_pred = model.predict(test_data).argmax(axis=-1)

ans_dict = { 'Id' : [], 'Class' : [] }

for i in range(len(test_pred)):
    ans_dict['Id'].append(i)
    ans_dict['Class'].append(test_pred[i])

df2 = pd.DataFrame.from_dict(ans_dict)
df2.to_csv(r'ans.csv', index=False)

model.save('model.keras')