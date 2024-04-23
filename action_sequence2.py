from sklearn.preprocessing import LabelEncoder
from RNN import RNN, LSTM, buildLSTM
import h5py
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
import tensorflow as tf
import numpy as np
import pandas as pd
from get_actions import get_data

model = LSTM()

train_data, train_labels, val_data, val_labels, test_data = get_data()

train_data = np.swapaxes(train_data, 0, 1)
val_data = np.swapaxes(val_data, 0, 1)
test_data = np.swapaxes(test_data, 0, 1)

encoder = OneHotEncoder()
train_labels_encoded = encoder.fit_transform(train_labels.reshape(train_labels.shape[0], 1)).toarray()

model.fit(train_data, train_labels_encoded, loss='categorical_crossentropy', 
          final_layers=[
                tf.keras.layers.Dense(
                    64, 
                  activation=tf.keras.activations.relu),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(
                  10, 
                  activation=tf.keras.activations.softmax)
          ], 
          epochs=60, mem_size=256)

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