from sklearn.preprocessing import OneHotEncoder
from get_actions import get_data
import tensorflow as tf

train_data, train_labels, val_data, val_labels, test_data = get_data()

train_data = train_data.reshape((train_data.shape[0], train_data[0].size))
val_data = val_data.reshape((val_data.shape[0], val_data[0].size))

encoder = OneHotEncoder()
train_labels_encoded = encoder.fit_transform(train_labels.reshape(train_labels.shape[0], 1)).toarray()

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(
                    128, 
                  activation=tf.keras.activations.relu))

model.add(tf.keras.layers.Dropout(0.2))

model.add(tf.keras.layers.Dense(
                  10, 
                  activation=tf.keras.activations.softmax))

model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy'])

model.fit(train_data, train_labels_encoded, epochs=60)

pass