import tensorflow as tf
import os

def MY_LSTM():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 1662)))
    model.add(tf.keras.layers.LSTM(128, return_sequences=True, activation='relu'))  # return_sequences=True,
    model.add(tf.keras.layers.LSTM(64, return_sequences=False, activation='relu'))  # return_sequences=False,
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(32, activation='relu'))
    model.add(tf.keras.layers.Dense(3, activation='softmax'))
    dirname = os.path.dirname(__file__)
    filename = os.path.join(dirname, "models/action.h5")
    model.load_weights(filename)
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

    return model
