# libraries
from dataprep import 
import pandas as pd
import numpy as np
import tensorflow as tf
import keras
from tensorflow.keras import models,layers
from keras.layers import Conv2D, MaxPool2D, Dense, Dropout, Flatten
from keras.models import Sequential
from tensorflow.keras import optimizers

def data_load_labels(data, labels):
    data_df = pd.read_csv(data, header=None)
    X = data_df.values.reshape((-1, 28, 28, 4)).clip(0, 255).astype(np.uint8)
    labels_df = pd.read_csv(labels, header=None)
    y = labels_df.values.getfield(dtype=np.int8)
    return X, y

X_train, y_train = data_load_labels(data='C:/Users/19405/Desktop/Capstone 2/X_train_sat6.csv',
                                    labels='C:/Users/19405/Desktop/Capstone 2/y_train_sat6.csv')

X_test, y_test = data_load_labels(data='C:/Users/19405/Desktop/Capstone 2/X_test_sat6.csv',
                                    labels='C:/Users/19405/Desktop/Capstone 2/y_test_sat6.csv')


print("Training data shape ", X_train.shape, y_train.shape)
print("Testing data shape ", X_test.shape, y_test.shape)

# Base model
model = Sequential()
model.add(Conv2D(16, (3,3), activation='relu', input_shape=(28,28,4)))
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.5))
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(6, activation='softmax'))
# model summary
print(model.summary())
# early stopping
from keras.callbacks import EarlyStopping
early_stop_mon=EarlyStopping(patience=3)
model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=546, epochs=30, verbose=1, validation_data=(X_test, y_test), callbacks=[early_stop_mon])

# Plot model fitting
import matplotlib.pyplot as plt
# adapted from Deep Learning With Python (Chollet)
h=model.history.history
epochs = range(len(h['accuracy']))
fig,ax=plt.subplots(1,2,figsize=(13,5))
ax[0].plot(epochs, h['accuracy'], 'r', label='Training')
ax[0].plot(epochs, h['val_accuracy'], 'b', label='Validation')
ax[0].set_title('Accuracy')
ax[0].set_xlabel('Epochs')
ax[0].legend()
ax[1].plot(epochs, h['loss'], 'r', label='Training')
ax[1].plot(epochs, h['val_loss'], 'b', label='Validation')
ax[1].set_title('Loss')
ax[1].set_xlabel('Epochs')
ax[1].legend()
plt.suptitle('Evaluation of Model Fit Over {} Epochs'.format(len(h['accuracy'])))