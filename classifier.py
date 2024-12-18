import librosa
import os
import numpy as np
from tensorflow.image import resize
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

def melspectogram(audio_file):
    target_shape=(128, 128)
    y, sr = librosa.load(audio_file, sr=None)
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_spectrogram = resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
    return mel_spectrogram

def mfcc_converter(audio_file):
    target_shape=(128, 128)
    y, sr = librosa.load(audio_file, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=3)
    mfcc = resize(np.expand_dims(mfcc, axis=-1), target_shape)
    return mfcc

def load_data(dir):
    data = []
    labels = []

    for file in os.listdir(dir):
        if file.endswith(".wav"):
            audio_file = os.path.join(dir, file)
            mfcc = mfcc_converter(audio_file)
            if "idle" in file:
                labels.append(0)
            else:
                labels.append(1)
            data.append(mfcc)
            
    return np.array(data), to_categorical(np.array(labels),2)

X_train, y_train = load_data("training_data")
X_test, y_test = load_data("testing_data")

input_shape = X_train[0].shape
input_layer = Input(shape=input_shape)
x = Conv2D(32, (3, 3), activation='relu')(input_layer)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Flatten()(x)
x = Dense(64, activation='relu')(x)
output_layer = Dense(2, activation='softmax')(x)
model = Model(input_layer, output_layer)

model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=4, batch_size=32, validation_data=(X_test, y_test))
test_accuracy=model.evaluate(X_test,y_test,verbose=0)
print(test_accuracy[1])