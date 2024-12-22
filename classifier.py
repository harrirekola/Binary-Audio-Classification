import librosa
import os
import numpy as np
from tensorflow.image import resize
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report

def melspectogram(audio_file):
    target_shape = (128, 128)
    sr_desired = 44100
    duration_seconds = 5
    n_samples = sr_desired * duration_seconds

    y, sr = librosa.load(audio_file, sr=sr_desired)

    if len(y) < n_samples:
        y = np.pad(y, (0, n_samples - len(y)), mode='constant')
    y = y[:n_samples]

    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)

    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    mel_spec_db = np.expand_dims(mel_spec_db, axis=-1)
    mel_spec_db = resize(mel_spec_db, target_shape)

    return mel_spec_db

def mfcc_converter_from_array(y, sr):
    target_shape = (128, 128)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    # Expand dims
    mfcc = np.expand_dims(mfcc, axis=-1)
    # Resize
    mfcc = resize(mfcc, target_shape)
    return mfcc

def load_data(data_dir):
    data = []
    labels = []
    sr_desired = 44100
    duration_seconds = 5
    n_samples = sr_desired * duration_seconds

    for filename in os.listdir(data_dir):
        if filename.endswith(".wav"):
            filepath = os.path.join(data_dir, filename)
            y, sr = librosa.load(filepath, sr=sr_desired)

            # pad if signal is shorter than 5s
            if len(y) < n_samples:
                y = np.pad(y, (0, n_samples - len(y)), mode='constant')
            # truncate if signal is longer than 5s
            y = y[:n_samples]

            mfcc = mfcc_converter_from_array(y, sr)

            # Append label
            if "idle" in filename.lower():
                labels.append(0)
            else:
                labels.append(1)

            data.append(mfcc)

    return np.array(data), to_categorical(np.array(labels), 2)


X_train, y_train = load_data("training_data")
X_test, y_test = load_data("testing_data")

mean = np.mean(X_train, axis=(0,1,2), keepdims=True)
std = np.std(X_train, axis=(0,1,2), keepdims=True)
X_train = (X_train - mean) / (std + 1e-7)
X_test = (X_test - mean) / (std + 1e-7)

input_shape = X_train[0].shape
input_layer = Input(shape=input_shape)
x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
x = MaxPooling2D((2, 2))(x)
x = Dropout(0.25)(x)

x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2))(x)
x = Dropout(0.25)(x)

x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2))(x)
x = Dropout(0.25)(x)

x = Flatten()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
output_layer = Dense(2, activation='softmax')(x)
model = Model(inputs=input_layer, outputs=output_layer)

model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)

model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=50,
    batch_size=32,
    callbacks=[early_stop, reduce_lr]
)

#model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
test_accuracy=model.evaluate(X_test, y_test, verbose=0)
print(test_accuracy[1])