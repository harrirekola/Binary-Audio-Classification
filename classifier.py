import librosa
import os
import numpy as np
from tensorflow.image import resize
from sklearn.model_selection import train_test_split

def melspectogram(audio_file):
    target_shape=(128, 128)
    y, sr = librosa.load(audio_file, sr=None)
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_spectrogram = resize(np.expand_dims(mel_spectrogram, axis=-1), target_shape)
    return mel_spectrogram

def load_data(dir):
    data = []
    labels = []

    for file in os.listdir(dir):
        if file.endswith(".wav"):
            audio_file = os.path.join(dir, file)
            mel_spectrogram = melspectogram(audio_file)
            if "idle" in file:
                labels.append(0)
            else:
                labels.append(1)
            data.append(mel_spectrogram)
    return np.array(data), np.array(labels)

X_train, y_train = load_data("training_data")
X_test, y_test = load_data("testing_data")