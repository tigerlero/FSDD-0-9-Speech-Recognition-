import glob
import numpy as np # n-dimentional array math lib
import random
import librosa # music and audio analysis lib
import pyaudio
import wave
import os.path
# pydub: audio analysis lib
from pydub import AudioSegment
from pydub.silence import split_on_silence
# import scikit-learn related helper function train_test_split
from sklearn.model_selection import train_test_split
# importing keras related helper classes:
#  LSTM, Dense, Dropout, Flatten, Sequential, Adam, EarlyStopping and ModelCheckpoint
import keras
from keras.layers import LSTM, Dense, Dropout, Flatten, Conv2D
from keras.models import Sequential, load_model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint

SEED = 2018
DATA_DIRECTORY = './data/spoken_numbers_pcm/'

# split all .wav files in a training set and validation set
wav_files = glob.glob(DATA_DIRECTORY + "*.wav")
X_train, X_val = train_test_split(wav_files, test_size=0.2, random_state=SEED)

# print training set's size and validation set's size
print("\ntraining set size: {}".format(len(X_train)))
print("validation set size: {}\n".format(len(X_val)))

N_FEATURES = 20
MAX_LEN = 120  # 120 -> test val , 80 old val
N_CLASSES = 10

# encoding function
def one_hot_encode(labels_dense, n_classes=10):
    return np.eye(n_classes)[labels_dense]

# read .wav files and transform them to usable output
def batch_generator(data, batch_size=16):
    while 1:
        random.shuffle(data)
        X, y = [], []
        for i in range(batch_size):
            wav_file = data[i]
            wave, sampling_rate = librosa.load(wav_file, mono=True)
            num = int(wav_file[26])
            label = one_hot_encode(num, N_CLASSES)
            y.append(label)
            mfcc = librosa.feature.mfcc(wave, sampling_rate)
            M = MAX_LEN - len(mfcc[0])
            # if M < 0:
            #     M = 0
            mfcc = np.pad(
                mfcc, 
                ((0, 0), (0, M)),
                mode='constant', 
                constant_values=0
            )
            X.append(np.array(mfcc))
        yield np.array(X), np.array(y)


# read .wav files and transform them to usable output *non generator function*
def take_nXs(data, n):
    random.shuffle(data)
    X, y = [], []
    for i in range(n):
        wav_file = data[i]
        wave, sampling_rate = librosa.load(wav_file, mono=True)
        label = int(wav_file[26])
        y.append(label)
        mfcc = librosa.feature.mfcc(wave, sampling_rate)
        M = MAX_LEN - len(mfcc[0])
        # if M < 0:
        #     M = 0
        mfcc = np.pad(
            mfcc,
            ((0, 0), (0, M)),
            mode='constant',
            constant_values=0
        )
        X.append(np.array(mfcc))
    return np.array(X), y


# take_nXs2 is the same as take_nXs but it doesn't return the labels
def take_nXs2(data, n):
    X = []
    for i in range(n):
        wav_file = data[i]
        wave, sampling_rate = librosa.load(wav_file, mono=True)
        mfcc = librosa.feature.mfcc(wave, sampling_rate)
        M = MAX_LEN - len(mfcc[0])
        # print(len(mfcc[0]))
        # if M < 0:
        #     M = 0
        mfcc = np.pad(
            mfcc,
            ((0, 0), (0, M)),
            mode='constant',
            constant_values=0
        )
        X.append(np.array(mfcc))
    return np.array(X)


# print the shapes of training set and label set
X_set, y_set = next(batch_generator(X_train, batch_size=1))
print('shape of data set: {}'.format(X_set.shape))
print('shape of label set: {}\n'.format(y_set.shape))


# define hyperparameters
LEARNING_RATE = 0.0025 # 0.001
BATCH_SIZE = 16 # 64
N_EPOCHS = 5 # 50
DROPOUT = 0.3 # 0.5
INPUT_SHAPE = X_set.shape[1:]
STEPS_PER_EPOCH = 2240 // BATCH_SIZE  # SAMPLES // BATCH_SIZE # 50


model_path = './model.h5'
model = None
# check if the model is already trained
if not os.path.exists(model_path):
    # define neural network's architecture
    model = Sequential()
    model.add(
        LSTM(256, return_sequences=True, input_shape=INPUT_SHAPE, dropout=DROPOUT)
    ) # 256 old val, 50 new val
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))  # 128 old, 100 test val
    model.add(Dropout(DROPOUT))
    model.add(Dense(N_CLASSES, activation='softmax'))


    # we set the loss function, compile our model and output the model's summary
    opt = Adam(lr=LEARNING_RATE)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=opt, 
        metrics=['accuracy']
    )
    model.summary()


    # show training and validation accuracy on terminal
    history = model.fit_generator(
        generator=batch_generator(X_train, BATCH_SIZE),
        steps_per_epoch=STEPS_PER_EPOCH,
        epochs=N_EPOCHS,
        verbose=1,
        validation_data=batch_generator(X_val, 32),
        validation_steps=2240 // BATCH_SIZE
    )

    model.save('model.h5')  # save model's state(architecture, weights, etc..)
else:
    # load existing model
    model = load_model('model.h5')


# tesing data from data/tests folder
file_path = "./data/tests/panos.wav"
audio = AudioSegment.from_wav(file_path) # read wav file

# split recording to chunks
chunks = split_on_silence(
    audio,
    min_silence_len=150,
    silence_thresh=-50
)

# save each chunk as .wav file
for i, chunk in enumerate(chunks):
    chunk.export("./data/chunks/chunk{0}.wav".format(i), format="wav")

# read all chunks
wav_chunks = glob.glob("./data/chunks/" + "*.wav") # find all .wav chunks
wav_chunks = sorted(wav_chunks)

X3_tests = take_nXs2(wav_chunks, len(wav_chunks)) # take all samples
predictions2 = model.predict_classes(X3_tests, batch_size=1, verbose=0) # predict output

# show predictions
idx = 0
for p in predictions2:
    print("digit-{0}: {1}\n".format(idx, p))
    idx += 1
