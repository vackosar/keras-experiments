from __future__ import print_function

import random

import numpy as np

from keras.layers import Dense, Dropout, Activation, Flatten, Reshape
from keras.models import Sequential
from keras.utils import np_utils

NUM_COUNT = 100
BOOLS_PER_CHAR = 10
MAX_CHARS_PER_NUM = 2
NB_CLASSES = 10
SUM_MEMBER_COUNT = 2
INPUT_DIM = NB_CLASSES * NB_CLASSES
OUTPUT_DIM = SUM_MEMBER_COUNT * NB_CLASSES
TEST_SIZE = 3 * 10
TRAIN_SIZE = 6 * 10
TOTAL_SIZE = NUM_COUNT * NUM_COUNT
batch_size = 128
nb_epoch = 5

np.random.seed(1337)  # for reproducibility

class CharacterTable(object):
    """Given a set of characters:
    + Encode them to a one hot integer representation
    + Decode the one hot integer representation to their character output
    + Decode a vector of probabilities to their character output
    """
    def __init__(self, chars):
        """Initialize character table.

        # Arguments
            chars: Characters that can appear in the input.
        """
        self.chars = sorted(set(chars))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))

    def encode(self, C, num_rows):
        """One hot encode given string C.

        # Arguments
            num_rows: Number of rows in the returned one hot encoding. This is
                used to keep the # of rows for each data the same.
        """
        X = np.zeros((num_rows, len(self.chars)))
        for i, c in enumerate(C):
            X[i, self.char_indices[c]] = 1
        return X

    def decode(self, X, calc_argmax=True):
        if calc_argmax:
            X = X.argmax(axis=-1)
        return ''.join(self.indices_char[x] for x in X)


def genData():
    input = np.zeros((TOTAL_SIZE, 2 * MAX_CHARS_PER_NUM, BOOLS_PER_CHAR), dtype=np.bool)
    output = np.zeros((TOTAL_SIZE, 2 * MAX_CHARS_PER_NUM, BOOLS_PER_CHAR), dtype=np.bool)
    i = 0
    for x in range(0, NUM_COUNT - 1):
        for y in range(0, NUM_COUNT - 1):
            i = i + 1
            input[i] = TABLE.encode(str(x).zfill(MAX_CHARS_PER_NUM) + str(y).zfill(MAX_CHARS_PER_NUM), 2 * MAX_CHARS_PER_NUM)
            output[i] = TABLE.encode(str(x + y).zfill(2 * MAX_CHARS_PER_NUM), 2 * MAX_CHARS_PER_NUM)
    return input, output

def split(J):
    X = J[0]
    y = J[1]
    trainIndex = 0
    testIndex = 0
    xTrain = np.zeros((TOTAL_SIZE, 2 * MAX_CHARS_PER_NUM, BOOLS_PER_CHAR), dtype=np.bool)
    yTrain = np.zeros((TOTAL_SIZE, 2 * MAX_CHARS_PER_NUM, BOOLS_PER_CHAR), dtype=np.bool)
    xTest = np.zeros((TOTAL_SIZE, 2 * MAX_CHARS_PER_NUM, BOOLS_PER_CHAR), dtype=np.bool)
    yTest = np.zeros((TOTAL_SIZE, 2 * MAX_CHARS_PER_NUM, BOOLS_PER_CHAR), dtype=np.bool)
    for i in range(0, TOTAL_SIZE - 1):
        if (random.random() > 0.1):
            xTrain[trainIndex] = X[i]
            xTrain[trainIndex] = X[i]
            yTrain[trainIndex] = y[trainIndex]
            trainIndex += 1
        else:
            xTest[testIndex] = X[i]
            xTest[testIndex] = X[i]
            yTest[testIndex] = y[testIndex]
            testIndex += 1
    print(xTrain)
    return (xTrain[:trainIndex], yTrain[:trainIndex]), (xTest[:testIndex], yTest[:testIndex])

TABLE = CharacterTable(('0123456789'))

# the data, shuffled and split between train and test sets
(X_train, Y_train), (X_test, Y_test) = split(genData())
print(X_test)

# if K.image_dim_ordering() == 'th':
# X_train = X_train.reshape(X_train.shape[0], 1)
# X_test = X_test.reshape(X_test.shape[0], 1)
#     input_shape = (1, 1)
# else:
#     X_train = X_train.reshape(X_train.shape[0], 1, 1)
#     X_test = X_test.reshape(X_test.shape[0], 1, 1)
#     input_shape = (1, 1)

# X_train = np_utils.to_categorical(X_train, 2 * NB_CLASSES)
# X_test = np_utils.to_categorical(X_test, 2 * NB_CLASSES)
print(X_test)
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
# Y_train = np_utils.to_categorical(y_train, OUTPUT_DIM)
# Y_test = np_utils.to_categorical(y_test, OUTPUT_DIM)




model = Sequential()

model.add(Dense(40, input_shape=(4, 10)))
model.add(Activation('tanh'))
model.add(Dropout(0.3))

# model.add(Dense(100))
# model.add(Activation('tanh'))
# model.add(Dropout(0.3))
#
# model.add(Dense(100))
# model.add(Activation('tanh'))
# model.add(Dropout(0.3))
#
# model.add(Dense(100))
# model.add(Activation('tanh'))
# model.add(Dropout(0.3))

model.add(Dense(10))
model.add(Activation('softmax'))
model.add(Reshape((4, 10)))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
print(model.metrics_names)


