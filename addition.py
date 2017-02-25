from __future__ import print_function

import random

import numpy as np

from keras.layers import Dense, Dropout, Activation
from keras.models import Sequential
from keras.utils import np_utils


nb_classes = 10
SUM_MEMBER_COUNT = 2
INPUT_DIM = nb_classes * nb_classes
OUTPUT_DIM = SUM_MEMBER_COUNT * nb_classes
TEST_SIZE = 3 * 10
TRAIN_SIZE = 6 * 10
TOTAL_SIZE = 100
batch_size = 128
nb_epoch = 2000

np.random.seed(1337)  # for reproducibility

def genData():
    input = np.zeros(TOTAL_SIZE, dtype=np.int8)
    output = np.zeros(TOTAL_SIZE, dtype=np.int8)
    for x in range(0, 9):
        for y in range(0, 9):
            input[x + 10 * y] = x + 10 * y
            output[x + 10 * y] = x + y
    return input, output

def split(J):
    X = J[0]
    y = J[1]
    trainIndex = 0
    testIndex = 0
    xTrain = np.zeros(TOTAL_SIZE, dtype=np.int8)
    yTrain =  np.zeros(TOTAL_SIZE, dtype=np.int8)
    xTest = np.zeros(TOTAL_SIZE, dtype=np.int8)
    yTest =  np.zeros(TOTAL_SIZE, dtype=np.int8)
    for i in range(0, 99):
        if (random.random() > 0.2):
            xTrain[trainIndex] = X[i]
            yTrain[trainIndex] = y[trainIndex]
            trainIndex += 1
        else:
            xTest[testIndex] = X[i]
            yTest[testIndex] = y[testIndex]
            testIndex += 1
    print(xTrain)
    return (xTrain[:trainIndex], yTrain[:trainIndex]), (xTest[:testIndex], yTest[:testIndex])



# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = split(genData())
print(X_train)

# if K.image_dim_ordering() == 'th':
# X_train = X_train.reshape(X_train.shape[0], 1)
# X_test = X_test.reshape(X_test.shape[0], 1)
#     input_shape = (1, 1)
# else:
#     X_train = X_train.reshape(X_train.shape[0], 1, 1)
#     X_test = X_test.reshape(X_test.shape[0], 1, 1)
#     input_shape = (1, 1)

X_train = np_utils.to_categorical(X_train, INPUT_DIM)
X_test = np_utils.to_categorical(X_test, INPUT_DIM)
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, OUTPUT_DIM)
Y_test = np_utils.to_categorical(y_test, OUTPUT_DIM)

model = Sequential()

model.add(Dense(OUTPUT_DIM, input_dim=INPUT_DIM))
model.add(Activation('tanh'))
model.add(Dropout(0.1))

model.add(Dense(OUTPUT_DIM))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          verbose=1, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
print(model.metrics_names)

