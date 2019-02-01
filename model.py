#import dependecies
from keras.datasets import mnist
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
from keras.utils import np_utils
import cv2
import numpy as np

#create a LSTM model to learn dependencies and save model for further prediction
class ModelLSTM(object):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.model = Sequential()
    def runAndExecuteModel(self, 
                loss_function = 'categorical_crossentropy', 
                optimizer_fn = 'adam'):

        self.model.add(LSTM(256, input_shape=(self.X.shape[1],self.X.shape[2])))
        self.model.add(Dense(256, activation = 'softmax'))
        self.model.compile(loss = loss_function, optimizer = optimizer_fn, metrics=['accuracy'])
        #execute model
        self.model.fit(self.X, self.Y, epochs=10, batch_size = 32768)#this should work
    
    def saveModel(self, name):
        self.model.save(str(name)+".h5")