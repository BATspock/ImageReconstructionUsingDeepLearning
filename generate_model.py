#genearte model to learn relationships among pixel on MNIST dataset
#import dependecies
from keras.datasets import mnist
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
from keras.utils import np_utils
import cv2
import numpy as np
from model import ModelLSTM

#using mnist dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#image generaion for given number
flag = 1
while(flag==1):
    num = int(input("Enter the number you want to generate:"))
    if not ((num>=0 or num<=9)and(type(num)==int)):
        print("Enter an integer number between 0 and 9")
    else:
        flag = 0
X = []
for i in range(len(y_train)):
    if y_train[i] == num:
        X.append(x_train[i])

#reshape array
X= np.array(X)
X = np.reshape(X, (1, len(X), 28, 28))

#select 3 pixel at a time to generate pixel in the following row 
X_train=[]
Y_train =[]
#create mask
#row by row and pixel by pixel approach
for it in range(X.shape[1]):
    for i in range(1,X.shape[2] -1):
        for j in range(1,X.shape[3] -1):
            X_train.append([X[0][it][i-1][j-1], X[0][it][i-1][j], X[0][it][i-1][j+1]])
            Y_train.append(X[0][it][i][j])

X_train = np.reshape(X_train,(((X.shape[1])*(X.shape[2]-2)*(X.shape[3]-2)),1,3))#reshape for timesteps

Y_train = np.array(Y_train)#change list to np array
Y_train = np.reshape(Y_train,(((X.shape[1])*(X.shape[2]-2)*(X.shape[3]-2)),1))#reshape
Y_train = np_utils.to_categorical(Y_train)#create one hot vector encoding

#create model

generationModel = ModelLSTM(X_train, Y_train)
ModelLSTM.runAndExecuteModel()
ModelLSTM.saveModel("check")