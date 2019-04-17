from keras.datasets import fashion_mnist, mnist
from keras.layers import LSTM, Dense, Dropout, SimpleRNN
from keras.models import Sequential
from keras.utils import np_utils
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize
import cv2
import numpy as np

(X_train, Y_train), (X_test, Y_test) = fashion_mnist.load_data()

X = np.array(X)
print(X.shape)

X_avg = np.mean(X, axis=0)

print(X_avg.shape)

plt.imshow(X_avg)
