import numpy as np
import random
from layers import *
from mlp import MLP

from keras.datasets import mnist
(train_X, train_y), (test_X, test_y) = mnist.load_data()
train_X = train_X/255.*2.-1.
test_X = test_X/255.*2.-1.

layers = [
    Flatten(),
    InputLayer(),
    DenseLayer(784, 50),
    Relu(),
    DenseLayer(50, 10),
    Sigmoid()
]
mlp = MLP(layers)

x_train = train_X
y_train = train_y
x_test = test_X
y_test = test_y

mlp.fit(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, lr=0.001, epochs=100)
