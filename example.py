from matplotlib.rcsetup import validate_color_or_auto
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import random

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


# Actually, the mnist data set consists of 60000 training samples and 10000 test samples. Nevertheless, we will train our model with less samples
# in order to make our training faster.
x_train = train_X[:20000]
y_train = train_y[:20000]
x_test = test_X[:2000]
y_test = test_y[:2000]

# Of course you can change some of the parameters to identify the best hyperparameters for this use case. 
# As an example you might want to train with more or less samples or change the learning rate and the number of iterations through the data.
mlp.fit(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, lr=0.001, epochs=1000)
