import numpy as np
import random
from keras.datasets import mnist
import timeit

FAST = True

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train/255.*2.-1.
x_test = x_test/255.*2.-1.

def encode(index, size):
  vec = [0. for i in range(size)]
  vec[index] = 1.
  return vec

def decode(vec):
  if FAST:
    index = np.argmax(vec, axis=1)
  else:
    index = None
    h = 0
    for i,v in enumerate(vec[0]):
      if h<v:
        h = v
        index = i
  return index

def batch_encode(indexes, size):
  vectors = []
  for i in range(len(indexes)):
    vec = np.zeros((size))
    vec[indexes[i]] = 1
    vectors = np.append(vectors, vec)
  return vectors.reshape(-1, size)

def batch_decode(vectors):
  if FAST:
    indexes = np.argmax(vectors, axis=1)
  else:
    indexes = []
    for vec in vectors:
      index = None
      h = 0
      for i,v in enumerate(vec):
        if h<v:
          h = v
          index = i
      indexes.append(index)
  return indexes


class DenseLayer:
  def __init__(self, inputs, outputs):
    self.x = None
    self.y = None
    self.inputs = inputs
    self.weight = np.random.normal(size=(inputs, outputs))/inputs
    self.bias = np.zeros((1, outputs))
    self.trainable = True
    self.flatten = False

  def forward(self, x, batch_size):
    self.x = x
    
    self.y = np.dot(x, self.weight) + self.bias
    return self.y

  def dY_dX(self):
    return self.x.T

class InputLayer:
  def __init__(self):
    self.x = None
    self.y = None
    self.trainable = False
    self.flatten = False

  def forward(self, x, batch_size):
    self.x = self.y = x
    return self.y

  def dY_dX(self):
    return self.x

class Sigmoid:
  def __init__(self):
      self.x = None
      self.y = None
      self.trainable = False
      self.flatten = False

  def forward(self, x, batch_size):
    self.x = x
    self.y = 1 / (1 + np.exp(-x))
    return self.y

  def dY_dX(self):
    return (1-self.y)*self.y

class Flatten:
  def __init__(self):
    self.x = None
    self.y = None
    self.trainable = False
    self.flatten = True

  def forward(self, x, batch_size):
    self.x = x
    self.y = x.reshape(batch_size, -1)
    return self.y
  
  
  
class MLP:
  def __init__(self, layers):
    self.layers = layers

  def forward(self, x, batch_size):
    out = np.array(x)
    for l in self.layers:
      out = l.forward(out, batch_size)
    return out

  def loss_gradient(self, y):
    return self.layers[-1].y - y

  def loss(self, y):
    return 0.5*(self.layers[-1].y - y)**2

  def back(self, y, lr, batch_size):
    loss = self.loss(y)
    dLdY = self.loss_gradient(y)

    for i in reversed(range(len(self.layers))):
      if self.layers[i].trainable:

        self.layers[i].bias -= np.average(dLdY, axis=0) * lr   
        self.layers[i].weight -= np.dot(self.layers[i].dY_dX(), dLdY) / batch_size * lr
        dLdY = np.dot(dLdY, self.layers[i].weight.T)

      else:
        if self.layers[i].flatten:
            dLdY = dLdY.reshape(self.layers[i].x.shape)
        else: 
          dLdY = dLdY * self.layers[i].dY_dX()
    return np.sum(loss)

  def validate(self, x_test, y_test):
    score = 0
    for i in range(len(x_test)):
      pred = decode(self.forward(x_test[i], 1))
      if pred == y_test[i]:
        score += 1
    return score/len(x_test)

  def fit(self, x_train, y_train, x_test, y_test, lr, epochs, batch_size):

    for e in range(epochs):
      # ideally the samples should be shuffled at this point
      loss = 0
      for i in range(0, len(x_train), batch_size):
        if len(x_train)-i >= batch_size:
          x_batch = x_train[i:i+batch_size]
          y_batch = y_train[i:i+batch_size]

        self.forward(x=x_batch, batch_size=batch_size)
        loss += self.back(y=batch_encode(y_batch, 10), lr=lr, batch_size=batch_size)

      test_accuracy = self.validate(x_test=x_test, y_test=y_test)
      training_accuracy = self.validate(x_test=x_train[:2000], y_test=y_train[:2000])

      if e%10 == 0 or e%10 == 5: #prints the current metrics every 5 iterations
        print(f"{e}/{epochs} EPOCHS, TEST ACCURACY: {test_accuracy}, TRAINING ACCURACY: {training_accuracy}, LOSS: {loss}")
        
        
        
        
layers = [
  InputLayer(),
  Flatten(),
  DenseLayer(784, 60),
  Sigmoid(),
  DenseLayer(60, 10),
  Sigmoid(),
]

mlp = MLP(layers)

def fun():
  mlp.fit(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,lr=0.1, epochs=3, batch_size=400)

timeit.timeit(fun, number=20)
# timeit.time(lambda x:   mlp.fit(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,lr=0.1, epochs=1, batch_size=400))
