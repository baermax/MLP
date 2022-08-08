import numpy as np

class FullyConnectedLayer:
  def __init__(self, inputs, outputs):
    self.inputs = inputs
    self.outputs = outputs
    self.weight = np.random.rand(outputs, inputs)
    self.bias = np.zeros((outputs, 1))
    self.x = None
    self.y = None
    self.trainable = True

  def forward(self, x):
    self.x = x
    self.y = np.dot(self.weight, x) + self.bias
    return self.y

  def dy_dw(self):
    return self.x.T

class InputLayer:
  def __init__(self):
    self.x = None
    self.y = None
    self.trainable = False

  def forward(self, x):
    self.x = self.y = x
    return self.y

  def dy_dx(self):
    return self.x

class Sigmoid:
  def __init__(self):
    self.x = None
    self.y = None
    self.trainable = False

  def forward(self, x):
    self.x = x
    self.y = 1 / (1 + np.exp(-x))
    return self.y

  def dy_dx(self):
    return (1-self.y)*self.y
  
class Relu:
  def __init__(self):
    self.x = None
    self.y = None
    self.trainable = False

  def forward(self, x):
    self.x = x
    self.y = np.maximum(0, x)
    return self.y

  def dy_dx(self):
    return (self.x > 0) * 1
