class MLP:
  def __init__(self, layers):
    self.layers = layers

  def forward(self, x):
    out = x
    for l in self.layers:
      out = l.forward(out)
    return out

  def error(self, y):
    return self.layers[-1].y - y


  def back(self, y):
    dL_dy = self.error(y)
    for i in reversed(range(1, len(self.layers))):

      if self.layers[i].trainable:
        dy_dw = self.layers[i].dy_dw()
        dW =  np.dot(dL_dy, dy_dw)
        self.layers[i].weight += dW

        dL_dy = np.dot(self.layers[i].weight.T, dL_dy)
        #dB = dL_dy
        #self.layers[i].bias += dB
        #b
      else:
        dL_dy = dL_dy * self.layers[i].dy_dx()
