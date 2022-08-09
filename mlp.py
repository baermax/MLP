class MLP:
  def __init__(self, layers):
    self.layers = layers

  def forward(self, x):
    out = np.array(x).reshape(-1, 1)
    for l in self.layers:
      out = l.forward(out)
    return out

  def error(self, y):
    return y - self.layers[-1].y


  def back(self, y, lr):
    dL_dy = self.error(y)
    loss = dL_dy
    
    for i in reversed(range(1, len(self.layers))):

      if self.layers[i].trainable:

        self.layers[i].bias += dL_dy * lr
        dy_dw = self.layers[i].dy_dw()
        dW =  np.dot(dL_dy, dy_dw)
        self.layers[i].weight += dW * lr
        dL_dy = np.dot(self.layers[i].weight.T, dL_dy)

      else:
        dL_dy = dL_dy * self.layers[i].dy_dx()

      return np.sum(abs(loss)) # returns current loss


  def fit(self, x, y, lr, epochs):
    for e in range(epochs):
      current_loss = 0
      for i in range(len(x)):
        self.forward(x[i])
        current_loss =+ self.back(y[i], lr)
      print(current_loss)
