#MLP

class MLP:
  def __init__(self, layers):
    self.layers = layers

  def forward(self, x):
    out = np.array(x).reshape(-1, 1)
    for l in self.layers:
      out = l.forward(out)
    return out

  def error_gradient(self, y):
    return y - self.layers[-1].y
  
  def error(self, y):
    return 0.5*(y - self.layers[-1].y)**2

  def back(self, y, lr):
    dL_dy = self.error_gradient(y)
    loss =  self.error(y)

    for i in reversed(range(1, len(self.layers))):
      
      if self.layers[i].trainable:
        
        self.layers[i].bias += dL_dy * lr
        dy_dw = self.layers[i].dy_dw()
        dW =  np.dot(dL_dy, dy_dw)
        
        self.layers[i].weight += dW * lr
        dL_dy = np.dot(self.layers[i].weight.T, dL_dy)
        
      else:
        dL_dy = dL_dy * self.layers[i].dy_dx()
      
    return np.sum(loss) # returns current loss
  
  def validate(self, x_test, y_test): # evaluates the model's performance on given data samples
    score = 0
    for i in range(len(x_test)):
      pred = decode(self.forward(x_test[i]))
      if pred == y_test[i]:
        score += 1
    return score/len(x_test)
  
  def fit(self, x_train, y_train, x_test, y_test, lr, epochs):
    for e in range(epochs):
      current_loss = 0
      for i in range(len(x_train)):
        self.forward(x_train[i])
        current_loss += self.back(encode(y_train[i], 10), lr)
      print(f"ACCURACY {self.validate(x_test, y_test)*100}% LOSS {current_loss}")
