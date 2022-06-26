import torch
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import statistics
import time
"""
A fully-connected ReLU network with one hidden layer, trained to predict y from x
by minimizing squared Euclidean distance.

This implementation defines the model as a custom Module subclass. Whenever you
want a model more complex than a simple sequence of existing Modules you will
need to define your model this way.
"""

def sigmoid(xx):
    return(1/(1+np.exp(-xx)))

def safe_log(xx):
    yy=np.zeros(shape=(len(xx),1))
    for ii in range(len(xx)):
        if xx[ii] < 1e-10 :
            yy[ii]=np.log(1e-10)
        else:
            yy[ii]=np.log(xx[ii])
    return(yy)

def safe_inv(xx):
    yy=np.zeros(shape=(len(xx),1))
    for ii in range(len(xx)):
        if np.abs(xx[ii]) < 1e-10 :
            yy[ii]=1e+10
        else:
            yy[ii]=1.0/xx[ii]
    return(yy)


def forward(xx,ww1,ww2,ww3):
    # Forward pass: compute predicted y
    # Original code
    zz1 = xx.dot(ww1)
    aa1 = np.maximum(zz1,0)       # ReLU
    zz2 = aa1.dot(ww2)
    aa2 = np.maximum(zz2,0)       # ReLU
    zz3 = aa2.dot(ww3)
    yy_pred = sigmoid(zz3)        # sigmoid
    """# New code
    zz1 = xx.dot(ww1)
    aa1 = np.maximum(zz1, 0.1*zz1)  # Leaky ReLU
    zz2 = aa1.dot(ww2)
    aa2 = np.maximum(zz2, 0.1*zz2)  # Leaky ReLU
    zz3 = aa2.dot(ww3)
    yy_pred = sigmoid(zz3)  # sigmoid"""
    """# New code
    a =0.01
    zz1 = xx.dot(ww1)
    aa1 = np.maximum(zz1, a*zz1)  # P ReLU
    zz2 = aa1.dot(ww2)
    aa2 = np.maximum(zz2, a*zz2)  # P ReLU
    zz3 = aa2.dot(ww3)
    yy_pred = sigmoid(zz3)  # sigmoid"""

    """zz1 = x.dot(w1)
    aa1 = np.maximum(zz1, ((math.e**zz1) - 1))  # leaky  ReLU
    zz2 = aa1.dot(ww2)
    aa2 = np.maximum(zz2, ((math.e**zz2) - 1))  # leaky ReLU
    zz3 = aa2.dot(ww3)
    yy_pred = sigmoid(zz3)  # Sigmoid"""

    return(yy_pred)

def plot_grid(ww1, ww2, ww3, xtest, ytest):
    plt.figure(figsize=(10, 10))
    plt.xlim(-2, 2)
    plt.ylim(-2, 2)

    plt.title("Classification areas, orange: class 0, green: class 1. Also shows testing set")
    plt.xlabel("x0")
    plt.ylabel("x1")

    GRID = 100
    grid = np.zeros(shape=(2 * GRID, 3))
    for height in range(-GRID, GRID):
        for i in range(0, 2 * GRID):
            grid[i, 0] = 2 * (i - GRID) / GRID
            grid[i, 1] = 2 * height / GRID
            grid[i, 2] = 1

        pred_grid = forward(grid, ww1, ww2, ww3)
        idgrid_1 = np.where(pred_grid > 0.5)[0]
        idgrid_0 = np.where(pred_grid <= 0.5)[0]

        if len(idgrid_0) > 0:
            plt.plot(grid[idgrid_0, 0], grid[idgrid_0, 1], "+", c="orange")
        if len(idgrid_1) > 0:
            plt.plot(grid[idgrid_1, 0], grid[idgrid_1, 1], "+", c="green")

    idtest_1 = np.where(ytest > 0.5)[0]
    idtest_0 = np.where(ytest <= 0.5)[0]

    plt.plot(xtest[idtest_0, 0], xtest[idtest_0, 1], "d", c="red")
    plt.plot(xtest[idtest_1, 0], xtest[idtest_1, 1], "d", c="blue")
    plt.show()
    return ()


def module(xx):
    return(np.sqrt(xx.dot(xx)))


class TwoLayerNet(torch.nn.Module):
  def __init__(self, D_in, H, D_out):
    """
    In the constructor we instantiate two nn.Linear modules and assign them as
    member variables.
    """
    super(TwoLayerNet, self).__init__()
    self.linear1 = torch.nn.Linear(D_in, H)
    self.linear2 = torch.nn.Linear(H, H)
    self.activation = torch.nn.Linear(H, D_out)
    #self.activation = torch.nn.Sigmoid()
    self.bias = torch.nn.Parameter(torch.rand(1, 1))

  def forward(self, x):
    """
    In the forward function we accept a Tensor of input data and we must return
    a Tensor of output data. We can use Modules defined in the constructor as
    well as arbitrary (differentiable) operations on Tensors.
    """
    h_relu = self.linear1(x).clamp(min=0)
    h_relu2 = self.linear2(h_relu).clamp(min=0)
    y_pred = self.activation(h_relu2)
    return y_pred

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
T, D_in, H, D_out = 640, 3, 20, 1

# Create random Tensors to hold inputs and outputs
radius1=1
radius2=1.5
class_noise=0.0


# Create random input data
x=np.zeros(shape=(T,D_in))
y = np.zeros(shape=(T,1))
x = x.astype('float32')
y = y.astype('float32')

x[:,0] = np.random.randn(T)
x[:,1] = np.random.randn(T)
x[:,2] = 1              # For adding a bias in the first stage

for i in range(T):
    mod=module(x[i,0:2])
    if( mod  + class_noise*np.random.randn()< radius1) or (mod  + class_noise*np.random.randn() > radius2):
        y[i]=0
    else:
        y[i]=1

#create now a testing_set...same size

# Create random input data
xtest=np.ones(shape=(T,3))
ytest = np.zeros(shape=(T,1))
xtest = xtest.astype('float32')
ytest = ytest.astype('float32')

xtest[:,0] = np.random.randn(T)
xtest[:,1] = np.random.randn(T)
xtest[:,2] = 1  # For adding a bias in the first stage

for i in range(T):
    mod=module(xtest[i,0:2])
    if( mod  + class_noise*np.random.randn()< radius1) or (mod  + class_noise*np.random.randn() > radius2):
        ytest[i]=0
    else:
        ytest[i]=1

id_1 = np.where(y == 1)[0]
id_0 = np.where(y == 0)[0]

# Construct our model by instantiating the class defined above.
model = TwoLayerNet(D_in, H, D_out)

SGD_model = torch.nn.Sequential(
torch.nn.Linear(D_in, H),
torch.nn.ReLU(),
torch.nn.Linear(H,H),
torch.nn.ReLU(),
torch.nn.Linear(H, D_out),
torch.nn.Sigmoid()
)

# Construct our loss function and an Optimizer. The call to model.parameters()
# in the SGD constructor will contain the learnable parameters of the two
# nn.Linear modules which are members of the model.
loss_fn = torch.nn.MSELoss(reduction='sum')
#optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.7)
optimizer = torch.optim.SGD(SGD_model.parameters(), lr=1e-4, momentum=0.9) # 0.4
ITER=2000*100#2000*10

"""x_tf = torch.randn(T, D_in)
bias = torch.randn((1, 1))
y_tf = torch.randn(T, D_out)
"""

tot_loss_array=np.zeros(shape=(ITER,2)) # stores training[0] and testing[1] errors

x_tf = torch.from_numpy(x)
y_tf = torch.from_numpy(y)
x_test_tf = torch.from_numpy(xtest)
y_test_tf = torch.from_numpy(ytest)

aux_loss = []
aux_loss2 = []
params = []

found = False

start_time = time.time()

for t in range(ITER):
  y_pred = SGD_model(x_tf)
  loss = loss_fn(y_pred, y_tf)
  tot_loss_array[t, 0] = loss.item()
  if t % 100 == 0:
      print(t, loss.item())

  if tot_loss_array[t, 0] <= 0.6111 and found == False:
      found = True
      print("Time to find the 0.6 error: ", time.time() - start_time)
      break


  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  """if (t%250==0):
      plot_grid(params[0], params[1], params[2], xtest, ytest)
      params.clear()"""

  SGD_model.eval()
  y_pred_test = SGD_model(x_test_tf)
  losstest = loss_fn(y_pred_test, y_test_tf)
  tot_loss_array[t, 1] = losstest.item()

plt.title("training loss (r), testing loss (b)")
plt.xlabel("# iteration")
plt.ylabel("Log(loss)")

print("Last training loss value: ", tot_loss_array[ITER-1,0])

print("Last tess loss value: ", tot_loss_array[ITER-1,1])

plt.plot(np.log(tot_loss_array[:,0]), c="red")
plt.plot(np.log(tot_loss_array[:,1]), c="blue")
plt.show()


"""
for t in range(ITER):
  for x_i, y_i in zip(x_tf, y_tf):
    # Forward pass: Compute predicted y by passing x to the model
    y_pred = model(x_i)

    # Compute and print loss
    loss = loss_fn(y_pred, y_i)

    #tot_loss_array[t, 0] = loss.item()
    aux_loss.append(loss.item())

    if t % 10 == 0:
       print(t, loss.item())

    # Zero gradients, perform a backward pass, and update the weights.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    #print(model.linear1())

    model.eval()
    # check testing error
    y_pred_test = model(x_test_tf)
    losstest = loss_fn(y_pred_test, y_test_tf)
    aux_loss2.append(losstest.item())
    
  tot_loss_array[t, 0] = statistics.mean(aux_loss)
  tot_loss_array[t, 1] = statistics.mean(aux_loss2)
  aux_loss.clear()
  aux_loss2.clear()


"""
