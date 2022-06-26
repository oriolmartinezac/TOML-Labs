# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 15:11:09 2022

@author: UPCnet
"""

#based on https://github.com/jcjohnson/pytorch-examples

# Code in file tensor/two_layer_net_numpy.py
import math

import numpy as np
import matplotlib.pyplot as plt
import time
from statistics import mean

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


def loss_function(data, theta):
  pass


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


def module(xx):
    return(np.sqrt(xx.dot(xx)))

def plot_grid(ww1,ww2,ww3, xtest, ytest):
    
    plt.figure(figsize=(10,10))
    plt.xlim(-2,2) 
    plt.ylim(-2,2) 

    plt.title("Classification areas, orange: class 0, green: class 1. Also shows testing set")
    plt.xlabel("x0")
    plt.ylabel("x1")

    GRID=100
    grid=np.zeros(shape=(2*GRID,3))
    for height in range(-GRID,GRID):  
        for i in range(0,2*GRID):
            grid[i,0]= 2*(i-GRID)/GRID
            grid[i,1]= 2*height/GRID
            grid[i,2]=1

        pred_grid=forward(grid,ww1,ww2,ww3)
        idgrid_1 = np.where(pred_grid > 0.5)[0]
        idgrid_0 = np.where(pred_grid <= 0.5)[0]
    
        if len(idgrid_0)>0:
            plt.plot(grid[idgrid_0,0],grid[idgrid_0,1],"+",c="orange")
        if len(idgrid_1)>0:
            plt.plot(grid[idgrid_1,0],grid[idgrid_1,1],"+",c="green")

    idtest_1 = np.where(ytest > 0.5)[0]
    idtest_0 = np.where(ytest <= 0.5)[0]

    plt.plot(xtest[idtest_0,0],xtest[idtest_0,1],"d",c="red")
    plt.plot(xtest[idtest_1,0],xtest[idtest_1,1],"d",c="blue")
    plt.show()
    return()

#####################################################
# T is batch size;
# H is hidden dimension
T, H = 640, 15 # original values: 640 20  # BEST 640, 20

D_in=3  # Input dimension (includes BIAS!)
D_out=1 #output dimension, class (1,0) or (0,1)


# create Training Set, classified according with radius to origin.
# you may include some noise in the classification

radius1=1
radius2=1.5
class_noise=0.0


# Create random input data
x=np.zeros(shape=(T,D_in))
y = np.zeros(shape=(T,1))

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


plt.figure(figsize=(10,10))
plt.xlim(-2,2) 
plt.ylim(-2,2) 

plt.title("Training set, red: class 0, blue: class 1")
plt.xlabel("x0")
plt.ylabel("x1")

plt.plot(x[id_0,0],x[id_0,1],"d",c="red")
plt.plot(x[id_1,0],x[id_1,1],"d",c="blue")

plt.show()



# Randomly initialize weights
w1 = np.random.randn(D_in, H)
w2 = np.random.randn(H, H)
w3 = np.random.randn(H, D_out)


learning_rate = 1e-4 # original value: 1e-4 # BEST 1e-4
ITER=2000 # original value: 2000

tot_loss_array=np.zeros(shape=(ITER,2)) # stores training[0] and testing[1] errors


forward_time = []
backward_time = []

for t in range(ITER):
  start_time_forward = time.time()
  # Forward pass: compute predicted y
  #Original
  z1 = x.dot(w1)
  a1 = np.maximum(z1,0)     # ReLU
  z2= a1.dot(w2)
  a2=np.maximum(z2,0)       # ReLU
  z3 = a2.dot(w3)
  y_pred = sigmoid(z3)      # Sigmoid

  """# New code
  z1 = x.dot(w1)
  a1 = np.maximum(z1, 0.1*z1)  # leaky  ReLU
  z2 = a1.dot(w2)
  a2 = np.maximum(z2, 0.1*z2)  # leaky ReLU
  z3 = a2.dot(w3)
  y_pred = sigmoid(zx3)  # Sigmoid"""

  """# New code
  a=0.01
  z1 = x.dot(w1)
  a1 = np.maximum(z1, a*z1)  # PReLU
  z2 = a1.dot(w2)
  a2 = np.maximum(z2, a*z2)  # PReLU
  z3 = a2.dot(w3)
  y_pred = sigmoid(z3)  # Sigmoid"""

  """# New code
  z1 = x.dot(w1)
  a1 = np.maximum(z1, ((math.e**z1)-1))  # leaky  ReLU
  z2 = a1.dot(w2)
  a2 = np.maximum(z2, ((math.e**z2)-1))  # leaky ReLU
  z3 = a2.dot(w3)
  y_pred = sigmoid(z3)  # Sigmoid"""


  #print("--- %s seconds to do forward ---" % (time.time() - start_time_forward))
  forward_time.append(time.time() - start_time_forward)

  # Compute and print loss
  loss = - y*safe_log(y_pred) - (1-y)*safe_log(1-y_pred)

  tot_loss_array[t,0]=loss.mean()
  if t%10==0: print(t, loss.mean())
  
   # check testing error
  ytest_pred=forward(xtest,w1,w2,w3)
  losstest = - ytest*safe_log(ytest_pred) - (1-ytest)*safe_log(1-ytest_pred)
  tot_loss_array[t,1]=losstest.mean()

 
  start_time_backpropagation = time.time() # Get start_time back propagation

   # Backprop to compute gradients of w1, w2, w3 with respect to loss
  grad_y_pred = - y*safe_inv(y_pred) + (1-y)*safe_inv(1-y_pred)

  grad_z3=  grad_y_pred*sigmoid(z3)*(1-sigmoid(z3))

  grad_w3 = a2.T.dot(grad_z3)

  grad_a2= grad_z3.dot(w3.T)

  grad_z2=grad_a2.copy()
  grad_z2[z2 < 0] = 0

  grad_w2 = a1.T.dot(grad_z2)
  grad_a1 = grad_z2.dot(w2.T)
  
  grad_z1 = grad_a1.copy()
  grad_z1[z1 < 0] = 0
  grad_w1 = x.T.dot(grad_z1)


  #print("--- %s seconds to do backpropagation ---" % (time.time() - start_time_backpropagation))
  backward_time.append(time.time() - start_time_backpropagation)

  #Update weights
  # Momentum
  alpha=0.7 # original 0.7: # BEST 0.9
  if t==0:
      v1=grad_w1
      v2=grad_w2
      v3=grad_w3

  v1 = alpha*v1 + (1-alpha)*grad_w1
  v2 = alpha*v2 + (1-alpha)*grad_w2
  v3 = alpha*v3 + (1-alpha)*grad_w3

  w1 -= learning_rate * v1
  w2 -= learning_rate * v2
  w3 -= learning_rate * v3


  """if t%100 == 0:
      print("WEIGHTS")
      print(w1)
      print("\n")
      print(w2[:, 0])  
  """

  if (t%250==0): 
      plot_grid(w1, w2, w3, xtest, ytest)

print("Total execution time spend in forward: ", sum(forward_time))
print("Total execution time spend in backward: ", sum(backward_time))

plot_grid(w1, w2, w3, xtest, ytest)

plt.title("training loss (r), testing loss (b)")
plt.xlabel("# iteration")
plt.ylabel("Log(loss)")

print("Last tess loss value: ", tot_loss_array[ITER-1,1])

plt.plot(np.log(tot_loss_array[:,0]), c="red")
plt.plot(np.log(tot_loss_array[:,1]), c="blue")
plt.show()



