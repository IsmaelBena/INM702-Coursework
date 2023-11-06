import numpy as np
import math
from numpy.random import default_rng
rng = default_rng()

class RegLayer:
    def __init__(self,n,input_dim=16):
        n_dims=n
        self.weights=rng.uniform(size=(n_dims,input_dim))
        self.bias=np.zeros((1,n))
        self.out=0

    def forward(self,input):
        self.out=np.dot(self.weights.T, input) + self.bias
        return np.dot(self.weights.T, input) + self.bias
    
    def backward(self):
        pass
    
class ActLayer: #for task a
    def __init__(self, type):
        self.type=type
        if(self.type=='relu'):
            pass
        elif(self.type=='sigmoid'):
            pass
        else:
            pass
        
class Outlayer: #for task b
    def __init__(self,n_out):
        self.n_out=n_out
        
class Dropout: #for task c
    def __init__(self,ratio):
        self.ratio=ratio
        

class my_nn:
    def __init__(self,data):
        

    def forward(self,input):
        return np.dot(self.weights.T, input) + self.bias
    
my_nn=