import numpy as np
from numpy.random import default_rng
rng = default_rng()

class DenseLayer:
    def __init__(self, num_inputs, num_neurons, activation="none"):
        self.weights = 0.1 * np.random.randm(num_inputs, num_neurons)
        self.biases = np.zeros(1, num_neurons)
        self.activation = activation
        
    def forward_pass(self, input):
        self.dense_output = np.dot(self.inputs, self.weights) + self.biases
        
    def activation_function(self):
        if self.activation == "relu":
            self.output = np.maximum(0, self.dense_output)
        elif self.activation == "sigmoid":
            pass
        elif self.activation == "softmax":
            exp_vals = np.exp(self.dense_output - np.max(self.dense_output, axis=1, keepdims=True))
            self.output = exp_vals / np.sum(exp_vals, axis=1, keepdims=True)
        else:
            self.output = self.dense_output
            
    def back_pass():
        pass
        
class Dropout: #for task c
    def __init__(self,ratio):
        self.ratio=ratio
        

class NN:
    def __init__(self, input, loss, optimizer, lr):
        self.layers = []
        self.input = input
        
    def addLayer(self, input_size, output_size, activation="none"):
        self.layers.append(DenseLayer(input_size, output_size, activation=activation))
    
    def fit(self, epochs):
        for i in self.layers:
            i.forward_pass(self.input)
            i.activation_function()
        self.layers[-1].output
        
        for j in self.layers[::-1]:
            j.back_pass()
    