import numpy as np
from numpy.random import default_rng
rng = default_rng()

def sigmoid(x):
    return 1/(1+np.exp(-1*x))

class DenseLayer:
    def __init__(self, num_inputs, num_neurons, activation="none"):
        self.weights = 0.1 * np.random.randn(num_inputs, num_neurons)
        self.biases = np.zeros((1, num_neurons))
        self.activation = activation
        self.input=input
        self.dense_output = 0
        self.output = 0
        
    def forward_pass(self, input):
        self.dense_output = np.dot(input, self.weights) + self.biases

        if self.activation == "relu":
            self.output = np.maximum(0, self.dense_output)
        elif self.activation == "sigmoid":
            self.output = sigmoid(self.dense_output)
        elif self.activation == "softmax":
            exp_vals = np.exp(self.dense_output - np.max(self.dense_output, axis=1, keepdims=True))
            self.output = exp_vals / np.sum(exp_vals, axis=1, keepdims=True)
        else:
            self.output = self.dense_output
            
        return self.output
        
    def back_pass(self):
        if self.activation == "relu":
            self.gradient = 1 if self.output>0 else 0
        elif self.activation == "sigmoid":
            self.gradient = sigmoid(self.output)*(1-sigmoid(self.output))
        elif self.activation == "softmax":
            exp_vals = np.exp(self.dense_output - np.max(self.dense_output, axis=1, keepdims=True))
            self.gradient = exp_vals / np.sum(exp_vals, axis=1, keepdims=True)
        else:
            self.output = self.dense_output
        
# class Dropout: #for task c
#     def __init__(self,ratio):
#         self.ratio=ratio
        

class NN:
    def __init__(self, input, loss, optimizer, lr):
        self.layers = []
        self.input = input
        self.layer_input = input
        self.loss = loss
        self.optimizer = optimizer
        self.lr = lr
        
    def addLayer(self, input_size, output_size, activation="none"):
        self.layers.append(DenseLayer(input_size, output_size, activation=activation))
    
    def fit(self, epochs=1):
        
        for i in self.layers:
            self.layer_input = i.forward_pass(self.layer_input)

        print("z2",self.layer_input) #should be same
        print("y",self.layers[-1].output) #should be same
        
        for i in self.layers[::-1]:
            pass
        
 
testinput=[10,10]
test=NN(testinput,0,0,0)
test.addLayer(2,2,'sigmoid')
test.addLayer(2,2,'relu')
test.addLayer(2,2,'softmax')
test.fit(1)