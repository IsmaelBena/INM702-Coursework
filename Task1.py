import numpy as np
from numpy.random import default_rng
rng = default_rng()

def sigmoid(x):
    return 1/(1+np.exp(-1*x))

class DenseLayer:
    def __init__(self, num_inputs, num_neurons, activation="none", lr=1):
        self.weights = np.random.uniform(0,1,size=(num_inputs, num_neurons)) ########################### dimension problem

        self.biases = np.zeros((num_neurons,1)) ########################################################## dimension problem
        self.activation = activation
        self.input=0
        self.dense_output = 0
        self.output = 0
        self.lr=lr
        
    def forward_pass(self, input):
        self.input=input
        # print(self.weights)

        self.dense_output = np.dot(self.weights.T,input) + self.biases ################################### dimension problem


        if self.activation == "relu":
            self.output = np.maximum(0, self.dense_output)
        elif self.activation == "sigmoid":
            self.output = sigmoid(self.dense_output)
        elif self.activation == "softmax":
            exp_vals = np.exp(self.dense_output - np.max(self.dense_output, axis=1, keepdims=True))
            self.output = exp_vals / np.sum(exp_vals, axis=1, keepdims=True)
        else:
            self.output = self.dense_output
        
        # print('out',self.output.shape)    
        return self.output
        
    def back_pass(self,gradient_input,current=0): #gradient input depends on the values fed by the layer before (layer i+1)
        #current = flag to check if the weights are in the current layer
        if self.activation == "relu":
            act_grad = np.where(self.output>1,1,0) #derivative of RELU
        elif self.activation == "sigmoid":
            act_grad = self.output*(1-self.output) #Derivative of sigmoid: sigmoid(1-sigmoid), sigmoid = sigmoid of this layer = self.output
        elif self.activation == "softmax":
            exp_vals = np.exp(self.dense_output - np.max(self.dense_output, axis=1, keepdims=True)) #Derivative of softmax)
            act_grad = exp_vals / np.sum(exp_vals, axis=1, keepdims=True)
        else:
            act_grad = gradient_input #activation gradient is equivalent to the gradient from previous layer (layer i+1)
            
        # print('gradinput',gradient_input.shape)
        # print('act',act_grad.shape)
        # print('weight',self.weights.shape)
        # print('input',self.input.shape)
        if(current): #checking if first layer, update weights if so
            # print('input',self.input)
            # print('actgrad',act_grad)
            gradient=np.dot(self.input,act_grad.T) #################################################### dimension problem
            # print('GRADIENT',gradient.shape)
            self.weights-=self.lr*gradient
        else:
            gradient=np.dot(act_grad.T,self.weights.T) ################################################## dimension problem
            # print('grADIENT',gradient.shape)
        return gradient
        
# class Dropout: #for task c
#     def __init__(self,ratio):
#         self.ratio=ratio
        

class NN:
    def __init__(self, input,test, loss, optimizer, lr):
        self.layers = []
        self.input = input
        self.layer_input = input
        self.test=test
        self.loss = loss
        self.optimizer = optimizer
        self.lr = lr
        self.cost=0
        
    def addLayer(self, input_size, output_size, activation="none"):
        self.layers.append(DenseLayer(input_size, output_size, activation=activation))
    
    def fit(self, epochs=1):
        for i in self.layers: #commence forward passing
            self.layer_input = i.forward_pass(self.layer_input)

        print("z2",self.layer_input) #should be same
        print('test',self.test)
        print("y",self.layers[-1].output) #should be same
        
        self.cost=(self.test-self.layer_input) #cost for backpropagation (MSE used for now, y-y^)
        # print('cost',self.cost.shape)
        print("before",self.layers[0].weights)
        
        for i in self.layers: #ascending order to update the weights (further layers (layers i+1) update are not affected by the layers before (i))
            temp_grad=self.cost
            for j in self.layers[::-1]: #descending order to accumulate the gradient values starting from the output
                
                if i!=j:
                    temp_grad=j.back_pass(temp_grad,current=0)
                else:
                    temp_grad=j.back_pass(temp_grad,current=1)
                    
        print("after",self.layers[0].weights)
 
testinput=np.transpose(np.array([[10,10]]))
testdata=np.array([[5],[5]])
# print(testinput.shape)
# print(testinput)
test=NN(testinput,testdata,0,0,0)
test.addLayer(testinput.shape[0],3,'relu')
test.addLayer(3,2,'relu')
# test.addLayer(2,testinput.shape[1],'softmax')
test.fit(1)
# print(test.layers[0]==test.layers[1])