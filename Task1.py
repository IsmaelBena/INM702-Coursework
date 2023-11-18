import numpy as np
from numpy.random import default_rng
rng = default_rng()

def sigmoid(x):
    return 1/(1+np.exp(-1*x))

class DenseLayer:
    def __init__(self, num_inputs, num_neurons, activation="none", lr=1e-3):
        self.weights = np.random.uniform(-2,2,size=(num_inputs, num_neurons)) ########################### dimension problem
        # self.weights=weight
        self.biases = np.zeros((num_neurons,1)) ########################################################## dimension problem
        # self.biases=bias
        self.activation = activation
        self.input=0
        self.dense_output = 0
        self.output = 0
        self.lr=lr
        
    def forward_pass(self, input):
        self.input=input

        self.dense_output = np.dot(self.weights.T,input) + self.biases ################################### dimension problem
        # print('dense',self.dense_output)
        # print('before',self.dense_output.shape)
        if self.activation == "relu":
            self.output = np.maximum(0, self.dense_output)
        elif self.activation == "sigmoid":
            self.output = sigmoid(self.dense_output)
        elif self.activation == "softmax":
            exp_vals = np.exp(self.dense_output - np.max(self.dense_output))
            self.output = exp_vals / np.sum(exp_vals)
        else:
            self.output = self.dense_output
        
        # print('out',self.output)    
        return self.output
        
    def back_pass(self,gradient_input,current=0): #gradient input depends on the values fed by the layer before (layer i+1)
        #current = flag to check if the weights are in the current layer
        if self.activation == "relu":
            act_grad = np.where(self.output>1,1,0) #derivative of RELU
        elif self.activation == "sigmoid":
            act_grad = self.output*(1-self.output) #Derivative of sigmoid: sigmoid(1-sigmoid), sigmoid = sigmoid of this layer = self.output
            # print('sigmoid',act_grad.shape)
        elif self.activation == "softmax":
            exp_vals = np.exp(self.output - np.max(self.output)) #Derivative of softmax)
            act_grad = exp_vals / np.sum(exp_vals)
        else:
            act_grad = np.eye(self.input.shape[1]) #activation gradient is equivalent to the gradient from previous layer (layer i+1)
            # print('none',act_grad.shape)
        # print('after',act_grad.shape)
        # print('gradinput',gradient_input.shape)
        # print('act',act_grad.shape)
        # print('weight',self.weights.shape)
        # print('input',self.input.shape)
        if(current): #checking if first layer, update weights if so
            temp=self.input*act_grad
            # print('temp',temp.shape)
            gradient=np.dot(temp,gradient_input.T) #################################################### dimension problem
            
            # print('GRADIENT',gradient)
            # print('WEIGHTS',self.weights)
            self.weights-=self.lr*gradient
            # print('WEIGHTSAFTER',self.weights)
        else:
            temp=gradient_input*act_grad
            # print('temp',temp.shape)
            gradient=np.dot(self.weights,temp) ################################################## dimension problem
            # print('GRADIENT NON FIRST',gradient)
        return gradient
        
# class Dropout: #for task c
#     def __init__(self,ratio):
#         self.ratio=ratio
        

class NN:
    def __init__(self, input,test, loss=0, optimizer=0, lr=0.001):
        self.layers = []
        self.input = input
        self.layer_input = input
        self.test=test
        self.loss = loss
        self.optimizer = optimizer
        self.lr = lr
        self.cost=0
        self.cost_deri=0
        
    def addLayer(self, input_size, output_size, activation="none"):
        self.layers.append(DenseLayer(input_size, output_size, activation=activation,lr=self.lr))
    
    def fit(self, epochs=1):
        for e in range(epochs):
            
            for i in self.layers: #commence forward passing
                self.layer_input = i.forward_pass(self.layer_input)

            # print("z2",self.layer_input) #should be same
            # print('test',self.test)
            # print("y",self.layers[-1].output) #should be same
            ypred=self.layers[-1].output
            self.cost=1/2*(self.test-ypred)**2 #cost for backpropagation (MSE used for now, y-y^)
            self.cost_deri=ypred-self.test
            # print('cost',self.cost)
            # print('cost deri',self.cost_deri)
            # print("before",self.layers[0].weights)
            for i in self.layers: #ascending order to update the weights (further layers (layers i+1) update are not affected by the layers before (i))
                temp_grad=self.cost_deri
                for j in self.layers[::-1]: #descending order to accumulate the gradient values starting from the output
                    
                    if i!=j:
                        temp_grad=j.back_pass(temp_grad,current=0)
                    else:
                        temp_grad=j.back_pass(temp_grad,current=1)
            self.layer_input=self.input
                    
        # print("after",self.layers[0].weights)
        
    def predict(self, input):
        for i in self.layers: #commence forward passing
            self.layer_input = i.forward_pass(self.layer_input)
        print(self.layers[-1].output)
        return(self.layers[-1].output) #should be same
 
testinput=np.transpose(np.array([[0.05,0.10]]))
testdata=np.array([[0.01],[0.99]])

# print(testinput.shape)
# print(testinput)
test=NN(testinput,testdata,0,0,lr=1)
test.addLayer(testinput.shape[0],2,'sigmoid')
test.addLayer(2,2,'sigmoid')
# test.addLayer(2,testinput.shape[1],'softmax')
test.fit(100)
test.predict(testinput)

# print(test.layers[-1].output)