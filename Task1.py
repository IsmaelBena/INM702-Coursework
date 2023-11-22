import numpy as np
from numpy.random import default_rng
import tensorflow as tf
from tensorflow import keras
import sklearn
from sklearn.preprocessing import OneHotEncoder

rng = default_rng()

def sigmoid(x):
    return 1/(1+np.exp(-1*x))

def softmax(x):
    exp_vals = np.exp(x - np.max(x))
    return exp_vals / np.sum(exp_vals)

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
            self.output = softmax(self.dense_output)
        else:
            self.output = self.dense_output
        
        # print('out',self.output[0])    
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
            act_grad = 1 #activation gradient is equivalent to the gradient from previous layer (layer i+1)
            # print('none',act_grad.shape)
        # print('after',act_grad.shape)
        # print('gradinput',gradient_input.shape)
        # print('act',act_grad.shape)
        # print('weight',self.weights.shape)
        # print('input',self.input.shape)
        if(current): #checking if first layer, update weights if so
            # print('updating weights')
            temp=gradient_input*act_grad
            # print('temp',temp)
            # print('input',self.input.shape)
            # print('weight',self.weights)
            gradient=np.dot(self.input,temp.T) #################################################### dimension problem
            
            # print('GRADIENT',np.sum(gradient))
            # print('WEIGHTS',self.weights[0][0])
            # tempweights=self.weights
            self.weights-=self.lr*gradient
            # print(np.sum(self.weights!=tempweights))
            # print('WEIGHTSAFTER',self.weights[0][0])
        else:
            # print('gradinput',gradient_input)
            # print('actgrad',act_grad.shape)
            temp=gradient_input*act_grad
            # print('temp',temp.shape)
            gradient=np.dot(self.weights,temp) ################################################## dimension problem
            # print('GRADIENT NON FIRST',gradient)
        return gradient
        
# class Dropout: #for task c
#     def __init__(self,ratio):
#         self.ratio=ratio
        

class NN:
    def __init__(self,  loss='MSE', optimizer=0, lr=0.001):
        self.layers = []
        self.input = 0
        self.layer_input = 0
        self.test=0
        self.loss = loss
        self.optimizer = optimizer
        self.lr = lr
        self.cost=0
        self.cost_deri=0
        
    def addLayer(self, input_size, output_size, activation="none"):
        self.layers.append(DenseLayer(input_size, output_size, activation=activation,lr=self.lr))
    
    def fit(self,input,test, epochs=1):
        self.input = input
        self.layer_input = input
        self.test=test
        for e in range(epochs):
            # print("before",self.layers[0].weights[0][0])
            for i in self.layers: #commence forward passing
                self.layer_input = i.forward_pass(self.layer_input)

            # print("z2",self.layer_input) #should be same
            # print('test',self.test)
            # print("y",self.layers[-1].output) #should be same
            ypred=self.layers[-1].output
            # print(ypred[0])
            if(self.loss=='MSE'):
                self.cost=1/2*(self.test-ypred)**2 #cost for backpropagation (MSE used for now, y-y^)
                self.cost_deri=ypred-self.test
            elif(self.loss=='CrossEntropy'):
                self.cost=-1*np.sum(self.test*np.log(ypred))
                # self.cost_deri=-1*np.sum(self.test*np.log(ypred))
            # print('ypred',ypred)
            # print('ytest',self.test)
            # print('cost',self.cost)
            # print('cost deri',self.cost_deri)
            # print("before",self.layers[0].weights)
            for iter,i in enumerate(self.layers): #ascending order to update the weights (further layers (layers i+1) update are not affected by the layers before (i))
                temp_grad=self.cost_deri
                print('i',i.weights.shape)
                for j in self.layers[iter:][::-1]: #descending order to accumulate the gradient values starting from the output
                    print(j.weights.shape)
                    # print('tempgrad',temp_grad.shape)
                    if i!=j:
                        temp_grad=j.back_pass(temp_grad,current=0)
                    else:
                        temp_grad=j.back_pass(temp_grad,current=1)
            self.layer_input=input
                    
            # print("after",self.layers[0].weights[0][0])
        
    def predict(self, input):
        self.prediction=input
        for i in self.layers: #commence forward passing
            self.prediction = i.forward_pass(self.prediction)
        # print('test',self.layers[-1].output[0])
        return(self.layers[-1].output) #should be same
 
 
#MAIN
 
cifartrain,cifartest=tf.keras.datasets.mnist.load_data()
Xtrain,ytrain=cifartrain[0],cifartrain[1]
Xtest,ytest=cifartest[0],cifartest[1]
# print(np.sum(Xtrain))
# print(Xtest.shape)
Xtrain=np.transpose(Xtrain.reshape(-1,28*28))
# print(np.sum(Xtrain))
Xtest=np.transpose(Xtest.reshape(-1,28*28))
ytrain=ytrain.reshape(-1,1)
ytest=ytest.reshape(-1,1)
train_enc=OneHotEncoder()
ytrain=np.transpose(train_enc.fit_transform(ytrain).toarray())
test_enc=OneHotEncoder()
ytest=np.transpose(test_enc.fit_transform(ytest).toarray())
# ytrain=train_enc.transform(ytrain).toarray()
# print(ytrain.shape)
# ytest=test_enc.transform(ytest).toarray()

testinput=Xtrain
testdata=np.array(ytrain)
num_classes=10

# print(testinput.shape)
# print(testinput)
test=NN(loss='MSE',optimizer=0,lr=1e-3)
test.addLayer(testinput.shape[0],64,'relu')
test.addLayer(64,32,'relu')
test.addLayer(32,num_classes,'sigmoid')
test.fit(testinput,testdata,epochs=3)
ypred=test.predict(Xtest)
# print(Xtest.shape)
a=ypred
b=ytest
print(a)
print(b)
# print('accuracy',np.equal(a,b)/60000*100,'%')
# print(ytest[0].argmax())
# print(test.layers[-1].output)