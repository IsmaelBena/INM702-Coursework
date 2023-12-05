import numpy as np
from numpy.random import default_rng
from math import floor
rng = default_rng()
import sklearn
from sklearn.preprocessing import StandardScaler

from keras.datasets import mnist
training_data, testing_data = mnist.load_data()

class Data:
    def __init__(self, training_data, testing_data):
        self.train_X, self.train_y = training_data
        self.test_X, self.test_y = testing_data
    
    def reshape(self):
        self.train_X = np.reshape(self.train_X, (self.train_X.shape[0], 28*28)) # Reshape X into (number inputs, number features)
        self.test_X = np.reshape(self.test_X, (self.test_X.shape[0], 28*28)) # Since the size of image for X is 28x28, the number of features is 28x28

    def one_hot_encode_func(self, raw_y):
        encoded_y = []
        for y in raw_y:
            one_hot = np.zeros((10,), dtype=int) #10 classes (0-9)
            one_hot[y] = 1 #one hot encode the position of the class to their respective one hot encoded space
            encoded_y.append(one_hot) #make the one hot code into an array
        return np.array(encoded_y)

    def one_hot_encode_data(self):
        self.train_y = self.one_hot_encode_func(self.train_y)
        self.test_y = self.one_hot_encode_func(self.test_y)

    def minmax(self, x):
        return (x-np.min(x))/(np.max(x)-np.min(x))
    
    def scale_data(self, scaling_type):
        if scaling_type == 'minmax':
            self.train_X = self.minmax(self.train_X)
            self.test_X = self.minmax(self.test_X)
        elif scaling_type == 'standard':
            scaler=StandardScaler()
            self.train_X = scaler.fit_transform(self.train_X)
            self.test_X=scaler.fit_transform(self.train_X)

        else:
            raise Exception('Incorrect scaling_type passed.')

    def create_batches_train(self, batch_size):
        sample_number = len(self.train_X)
        X_batches = []
        y_batches = []
        for i in range(batch_size):
            X_batches.append(np.array(self.train_X[floor(sample_number*(i/batch_size)):floor(sample_number*((i+1)/batch_size))]))
            y_batches.append(np.array(self.train_y[floor(sample_number*(i/batch_size)):floor(sample_number*((i+1)/batch_size))]))
        return X_batches, y_batches

    def create_batches_test(self, batch_size):
        sample_number = len(self.test_X)
        X_batches = []
        y_batches = []
        for i in range(batch_size):
            X_batches.append(np.array(self.test_X[floor(sample_number*(i/batch_size)):floor(sample_number*((i+1)/batch_size))]))
            y_batches.append(np.array(self.test_y[floor(sample_number*(i/batch_size)):floor(sample_number*((i+1)/batch_size))]))
        return X_batches, y_batches

# Each Layer has a Dense Layer before an optional activation function, so we have a parent Dense Layer
class Dense_Layer:
    def __init__(self, num_inputs, num_neurons):
        self.num_inputs = num_inputs
        self.num_neurons = num_neurons
        self.weights = np.random.normal(0,scale=1/np.sqrt(num_inputs),size=(num_inputs, num_neurons)) 
        self.biases = np.zeros((num_neurons,1))

    def forward_pass(self, inputs):
        self.inputs = inputs
        self.dense_output = np.dot(self.inputs, self.weights) + self.biases.T #dot product with the inputs the the weights
        self.output = self.dense_output

    def back_pass(self, prev_grad):
        self.d_weights = np.dot(self.inputs.T, prev_grad)
        self.d_bias = np.sum(prev_grad, axis=0, keepdims=True)
        self.current_grad = np.dot(prev_grad, self.weights.T)

# Layer with a Relu activation function
class Relu_Layer(Dense_Layer):
    def __init__(self, num_inputs, num_neurons):
        super().__init__(num_inputs, num_neurons)

    def forward_pass(self, inputs):
        # Work out dense values
        super().forward_pass(inputs)
        # Call Relu activation function
        self.output = np.maximum(0, self.dense_output)

    def back_pass(self, prev_grad):
        # derivitve of activation function
        self.d_activation = prev_grad.copy()
        self.d_activation[self.dense_output <= 0] = 0
        # derivative of dense layer
        super().back_pass(self.d_activation)

# Layer with a Relu activation function
class Sigmoid_Layer(Dense_Layer):
    def __init__(self, num_inputs, num_neurons):
        super().__init__(num_inputs, num_neurons)

    def sigmoid(self, x):
        return np.where(x>=0, 1/(1+np.exp(-1*x)), np.exp(x)/(1+np.exp(x)))

    def forward_pass(self, inputs):
        # Work out dense values
        super().forward_pass(inputs)
        # Call Sigmoid activation function
        self.output = self.sigmoid(self.dense_output)

    def back_pass(self, prev_grad):
        # derivitve of activation function
        self.d_activation = self.sigmoid(self.dense_output) * (1 - self.sigmoid(self.dense_output))
        self.d_activation *= prev_grad
        # derivative of dense layer
        super().back_pass(self.d_activation)

# Layer with a Relu activation function
class Softmax_Layer(Dense_Layer):
    def __init__(self, num_inputs, num_neurons):
        super().__init__(num_inputs, num_neurons)

    def softmax(self, x):
        exp_vals = np.exp(x- np.max(x,axis=0,keepdims=True))
        softmax_out = exp_vals / np.sum(exp_vals, axis=0, keepdims=True)
        return softmax_out

    def forward_pass(self, inputs):
        # Work out dense values
        super().forward_pass(inputs)
        # Call Softmax activation function
        self.output = self.softmax(self.dense_output)

    def back_pass(self, prev_grad):
        super().back_pass(prev_grad)

class Categorigal_Cross_Entropy_Loss:
    def forward_pass(self, pred_y, true_y):
        pred_y_clipped = np.clip(pred_y, 1e-7, 1 - 1e-7)
        pred_confidences = np.sum(pred_y_clipped * true_y, axis=1)
        neg_log = -np.log(pred_confidences)
        return neg_log

    def back_pass(self, prev_grad, true_y):
        self.current_grad = -true_y / prev_grad
        # normalise the gradient
        self.current_grad = self.current_grad / len(prev_grad)

class Softmax_CategoricalCrossEntroyLoss(Softmax_Layer):
    def __init__(self, num_inputs, num_neurons):
        super().__init__(num_inputs, num_neurons)
        self.loss = Categorigal_Cross_Entropy_Loss()
    
    def forward_pass(self, inputs, targets):
        super().forward_pass(inputs)
        return self.loss.forward_pass(self.output, targets)
    
    def back_pass(self, prev_grad, true_y):
        self.true_y = np.argmax(true_y, axis=1)
        self.current_grad = prev_grad.copy()
        self.current_grad[range(len(prev_grad)), self.true_y] -= 1
        self.current_grad = self.current_grad / len(prev_grad)
        super().back_pass(self.current_grad)

class Stochasic_Gradient_Descent:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def update_layer(self, layer):
        layer.weights -= self.learning_rate * layer.d_weights
        layer.biases -= self.learning_rate * layer.d_bias.T

class NN:
    def __init__(self, optimizer='sgd', learning_rate=1e-3):
        self.layers = []
        self.losslog=[]
        if optimizer == 'sgd':
            self.opt = Stochasic_Gradient_Descent(learning_rate)
        else:
            raise Exception('Not valid optimizer')
        
    def accuracy(self, pred_y, true_y):
        predictions = np.argmax(pred_y, axis=1)
        argmax_targets = np.argmax(true_y, axis=1)
        accuracy = np.mean(predictions==argmax_targets)
        return accuracy

    def add_layer(self, num_inputs, num_neurons, activation_function='none'):
        if (len(self.layers) != 0) and (self.layers[-1].num_neurons != num_inputs):
            raise Exception("The number of outputs from the previous layer do not match the inputs of the layer you are attempting to add.")
        elif (len(self.layers) != 0) and type(self.layers[-1]) == Softmax_CategoricalCrossEntroyLoss:
            raise Exception("Softmax output layer already assigned.")
        else:
            if activation_function == "relu":
                self.layers.append(Relu_Layer(num_inputs, num_neurons))
            elif activation_function == "sigmoid":
                self.layers.append(Sigmoid_Layer(num_inputs, num_neurons))
            elif activation_function == "softmax_output":
                self.layers.append(Softmax_CategoricalCrossEntroyLoss(num_inputs, num_neurons))
            elif activation_function == "none":
                self.layers.append(Dense_Layer(num_inputs, num_neurons))
            else:
                raise Exception("Invalid activation function provided.")

    def fit(self, data, batch_size, epochs):
        if self.layers[0].num_inputs != data.train_X.shape[1]:
            raise Exception("Shape of inputs does not match first layers input")
        elif type(self.layers[-1]) != Softmax_CategoricalCrossEntroyLoss:
            raise Exception("Output layer needs to be softmax")
        else:
            self.b_inputs, self.b_targets = data.create_batches_train(batch_size)
            for epoch in range(epochs):
                self.b_outputs = []
                for b_index, batch in enumerate(self.b_inputs):
                    self.output = batch
                    for index, layer in enumerate(self.layers):
                        if index == len(self.layers) - 1:
                            self.loss = layer.forward_pass(self.output, self.b_targets[b_index][index])
                            self.losslog.append(self.loss)
                        else:
                            layer.forward_pass(self.output)
                        self.output = layer.output
                    self.b_outputs.append(self.output)
                    for layer_index in range(len(self.layers) - 1, -1, -1):
                        layer = self.layers[layer_index]
                        if layer_index == len(self.layers) - 1:
                            layer.back_pass(self.b_outputs[b_index], self.b_targets[b_index]) 
                        else:
                            layer.back_pass(self.layers[layer_index + 1].current_grad)
                    for layer_index in range(len(self.layers) - 1, -1, -1):
                        self.opt.update_layer(layer)
                    self.accuracy(self.b_outputs[b_index], self.b_targets[b_index])
                    print(f'\nEpoch: {epoch + 1} | Batch: {b_index + 1} | Accuracy: {round(self.accuracy(self.output, self.b_targets[b_index])*100)}%')
                print(self.b_outputs[-1][0:5])
 

    def test(self, data, batch_size):
        batches_X, batches_y = data.create_batches_test(batch_size) #create number of batches based on batch size
        accuracies = [] #accuracy log
        for b_index, batch in enumerate(batches_X):
            self.output = batch
            for index, layer in enumerate(self.layers):
                if index == len(self.layers) - 1:
                    layer.forward_pass(self.output, batches_y[b_index][index]) #forward pass based on batches
                else:
                    layer.forward_pass(self.output)
                self.output = layer.output
            accuracies.append(self.accuracy(self.output, batches_y[b_index])) #log accuracy changes
            print(f'Batch: {b_index + 1} | Accuracy: {round(self.accuracy(self.output, batches_y[b_index])*100, 2)}') #print batch and accuracy
        print(f'=============== Test Results ===============')
        print(f'Average accuracy: {np.mean(accuracies)}')
        print('============================================')

data = Data(training_data, testing_data)
data.reshape()
data.one_hot_encode_data()
data.scale_data('minmax')

my_nn = NN(learning_rate=1e-1)
my_nn.add_layer(data.train_X.shape[-1], 256, "sigmoid")
my_nn.add_layer(256, 128, "sigmoid")
my_nn.add_layer(128, 10, "softmax_output")
my_nn.fit(data, 10, 100)
my_nn.test(data, 10)