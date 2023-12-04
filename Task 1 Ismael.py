import numpy as np
from numpy.random import default_rng
from math import floor
rng = default_rng()

from keras.datasets import mnist
(train_X, raw_train_y), (test_X, raw_test_y) = mnist.load_data()

train_X = np.reshape(train_X, (train_X.shape[0], 28*28)) #reshape X into (number inputs, number features)
test_X = np.reshape(test_X, (test_X.shape[0], 28*28)) #Since the size of image for X is 28x28, the number of features is 28x28

def one_hot_encode(raw_y): #Function for one hot encoding
    encoded_y = []
    for y in raw_y:
        one_hot = np.zeros((10,), dtype=int) #10 classes (0-9)
        one_hot[y] = 1 #one hot encode the position of the class to their respective one hot encoded space
        encoded_y.append(one_hot) #make the one hot code into an array
    return np.array(encoded_y)

train_y = one_hot_encode(raw_train_y)
test_y = one_hot_encode(raw_test_y)
# print(raw_train_y[0:8], raw_train_y.shape)
# print(train_y[0:8], train_y.shape)

def normalize(x): #normalize the inputs using minmax scaler
    return (x-np.min(x))/(np.max(x)-np.min(x))

train_X = normalize(train_X)
test_X = normalize(test_X)

class DenseLayer:
    def __init__(self, num_inputs, num_neurons):
        self.num_neurons = num_neurons
        self.num_inputs = num_inputs
        #initialise the weights with respect to the no. of inputs
        #size of weights is (number of inputs, number of neurons)
        self.weights = np.random.normal(0,scale=1/np.sqrt(num_inputs),size=(num_inputs, num_neurons)) 
        # self.weights = np.random.randn(num_inputs, num_neurons) * 0.01
        self.biases = np.zeros((num_neurons,1))
        # print(self.input, self.input.shape)
        # print(self.biases, self.biases.shape)
        # print(self.weights, self.weights.shape)

    def forward_pass(self, inputs): #forward pass
        # print('dense input', inputs[0])
        self.inputs = inputs
        self.dense_output = np.dot(inputs, self.weights) #dot product with the inputs the the weights
        self.output = self.dense_output        

    def back_pass(self, prev_grad):
        # print('prev_grad', prev_grad)
        # print('b_input', self.inputs[0])        
        self.d_weights = np.dot(self.inputs.T, prev_grad)
        self.d_bias = np.sum(prev_grad, axis=0, keepdims=True)
        self.current_grad = np.dot(prev_grad, self.weights.T)
        # print('dense current grad', self.current_grad)

class Relu_Layer(DenseLayer):
    def __init__(self, num_inputs, num_neurons):
        super().__init__(num_inputs, num_neurons)

    def forward_pass(self, inputs):
        super().forward_pass(inputs)
        self.output = np.maximum(0, self.dense_output)

    def back_pass(self, prev_grad):
        self.d_activation = prev_grad.copy()
        self.d_activation[self.dense_output <= 0] = 0
        super().back_pass(self.d_activation)

class Sigmoid_Layer(DenseLayer):
    def __init__(self, num_inputs, num_neurons):
        super().__init__(num_inputs, num_neurons)

    def sigmoid(self, x):
        return np.where(x>=0, 1/(1+np.exp(-1*x)), np.exp(x)/(1+np.exp(x)))

    def forward_pass(self, inputs):
        # print('sig inputs', inputs[0])
        super().forward_pass(inputs)
        print('output', self.output)
        # print('weights', self.weights)
        self.output = self.sigmoid(self.dense_output)

    def back_pass(self, prev_grad):
        prev_grad = self.dense_output.copy()
        self.d_activation = self.sigmoid(prev_grad) * (1 - self.sigmoid(prev_grad))
        # print('dact', self.d_activation)
        super().back_pass(self.d_activation)
        # print('dweights', self.d_weights)

class Softmax_Layer(DenseLayer):
    def __init__(self, num_inputs, num_neurons):
        super().__init__(num_inputs, num_neurons)

    def forward_pass(self, inputs):
        super().forward_pass(inputs)
        #print('weights', self.weights)
        #print('denes output of softamx layer', self.dense_output)
        exp_vals = np.exp(self.dense_output - np.max(self.dense_output, axis=1, keepdims=True))
        self.output = exp_vals / np.sum(exp_vals, axis=1, keepdims=True)
        #print('softamx output', self.output)

    def back_pass(self, prev_grad):
        super().back_pass(prev_grad)

class Categorigal_Cross_Entropy_Loss:
    def forward_pass(self, predicted_y, true_y):
        # print(predicted_y.shape)
        predicted_y_clipped = np.clip(predicted_y, 1e-7, 1 - 1e-7)
        # print(predicted_y_clipped.shape)
        prediction_confidences = np.sum(predicted_y_clipped * true_y, axis=1)
        # average_loss = np.mean(-np.log(prediction_confidences))
        neg_log = -np.log(prediction_confidences)
        # print(neg_log)
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
        # print(range(len(prev_grad)))
        # print(self.true_y)
        #print('softmax pre', self.current_grad)
        self.current_grad[range(len(prev_grad)), self.true_y] -= 1
        #print('softmax mid', self.current_grad)
        self.current_grad = self.current_grad / len(prev_grad)
        #print('softmax post', self.current_grad)
        super().back_pass(self.current_grad)

class Stochasic_Gradient_Descent:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def update_layer(self, layer):
        #print('init weights', layer.weights)
        #print(layer.weights.shape)
        #print(layer.d_weights)
        layer.weights -= self.learning_rate * layer.d_weights
        #print('updated weights', layer.weights)
        #print('init biases', layer.biases)
        layer.biases -= self.learning_rate * layer.d_bias.T
        #print('updated biases', layer.biases)


class NN:
    def __init__(self,learning_rate=1e-3):
        self.layers = []
        self.opt = Stochasic_Gradient_Descent(learning_rate)
        self.losslog = []

    def create_batches(self, inputs, targets, batch_size):
        sample_number = len(inputs)
        X_batches = []
        y_batches = []
        for i in range(batch_size):
            # print(sample_number*(i/batch_size))
            X_batches.append(np.array(inputs[floor(sample_number*(i/batch_size)):floor(sample_number*((i+1)/batch_size))]))
            y_batches.append(np.array(targets[floor(sample_number*(i/batch_size)):floor(sample_number*((i+1)/batch_size))]))
        return X_batches, y_batches

    def accuracy(self, pred_y, true_y):
        predictions = np.argmax(pred_y, axis=1)
        # print(f'Predictions: {predictions}')
        argmax_targets = np.argmax(true_y, axis=1)
        accuracy = np.mean(predictions==argmax_targets)
        #print(f'Accuracy: {accuracy}')
        return accuracy

    def add_layer(self, num_inputs, num_neurons, activation_function="none"):
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
                self.layers.append(DenseLayer(num_inputs, num_neurons))
            else:
                raise Exception("Invalid activation function provided.")

    def fit(self, inputs, targets, batch_size, epochs):
        if self.layers[0].num_inputs != inputs.shape[1]:
            raise Exception("Shape of inputs does not match first layers input")
        elif type(self.layers[-1]) != Softmax_CategoricalCrossEntroyLoss:
            raise Exception("Output layer needs to be softmax")
        else:
            self.b_inputs, self.b_targets = self.create_batches(inputs, targets, batch_size)
            for epoch in range(epochs):
                self.b_outputs = []
                for b_index, batch in enumerate(self.b_inputs):
                    # print("======================================== input ======================================================================================")
                    self.output = batch
                    for index, layer in enumerate(self.layers):
                        # print(f'starting {index} layer')
                        if index == len(self.layers) - 1:
                            self.loss = layer.forward_pass(self.output, self.b_targets[b_index][index])
                            self.losslog.append(self.loss)
                        else:
                            layer.forward_pass(self.output)
                        # print(f'{layer.activation_function} output: {layer.output}')
                        self.output = layer.output
                    # print("=====================================================================================================================================")
                    # print(f'Epoch output: {self.output}')
                    self.b_outputs.append(self.output)
                    # print(f'Collective outputs: {self.outputs}')
                    # print("=====================================================================================================================================")
                    for layer_index in range(len(self.layers) - 1, -1, -1):
                        layer = self.layers[layer_index]
                        if layer_index == len(self.layers) - 1:
                            layer.back_pass(self.b_outputs[b_index], self.b_targets[b_index]) 
                        else:
                            layer.back_pass(self.layers[layer_index + 1].current_grad)
                        self.opt.update_layer(layer)
                    self.accuracy(self.b_outputs[b_index], self.b_targets[b_index])
                    print(f'\nEpoch: {epoch + 1} | Batch: {b_index + 1} | Accuracy: {round(self.accuracy(self.output, self.b_targets[b_index])*100)}%')
                    
    def test(self, inputs, targets, batch_size):
        batches_X, batches_y = self.create_batches(inputs, targets, batch_size) #create number of batches based on batch size
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


my_nn = NN()
my_nn.add_layer(train_X.shape[1], 256, "sigmoid")
my_nn.add_layer(256, 10, "softmax_output")
my_nn.fit(train_X, train_y, 10, 20)
my_nn.test(test_X, test_y, 10)