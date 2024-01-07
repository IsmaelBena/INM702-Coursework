#Import Required Libraries
import numpy as np
from numpy.random import default_rng
from math import floor
import sklearn
from sklearn.preprocessing import StandardScaler
import json
import copy
from keras.datasets import mnist

#Load MNIST dataset and unpack them into training data and testing data
training_data, testing_data = mnist.load_data()

#Define Data Class
class Data:
    def __init__(self, training_data, testing_data): #initialisation
        #self.train_x: training data
        #self.train_y: training label
        #self.test_x: testing data
        #self.test_y: testing label
        self.train_X, self.train_y = training_data
        self.test_X, self.test_y = testing_data
    
    def reshape(self): #reshape
        # Reshape X into (number inputs, number features)
        # Since the size of image for X is 28x28, the number of features should be 28x28
        self.train_X = np.reshape(self.train_X, (self.train_X.shape[0], 28*28)) 
        self.test_X = np.reshape(self.test_X, (self.test_X.shape[0], 28*28)) 

    def one_hot_encode_func(self, raw_y): #one hot encoding
        #inputs:
        #raw_y = data label
        
        #intialise parameters
        encoded_y = []
        for y in raw_y:
            one_hot = np.zeros((10,), dtype=int) #10 classes (0-9)
            one_hot[y] = 1 #one hot encode the position of the class to their respective one hot encoded space
            encoded_y.append(one_hot) #make the one hot code into an array
        return np.array(encoded_y) #return the array as a np array

    def one_hot_encode_data(self):
        
        #call the encode fucntion for training and testing labels
        self.train_y = self.one_hot_encode_func(self.train_y)
        self.test_y = self.one_hot_encode_func(self.test_y)

    def minmax(self, x):
        #minmax scale the data
        return (x-np.min(x))/(np.max(x)-np.min(x))
    
    def scale_data(self, scaling_type): #scales the data, only takes in 'minmax' and 'standard' as input, returns error otherwise
        if scaling_type == 'minmax':
            self.train_X = self.minmax(self.train_X)
            self.test_X = self.minmax(self.test_X)
        elif scaling_type == 'standard':
            #call the standard scaler to scale the input data
            scaler=StandardScaler()
            self.train_X = scaler.fit_transform(self.train_X)
            self.test_X=scaler.fit_transform(self.test_X)
        else:
            raise Exception('Incorrect scaling_type passed.')

    def create_batches_train(self, batch_size): #create batches for training 
        #input parameters:
        #batch_size: takes in an integer to load as batches for the model
        
        sample_number = len(self.train_X) #determine the length of the input data
        X_batches = []
        y_batches = []
        
        #create batches for X and y as a list, such that 
        for i in range(batch_size):
            X_batches.append(np.array(self.train_X[floor(sample_number*(i/batch_size)):floor(sample_number*((i+1)/batch_size))]))
            y_batches.append(np.array(self.train_y[floor(sample_number*(i/batch_size)):floor(sample_number*((i+1)/batch_size))]))
        return X_batches, y_batches

    def create_batches_test(self, batch_size): #create batches for testing
        #input parameters:
        #batch_size: takes in an integer to load as batches for the model
        
        sample_number = len(self.test_X)
        X_batches = []
        y_batches = []
        for i in range(batch_size):
            X_batches.append(np.array(self.test_X[floor(sample_number*(i/batch_size)):floor(sample_number*((i+1)/batch_size))]))
            y_batches.append(np.array(self.test_y[floor(sample_number*(i/batch_size)):floor(sample_number*((i+1)/batch_size))]))
        return X_batches, y_batches

# Each Layer has a Dense Layer before an optional activation function, so we have a parent Dense Layer
class Dense_Layer:
    def __init__(self, num_inputs, num_neurons, has_dropout, keep_rate, regularizer, lamda):
        # input parameters:
        # num_inputs: number of features in a data
        # num_neurons: designated number of neurons
        # regularizer: regularizer used, accepts 'none', 'l1', 'l2', and 'l1l2' as valid inputs
        # lamda: weight decay parameter for regularizer, default is set to 0 as there is no weight decay unless opted in otherwise
        # has_dropout: defines if a dropout layer exists
        # keep_rate: rate at which the percentage of weights are preserved
        
        
        self.num_inputs = num_inputs
        self.num_neurons = num_neurons
        self.has_dropout = has_dropout
        self.keep_rate = keep_rate
        self.regularizer = regularizer
        self.lamda=lamda 
        
        #weights, initialised on a scale of 1/sqrt(num_inputs)
        self.weights = np.random.normal(0,scale=1/np.sqrt(num_inputs),size=(num_inputs, num_neurons))  
        
        #biases, initialised to the number of neurons
        self.biases = np.zeros((num_neurons,1)) 
        
        # weights momentum, initalised at zero
        self.current_weight_momentum = np.zeros_like(self.weights) 

    def dropout(self):
        # creates mask where weights are dropped out, leaving only a percentage based on the keep_rate
        mask = (np.random.rand(*self.output.shape) < self.keep_rate) / self.keep_rate
        self.output = mask * self.output

    def forward_pass(self, inputs): #forward pass function
        # input parameter:
        # inputs: input of the Dense layer, could be the dataset or an output from a previous Dense layer
        self.inputs = inputs
        self.dense_output = np.dot(self.inputs, self.weights) + self.biases.T #dot product with the inputs the the weights added with the biases
        self.output = self.dense_output

    def back_pass(self, prev_grad): #back pass function
        # input parameter:
        # prev_grad: gradient derived from backpropagated layers
        
        #initialise variables for regularizers
        d_l1=np.ones_like(self.weights)
        d_l1[d_l1<0]=-1
        d_l2=self.weights
        
        #calculated derivative of weights and biases to be updated for this layer
        self.d_weights = np.dot(self.inputs.T, prev_grad)
        self.d_bias = np.sum(prev_grad, axis=0, keepdims=True)
        
        #regularizers
        if(self.regularizer=='none'):
            pass
        elif(self.regularizer=='l1'):
            self.d_weights += self.lamda*d_l1
        elif(self.regularizer=='l2'):
           self.d_weights += self.lamda*d_l2*2
        elif(self.regularizer=='l1l2'):
            self.d_weights += self.lamda*(2*d_l2+d_l1)
        else:
            raise Exception('Unexpected error occured during regularization backpass')

        #calculate the gradient to be derived for the frontal layers
        self.current_grad = np.dot(prev_grad, self.weights.T) 
        
# Layer with a Relu activation function
class Relu_Layer(Dense_Layer):
    def __init__(self, num_inputs, num_neurons, has_dropout, keep_rate, regularizer, lamda):
        # inherit parameters from the Dense layer
        super().__init__(num_inputs, num_neurons, has_dropout, keep_rate, regularizer, lamda) 

    def forward_pass(self, inputs, training=True):
        # forward passes through the Dense layer
        super().forward_pass(inputs) 
        
        # Call Relu activation function in addition to the Dense layer forward pass
        self.output = np.maximum(0, self.dense_output)
        
        #if dropout exists, commence dropout
        if self.has_dropout and training: 
            super().dropout()

    def back_pass(self, prev_grad):
        # derivitve of activation function
        self.d_activation = prev_grad.copy()
        
        #sets gradient to 0 only if the output of the ReLU layer is <0, as per the formula for the derivative of the ReLU layer
        self.d_activation[self.dense_output <= 0] = 0
        
        # derivative of dense layer
        super().back_pass(self.d_activation)

# Layer with a Sigmoid activation function
class Sigmoid_Layer(Dense_Layer):
    def __init__(self, num_inputs, num_neurons, has_dropout, keep_rate, regularizer, lamda):
        #inherit variables
        super().__init__(num_inputs, num_neurons, has_dropout, keep_rate, regularizer, lamda)

    def sigmoid(self, x): #define sigmoid layer
        return np.where(x>=0, 1/(1+np.exp(-1*x)), np.exp(x)/(1+np.exp(x))) #return sigmoid values depending on the sign of the input to prevent overflow errors
        # return 1/(1+np.exp(-1*x))
    
    def forward_pass(self, inputs, training=True):
        # forward passes through the Dense layer
        super().forward_pass(inputs)
        
        # Call Sigmoid activation function
        self.output = self.sigmoid(self.dense_output)
        
        #if dropout exists, commence dropout
        if self.has_dropout and training:
            super().dropout()

    def back_pass(self, prev_grad):
        # derivitve of sigmoid layer, as per the formula
        self.d_activation = self.sigmoid(self.dense_output) * (1 - self.sigmoid(self.dense_output))
        
        # multiply with the previously derived gradients from the previous layers as per the chain rule
        self.d_activation *= prev_grad
        
        # derivative of dense layer
        super().back_pass(self.d_activation)

# Layer with a Softmax activation function
class Softmax_Layer(Dense_Layer):
    def __init__(self, num_inputs, num_neurons, regularizer, lamda, has_dropout=False, keep_rate=1.0,):
        #inherit variables from the Dense layer
        super().__init__(num_inputs, num_neurons, has_dropout, keep_rate, regularizer, lamda)
        

    def softmax(self, x, axis_val):
        # input parameters:
        # x: input of the layer
        # axis_val: binary value representing the activation function used
        
        if axis_val == 1:
            exp_vals = np.exp(x- np.max(x, axis=1, keepdims=True))
            softmax_out = exp_vals / np.sum(exp_vals, axis=1, keepdims=True)
        else:
            exp_vals = np.exp(x- np.max(x, axis=0, keepdims=True))
            softmax_out = exp_vals / np.sum(exp_vals, axis=0, keepdims=True)
        return softmax_out

    def forward_pass(self, inputs, axis_val, training=True):
        # Work out dense values
        super().forward_pass(inputs)
        
        # Call Softmax activation function
        self.output = self.softmax(self.dense_output, axis_val)
        
        #if dropout exists, commence dropout
        if self.has_dropout and training:
            super().dropout()

    def back_pass(self, prev_grad):
        super().back_pass(prev_grad)

class Categorigal_Cross_Entropy_Loss:
    def loss_forward_pass(self, pred_y, true_y):
        # clip the predictions to prevent mathematical errors
        pred_y_clipped = np.clip(pred_y, 1e-7, 1 - 1e-7)
        
        # calculate the negative log likelihood
        pred_confidences = -np.sum(np.log(pred_y_clipped) * true_y, axis=1)/pred_y.shape[0]

        return pred_confidences

    def back_pass(self, prev_grad, true_y):
        # calculate the gradient of the loss function as per the formula of categorical cross entropy loss
        self.current_grad = -true_y / prev_grad
        
        # normalise the gradient
        self.current_grad = self.current_grad / len(prev_grad)

class Softmax_CategoricalCrossEntroyLoss(Softmax_Layer):
    def __init__(self, num_inputs, num_neurons, regularizer, lamda):
        # inherits variables from the Softmax layer
        super().__init__(num_inputs, num_neurons, regularizer, lamda)
        self.loss = Categorigal_Cross_Entropy_Loss()
    
    def loss_forward_pass(self, inputs, targets, axis_val, training=True):
        # input parameters:
        # inputs: input data
        # targets: input label
        # axis_val: binary value of activation functions
        # training: determines if model is training or evaluating to implement dropout
        
        # forward pass for loss function
        super().forward_pass(inputs, axis_val, training)
        return self.loss.loss_forward_pass(self.output, targets)
    
    def loss_back_pass(self, prev_grad, true_y):
        # reverts the true labels to its original state
        self.true_y = np.argmax(true_y, axis=1)

        # copy gradient to prevent pointer issues
        self.current_grad = prev_grad.copy()
        
        # subtract the index of the true label by 1 from the prediction array
        self.current_grad[range(len(prev_grad)), self.true_y] -= 1
        
        # normalise gradients
        self.current_grad = self.current_grad / len(prev_grad)
        
        # back pass to previous layers
        super().back_pass(self.current_grad)

# class for Stochastic Gradient Descent
class Stochasic_Gradient_Descent:
    def __init__(self, learning_rate, has_momentum, momentum):
        # input parameters:
        # learning_rate: learning rate when updating weights and biases
        # has_momentum: binary value to determine if momentum gradient descent is used
        # momentum: momemntum value for momentum gradient descent
        
        self.learning_rate = learning_rate
        self.has_momentum=has_momentum
        self.momentum = momentum

    def update_layer(self, layer):
        # input parameters
        # layer: layer to be updated weights and biases
        
        # update weights and biases with respect to the momentum
        if self.has_momentum:
            weight_update_amount = self.momentum * layer.current_weight_momentum - self.learning_rate * layer.d_weights
            layer.current_weight_momentum = weight_update_amount
            bias_update_amount = self.learning_rate * layer.d_bias.T
        else:
            weight_update_amount = self.learning_rate * layer.d_weights
            bias_update_amount = self.learning_rate * layer.d_bias.T
        layer.weights += weight_update_amount           
        layer.biases += bias_update_amount

class NN:
    def __init__(self, optimizer='sgd', learning_rate=1e-3, is_decay=False, learning_decay=5, gamma=1, has_momentum=False, momentum=0, hpt=None):
        self.layers = []
        self.losslog=[]
        self.is_decay=is_decay
        self.learning_decay=learning_decay
        self.gamma=gamma
        self.hpt = hpt #variable for hyperparameter training
        
        if optimizer == 'sgd':
            self.opt = Stochasic_Gradient_Descent(learning_rate, has_momentum, momentum)
        else:
            raise Exception('Not valid optimizer')
        
    def accuracy(self, pred_y, true_y):
        #find the accuracy of the predcitions with respect to the targets
        predictions = np.argmax(pred_y, axis=1)
        argmax_targets = np.argmax(true_y, axis=1)
        accuracy = np.mean(predictions==argmax_targets)
        return accuracy

    def add_layer(self, num_inputs, num_neurons, activation_function='none', has_dropout=False, keep_rate=1.0, regularizer='none', lamda=0):
        # add layers depending on arbitrary inputs
        if (len(self.layers) != 0) and (self.layers[-1].num_neurons != num_inputs):
            raise Exception("The number of outputs from the previous layer do not match the inputs of the layer you are attempting to add.")
        elif (len(self.layers) != 0) and type(self.layers[-1]) == Softmax_CategoricalCrossEntroyLoss:
            raise Exception("Softmax output layer already assigned.")
        else:
            if activation_function == "relu":
                self.layers.append(Relu_Layer(num_inputs, num_neurons, has_dropout, keep_rate, regularizer, lamda))
            elif activation_function == "sigmoid":
                self.layers.append(Sigmoid_Layer(num_inputs, num_neurons, has_dropout, keep_rate, regularizer, lamda))
            elif activation_function == "softmax_output":
                if has_dropout:
                    raise Exception("Softmax can't have dropout")
                else:
                    self.layers.append(Softmax_CategoricalCrossEntroyLoss(num_inputs, num_neurons, regularizer, lamda))
            elif activation_function == "none":
                self.layers.append(Dense_Layer(num_inputs, num_neurons, has_dropout, keep_rate, regularizer, lamda))
            else:
                raise Exception("Invalid activation function provided.")

    def fit(self, data, batch_size, epochs):
        #fit model based on data and batch_size using mini-batch gradient descent
        if self.layers[0].num_inputs != data.train_X.shape[1]:
            raise Exception("Shape of inputs does not match first layers input")
        elif type(self.layers[-1]) != Softmax_CategoricalCrossEntroyLoss:
            raise Exception("Output layer needs to be softmax")
        else:
            self.axis_val = 1 if (type(self.layers[-2]) == Relu_Layer) else 0
            print(f'axis value {self.axis_val}')
            self.training_accuracies = []
            self.b_inputs, self.b_targets = data.create_batches_train(batch_size)
            for epoch in range(epochs):
                if(self.is_decay): #Check if model is set for learning rate decay
                    if(epoch%self.learning_decay==0): #check with the timing of the learning rate decay
                        self.opt.learning_rate*=self.gamma #decay learning rate
                self.b_outputs = []
                for b_index, batch in enumerate(self.b_inputs): #iterate through batches
                    self.output = batch
                    
                    #commence forward pass
                    for index, layer in enumerate(self.layers): #iterate through layers
                        if index == len(self.layers) - 1: #indicating "loss layer", i.e. the loss function after the last layer, typically the softmax function
                            self.loss = layer.loss_forward_pass(self.output, self.b_targets[b_index], self.axis_val)
                            self.losslog.append(self.loss)
                        else: #regular forward passing
                            layer.forward_pass(self.output)
                        self.output = layer.output #final output is the prediction
                    self.b_outputs.append(self.output) #append as batches
                    
                    #commence backpass, loop through layers backwards
                    for layer_index in range(len(self.layers) - 1, -1, -1):
                        layer = self.layers[layer_index]
                        if layer_index == len(self.layers) - 1:
                            layer.loss_back_pass(self.b_outputs[b_index], self.b_targets[b_index]) 
                        else:
                            layer.back_pass(self.layers[layer_index + 1].current_grad)
                            
                    #update step
                    for layer_index in range(len(self.layers) - 1, -1, -1):
                        self.opt.update_layer(layer)
                        
                    # If the model is part of a hyper parameter training run, create cheackpoint/copies of the model at specified epochs in order to test accuracy at those epochs later
                    if self.hpt != None:
                        if self.hpt.epoch_checkmarks[self.hpt.epoch_ptr] == epoch:
                            self.hpt.copy_nn(self)
                    self.accuracy(self.b_outputs[b_index], self.b_targets[b_index])
                self.training_accuracies.append(self.accuracy(self.output, self.b_targets[b_index]))
                print(f'\nEpoch: {epoch + 1} | Accuracy: {round(self.accuracy(self.output, self.b_targets[b_index])*100)}%')
                print(f'\nEpoch: {epoch + 1} | Loss: {np.sum(self.loss)}')
 

    def test(self, data, batch_size):
        batches_X, batches_y = data.create_batches_test(batch_size) #create number of batches based on batch size
        accuracies = [] #accuracy log
        for b_index, batch in enumerate(batches_X):
            self.output = batch
            for index, layer in enumerate(self.layers):
                if index == len(self.layers) - 1:
                    layer.loss_forward_pass(self.output, batches_y[b_index][index], self.axis_val, training=False) #forward pass based on batches
                else:
                    layer.forward_pass(self.output, training=False)
                self.output = layer.output
            accuracies.append(self.accuracy(self.output, batches_y[b_index])) #log accuracy changes
            print(f'Batch: {b_index + 1} | Accuracy: {round(self.accuracy(self.output, batches_y[b_index])*100, 2)}') #print batch and accuracy
        print(f'=============== Test Results ===============')
        print(f'Average accuracy: {np.mean(accuracies)}')
        print('============================================')
        return np.mean(accuracies)

# crete a data object
data = Data(training_data, testing_data)


class Hyper_Param_Trainer:
    def __init__(self, filename, data, learning_rates, epoch_values, num_layers, activation_functions, neuron_values, dropout_values, regularizers, lamdas):
        self.filename = filename
        # data manipulation before passing into NN
        self.data = data
        self.data.reshape()
        self.data.one_hot_encode_data()
        self.data.scale_data('standard')
        self.epoch_checkmarks = epoch_values
        self.num_layers = num_layers

        # Options for each regularization
        self.regularizer_configs = []
        for regularizer in regularizers:
            for lamda in lamdas:
                self.regularizer_configs.append({
                    "regularizer": regularizer,
                    "lamda": lamda
                })

        # configure each layer in the neural network, each hidden layer in the NN would be the same
        self.layer_configs = []
        for activation_function in activation_functions:
            for neuron_value in neuron_values:
                for dropout_value in dropout_values:
                        for regularizer in regularizers:
                            if regularizer != 'none':
                                for lamda in lamdas:
                                    self.layer_configs.append({
                                        "layers": num_layers,
                                        "activation_function": activation_function,
                                        "neuron_value": neuron_value,
                                        "dropout_value": dropout_value,
                                        "regularizer": regularizer,
                                        "lamda": lamda,
                                    })
                            else:
                                self.layer_configs.append({
                                        "layers": num_layers,
                                        "activation_function": activation_function,
                                        "neuron_value": neuron_value,
                                        "dropout_value": dropout_value,
                                        "regularizer": regularizer
                                    })

        # Configure variables the are passed in when model is initialised
        self.model_configs = []
        for learning_rate in learning_rates:
            self.model_configs.append({
                "learning_rate": learning_rate
            })

        # Complete configurations for each model that will be run, creates templates for each model and a location to store resulting values returned by the models, this is the object later stored in a json file
        self.all_runs_configs = []
        for model_config in self.model_configs:
            for layer_config in self.layer_configs:
                run_data = {}
                for key, value in model_config.items():
                    run_data[key] = value
                for key, value in layer_config.items():
                    run_data[key] = value
                self.all_runs_configs.append(run_data)
        print(len(self.all_runs_configs))

    # An interface to be called by the models passing itself in order to create a deep copy. 
    def copy_nn(self, nn):
        self.current_run_models.append(copy.deepcopy(nn))
        self.epoch_ptr += 1

        # Keeps track of the hyper paramtere trainers progress, purely for development convenience
        self.progress = 1

    # Core of the class, loops through all configurations and creates, fits and runs tests of the models
    def run(self):
        for rc in self.all_runs_configs:
            print(f'========================== On Model config {self.progress} of {len(self.all_runs_configs)}')
            print(f'Running: {rc}')
            self.epoch_ptr = 0
            # create mdoel
            my_nn = NN(learning_rate=rc["learning_rate"], is_decay=False, learning_decay=5, gamma=0.99, has_momentum=True, momentum=0.5, hpt=self)
            # each model need at least 1 hidden layer
            my_nn.add_layer(data.train_X.shape[-1], rc["neuron_value"], rc["activation_function"], has_dropout=True if (rc["dropout_value"] != 1) else False, keep_rate=rc["dropout_value"], regularizer=rc["regularizer"], lamda=None if (rc["regularizer"] == 'none') else rc["lamda"])
            for l in range(self.num_layers-1):
                my_nn.add_layer(rc["neuron_value"], rc["neuron_value"], rc["activation_function"], has_dropout=True if (rc["dropout_value"] != 1) else False, keep_rate=rc["dropout_value"], regularizer=rc["regularizer"], lamda=None if (rc["regularizer"] == 'none') else rc["lamda"])
            my_nn.add_layer(rc["neuron_value"], 10, "softmax_output", has_dropout=False, keep_rate=1, regularizer=rc["regularizer"], lamda=None if (rc["regularizer"] == 'none') else rc["lamda"])
            # array to store the copies of the model at given epochs
            self.current_run_models = []
            my_nn.fit(data, batch_size=8, epochs=self.epoch_checkmarks[-1])
            rc["training_accuracies"] = my_nn.training_accuracies
            # array to store test results at each epoch
            rc[f'testing_averages'] = []
            for index, nn in enumerate(self.current_run_models):
                rc[f'testing_averages'].append({"epoch": self.epoch_checkmarks[index], "test_average": nn.test(data, batch_size=8)})
            # final epoch value is when the model finishes running so the test function is called on the initial model instead of a stored copy
            rc[f'testing_averages'].append({"epoch": self.epoch_checkmarks[-1], "test_average": my_nn.test(data, batch_size=8)})

            # save results to json file
            with open(f'{self.filename}.json', "w") as results_file:
                json.dump(self.all_runs_configs, results_file)

            self.progress += 1

# Example running hyper parameter training with paramters used in the report
# parameters are in the following order: file name or json to store in, unchanged data, before one hot encoding or scaling, learning rates, epochs, number of layers, activation functions, number of neurons per layer, dropout rate, regulerizers, lambdas
hpt = Hyper_Param_Trainer('task1-all-results', data, [1e-2, 1e-3, 1e-4], [32, 64, 128, 256], 3, ["sigmoid", "relu"], [64, 128, 256], [0.8, 1], ['l1', 'l2'], [0.01])
hpt.run()