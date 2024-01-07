Task 1:

The data can be managed as follows:
    data = Data(training_data, testing_data)
    data.reshape()
    data.one_hot_encode_data()
    data.scale_data('standard')

Initialize neural network example:
    my_nn = NN(learning_rate, is_decay=False, learning_decay=5, gamma=0.99, has_momentum=True, momentum=0.5, hpt=self)
Add layers:
    my_nn.add_layer(input_shape, amount_of_neurons, activation_function, has_dropout=False, keep_rate=1, regularizer='l2', lamda=0.01)
activation function is a string of either 'relu' or 'sigmoid' or for the output layer 'softmax'

Train the model with:
    my_nn.fit(data, batch_size=8, epochs=128)
Testing the data:
    my_nn.test(data, batch_size=8)

Example of the hyper parameter training is at the bottom of the task 1 file.

Task 2:

Task 2 does not use any seeds, the code if organised and can be replicated by stepping throught the cells.