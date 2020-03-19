# Created by Ashwin Vinoo
# Date 2/27/2020

# This code implements a recurrent neural network that takes the last 12 observations (Lidar data) and actions
# (velocity, steering angle) to predict the current state and next state.

# --------------- IMPORTING THE NECESSARY MODULES ----------------

# importing the necessary modules
from __future__ import print_function
from __future__ import division
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as opt
import torch.nn.functional as tf
from torch.utils import data
from torch.autograd import Variable

# --------------- HYPER PARAMETERS ----------------

# The name of the file from which we obtain the dynamics data to train
dynamics_data_file = "dynamics_data_rnn.csv"
# Indicates whether we want to load the pretrained neural network state dictionary
load_trained_network_dict = True
# The name of the file from which we can load the state dictionary
trained_network_state_dict_file = 'trained_network_dict_rnn.pth'
# The name of the file from which we can load the dynamics data parameters
dynamics_data_parameters_file = 'dynamics_data_parameters_rnn.pkl'
# The ratio of training data size to the total available. Remaining will be test data
train_ratio = 0.8
# The optimizer to use for training
optimizer_to_use = 'ADAGRAD'
# The optimizer learning rate
optimizer_learning_rate = 0.03
# The loss function to use for training
loss_function_to_use = 'MSE'
# The maximum number of epochs to train the data
training_epochs = 10
# This is the mini_batch size to be used while training/evaluating the data
batch_size = 64
# This is the sequence length that determines how far back in time steps we back propagate
sequence_length = 12
# The number of RNN layers
rnn_layers = 3
# The number of hidden cells in each rnn layer
rnn_hidden = 1024
# The percentage of previous value to which we drop the optimizer learning rate every epoch
optimizer_learning_drop_percentage = 0.9

# --------------- DEEP NEURAL NETWORK FOR DYNAMICS MODELING ----------------


# The neural network class which inherits from torch.nn.Module class
class DynamicsModelRNN(nn.Module):

    # The class constructor
    def __init__(self):
        # We call init of the super class
        nn.Module.__init__(self)
        # The rnn layers
        self.rnn_1 = nn.LSTM(input_size=62, hidden_size=rnn_hidden, num_layers=rnn_layers, dropout=0, batch_first=True)
        # The batch normalization layers
        self.batch_norm_1 = nn.BatchNorm1d(rnn_hidden)
        self.batch_norm_2 = nn.BatchNorm1d(512)
        # The linear transform layer
        self.linear_1 = nn.Linear(rnn_hidden, 512)
        self.linear_2 = nn.Linear(512, 6)
        # We check which device is available to run the neural network - GPU else CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Uninitialized optimizer and criterion(loss function) used to train the neural network
        self.optimizer = None
        self.criterion = None
        # Uninitialized min and max values of the dynamics dataset are stored here
        self.min_dataset_values = None
        self.max_dataset_values = None

    # This function specifies the working of the forward pass
    def forward(self, x, hidden, cell):
        # We pass the data through the LSTM layer
        x, (hidden, cell) = self.rnn_1(x, (hidden, cell))
        # We convert it to a form so that it can be batch normalized
        x = x.permute(0, 2, 1)
        # We pass the output for batch normalization
        x = self.batch_norm_1(x)
        # We convert it back to the original form
        x = x.permute(0, 2, 1)
        # ReLU activation unit
        x = tf.relu(x)
        # linear transform layer one
        x = self.linear_1(x)
        # We convert it to a form so that it can be batch normalized
        x = x.permute(0, 2, 1)
        # We pass the output for batch normalization
        x = self.batch_norm_2(x)
        # We convert it back to the original form
        x = x.permute(0, 2, 1)
        # ReLU activation unit
        x = tf.relu(x)
        # linear transform layer one
        x = self.linear_2(x)
        # We only care about the output from the last sequence
        x = x[:, -1, :]
        # We return the hidden outputs and the output of the feed forward layers
        return x, hidden, cell

    # This functions adjust the learning rate of the optimizer to the specified percentage
    def adjust_learning_rate(self, learning_drop=0.5):
        # We move through the parameter groups of the optimizer
        for param_group in self.optimizer.param_groups:
            # The learning rate drops to the specified percentage of its original value
            param_group['lr'] *= learning_drop

    # This is a function to get predictions using the trained network.
    # Input is a numpy matrix of rows equal to the sequence number and 62 columns containing actions + observations
    def get_dynamics_predictions(self, input_dataset):
        # Making the network to be in evaluation mode so that losses won't be accumulated, thereby saving memory
        self.eval()
        # We convert to float32 in case its isn't in that format
        input_dataset = np.float32(input_dataset)
        # We normalize the test dataset based on the recorded parameters of the dynamics dataset
        input_dataset = ((input_dataset - self.min_dataset_values[3:65]) /
                         (self.max_dataset_values[3:65] - self.min_dataset_values[3:65]))
        # We convert the dataset into a tensor
        input_dataset = torch.from_numpy(input_dataset)
        # We convert the input dataset to [batch size=1, sequence length, input features] format
        input_dataset = input_dataset.view((1, sequence_length, 62))
        # Convert to nn variable. cuda() creates another Variable that isn’t a leaf node in the computation graph
        input_dataset = Variable(input_dataset).to(self.device)
        # We initialize the hidden layers of the RNN to zeros
        hidden = Variable(torch.zeros(rnn_layers, 1, rnn_hidden)).to(self.device)
        # We initialize the cell state of the RNN to zeros
        cell = Variable(torch.zeros(rnn_layers, 1, rnn_hidden)).to(self.device)
        # The output obtained from the neural network
        output, _, _ = self(input_dataset, hidden, cell)
        # We convert the output to 1D format
        output = torch.flatten(output)
        # We convert the dataset back to a numpy matrix and return
        output = output.detach().cpu().numpy()
        # The maximum values associated with the output
        max_values = np.concatenate((self.max_dataset_values[0:3], self.max_dataset_values[65:68]))
        # The minimum values associated with the output
        min_values = np.concatenate((self.min_dataset_values[0:3], self.min_dataset_values[65:68]))
        # We de-normalize the output from the neural network
        output = (output * (max_values - min_values) + min_values)
        # We return the output dataset
        return output

    # This function loads the network state dictionary from the path specified
    def load_network_state_dictionary(self, network_state_dict_path):
        # Loads the pretrained dictionary
        self.load_state_dict(torch.load(network_state_dict_path))

    # This function loads the minimum and maximum values of the datasets. Helps normalize data during application
    def load_dynamics_dataset_parameters(self, dynamics_dataset_params_path):
        # The minimum and maximum dataset values
        [self.min_dataset_values, self.max_dataset_values] = pickle.load(open(dynamics_dataset_params_path, "rb"))

    # This function automatically handles training of the neural network using normalized data and labels
    def norm_train_network(self, train_dataset, epochs):
        # Converting the network back into training mode
        self.train()
        # The shift parameter helps us shift the dataset by one after each epoch
        shift = 0
        # loop over the dataset multiple times till we cover all the epochs with shifts up to the sequence length
        for epoch in range(epochs * sequence_length):
            # The total loss of the last cell of the sequence is initialized to zero
            last_cell_loss = 0.0
            # The current training dataset index we are pointing at
            index = shift
            # We use this variable to count the number of iterations
            iteration = 0
            # Enumerating through the training dataset from the shifted index
            while index + batch_size * sequence_length <= train_dataset.__len__():
                # We create tensors to hold the inputs
                inputs = torch.empty([batch_size, sequence_length, 62], dtype=torch.float32)
                # We create tensors to hold the labels
                labels = torch.empty([batch_size, sequence_length, 6], dtype=torch.float32)
                # We iterate through the number of elements in the batch size
                for i in range(batch_size):
                    # We iterate through the number of elements in the sequence length
                    for j in range(sequence_length):
                        # We obtain the inputs and labels required to train the neural network
                        inputs[i, j, :], labels[i, j, :] = train_dataset[index + i*sequence_length + j]
                # We wrap the inputs to a torch variable and push to the device
                inputs = Variable(inputs).to(self.device)
                # We wrap labels to a torch variable and push to the device
                labels = Variable(labels).to(self.device)
                # We are only interested in the label for the last cell of the sequence
                labels = labels[:, -1, :]
                # We initialize the hidden layers of the RNN to zeros
                hidden = Variable(torch.zeros(rnn_layers, batch_size, rnn_hidden)).to(self.device)
                # We initialize the cell state of the RNN to zeros
                cell = Variable(torch.zeros(rnn_layers, batch_size, rnn_hidden)).to(self.device)
                # obtain the outputs of the neural network
                outputs, hidden, cell = self(inputs, hidden, cell)
                # zero the parameter gradients
                self.optimizer.zero_grad()
                # Calculate the loss between outputs and labels
                loss = self.criterion(outputs, labels)
                # Perform backward propagation to calculate the gradients at every layer
                loss.backward()
                # Optimizer changes the weights at every weight according to stochastic gradient descent
                self.optimizer.step()
                # We update the loss at the last cell  of the sequence
                last_cell_loss += loss.data
                # We shift the index since we read the above values
                index += batch_size * sequence_length
                # We increment the iteration count
                iteration += 1
            # We print the results to show the user the training progress
            print('    Epochs Completed: %2d, MSE loss at last cell: %.4f'
                  % (np.floor(epoch / sequence_length), last_cell_loss/iteration))
            # We drop the learning rate every epoch
            self.adjust_learning_rate(optimizer_learning_drop_percentage)
            # We shift the starting index after every epoch
            shift = (shift + 1) % sequence_length

    # This is a function to evaluate the neural network using normalized data and labels
    def norm_evaluate_network(self, test_dataset):
        # Making the network to be in evaluation mode so that losses won't be accumulated, thereby saving memory
        self.eval()
        # The total loss of the sequence is initialized to zero
        total_sequence_loss = 0.0
        # Iterating through the data is data_loader
        for index in range(0, test_dataset.__len__(), sequence_length):
            # We ignore cases that don't have enough entries to match the sequence length
            if index + sequence_length >= test_dataset.__len__():
                # We break the code execution
                break
            # We initialize the hidden layers of the RNN to zeros
            hidden = Variable(torch.zeros(rnn_layers, 1, rnn_hidden)).to(self.device)
            # We initialize the cell state of the RNN to zeros
            cell = Variable(torch.zeros(rnn_layers, 1, rnn_hidden)).to(self.device)
            # Splitting into images and labels
            inputs, labels = test_dataset[index:(index + sequence_length)]
            # We convert inputs to the form [batch, sequence number, input]
            inputs = inputs.view((1, sequence_length, -1))
            # We convert labels to the form [batch, sequence number, input]
            labels = labels.view((1, sequence_length, -1))
            # We are only interested in the label for the last cell of the sequence
            labels = labels[:, -1, :]
            # Convert to nn variable. cuda() creates another Variable that isn’t a leaf node in the computation graph
            inputs, labels = Variable(inputs).to(self.device), Variable(labels).to(self.device)
            # obtain the outputs of the neural network
            outputs, hidden, cell = self(inputs, hidden, cell)
            # Calculate the loss between outputs and labels
            loss = self.criterion(outputs, labels)
            # We update the running loss
            total_sequence_loss += loss.data
        # Prints the average loss over the normalized testing dataset
        print('    Average Loss Over the Test Dataset is: %.4f' % (total_sequence_loss/test_dataset.__len__()))

# --------------- DATASET CLASS FOR HANDLING THE DYNAMICS DATA ----------------


# We create this class that inherits from dataset class to simplify access to train and test datasets
class DatasetHandler(data.Dataset):

    # Initializes the constructor for the dataset class
    def __init__(self, dataset):
        # We get the length of the dataset
        self.length = len(dataset)
        # We obtain the inputs going to the neural network
        self.inputs = torch.from_numpy(dataset[:, 3:65])
        # We obtain the labels of the output of the neural network - current state and next state
        self.labels = torch.from_numpy(np.concatenate((dataset[:, 0:3], dataset[:, 65:68]), axis=1))

    # This function returns the length of the dataset
    def __len__(self):
        return self.length

    # Getter function for accessing the elements of the dataset
    def __getitem__(self, index):
        # Returns the items of the dataset
        return self.inputs[index], self.labels[index]

# --------------- MAIN CODE BLOCK ----------------

# If this file is the main one called for execution
if __name__ == "__main__":

    # ----- Loading the dynamics dataset -----
    # We set a seed to get repeatable results in random operations
    np.random.seed(1)
    # Gets the directory of this python file
    dir_path = os.path.dirname(os.path.realpath(__file__))
    # Reading the dynamics data from the csv file. Skipping the header row
    dynamics_data = np.genfromtxt(dir_path + "\\dynamics_data\\" + dynamics_data_file,
                                  delimiter=',', dtype=np.float32, skip_header=1)
    # We obtain the number of samples in the dynamics dataset
    dynamics_data_samples_count = dynamics_data.shape[0]
    # We obtain the number of training samples
    training_samples_count = int(dynamics_data_samples_count * train_ratio)
    # We obtain the number of test samples
    test_samples_count = dynamics_data_samples_count - training_samples_count
    # We obtain the list of rows that we treat as training samples
    training_row_list = list(range(training_samples_count))
    # We obtain the list of rows we will treat as test samples
    test_row_list = list(range(training_samples_count, dynamics_data_samples_count))

    # ----- Normalizing the dataset -----
    # We obtain the minimum value of each column
    min_data = np.min(dynamics_data, axis=0)
    # We obtain the maximum value of each column
    max_data = np.max(dynamics_data, axis=0)
    # We normalize the data to between -0.5 and 0.5
    dynamics_data = (dynamics_data - min_data)/(max_data - min_data)
    # We dump the min and max data into a pickle file for use post training
    with open(dir_path + "\\trained_network\\" + dynamics_data_parameters_file, 'wb') as pickle_file:
        pickle.dump([min_data, max_data], pickle_file)

    # ----- Obtaining the training and testing datasets along with their loaders-----
    # extracting the training dataset
    training_dataset = DatasetHandler(dynamics_data[training_row_list, :])
    # extracting the training dataset
    testing_dataset = DatasetHandler(dynamics_data[test_row_list, :])

    # ----- Building the model -----
    print('Building model...')
    # Build the neural network model
    net = DynamicsModelRNN().cuda()
    # Choosing the different types of loss functions available
    if loss_function_to_use == 'MAE':
        # The loss function which is used as the training criterion
        net.criterion = nn.L1Loss()
    else:
        # The loss function which is used as the training criterion
        net.criterion = nn.MSELoss()
    # Choosing the different types of optimizers available
    if optimizer_to_use == 'ADAM':
        # The optimizer to use during training is Adam
        net.optimizer = opt.Adam(net.parameters(), lr=0.01)
    elif optimizer_to_use == 'RMSPROP':
        # The optimizer to use during training is RMSprop
        net.optimizer = opt.RMSprop(net.parameters(), lr=0.01, momentum=0.9)
    elif optimizer_to_use == 'ADAGRAD':
        # The optimizer to use during training is Adagrad
        net.optimizer = opt.Adagrad(net.parameters(), lr=0.01)
    else:
        # The optimizer to use during training is stochastic gradient descent
        net.optimizer = opt.SGD(net.parameters(), lr=0.01, momentum=0.9)

    # ----- Loading a pretrained network state dictionary if requested -----

    # Check if we wanted to load the trained network state dictionary
    if load_trained_network_dict:
        print('Loading the pretrained network state dictionary')
        # We load a trained network state dictionary from the path specified
        net.load_network_state_dictionary(dir_path + "\\trained_network\\" + trained_network_state_dict_file)

    # ----- Training the model -----

    print('Start training the network...')
    # We train the neural network for the number of epochs specified
    net.norm_train_network(training_dataset, training_epochs)

    # ----- Evaluating the model -----

    print('Start evaluating the network...')
    # We evaluate the neural network for the number of epochs specified
    net.norm_evaluate_network(training_dataset)

    print('Saving the network state dictionary...')
    # Save the neural network model to the specified path
    torch.save(net.state_dict(), dir_path + "\\trained_network\\" + trained_network_state_dict_file)
    print('Execution Completed')

# --------------- END OF CODE ----------------
