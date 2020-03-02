# Created by Ashwin Vinoo
# Date 2/27/2020

# This code implements a deep neural network that takes the position (x,y, heading) and the driving parameters
# (velocity, steering angle) to return the position at the next time instance (x_next, y_next, heading_next)

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
dynamics_data_file = "dynamics_data.csv"
# Indicates whether we want to load the pretrained neural network state dictionary
load_trained_network_dict = True
# The name of the file from which we can load the state dictionary
trained_network_state_dict_file = 'trained_network_dict.pth'
# The name of the file from which we can load the dynamics data parameters
dynamics_data_parameters_file = 'dynamics_data_parameters.pkl'
# The ratio of training data size to the total available. Remaining will be test data
train_ratio = 0.8
# The optimizer to use for training
optimizer_to_use = 'ADAGRAD'
# The optimizer learning rate
optimizer_learning_rate = 0.01
# The loss function to use for training
loss_function_to_use = 'MSE'
# The maximum number of epochs to train the data
training_epochs = 5
# This is the mini_batch size to be used while training/evaluating the data
batch_size = 32
# The percentage of previous value to which we drop the optimizer learning rate every epoch
optimizer_learning_drop_percentage = 0.5

# --------------- DEEP NEURAL NETWORK FOR DYNAMICS MODELING ----------------


# The neural network class which inherits from torch.nn.Module class
class DynamicsModelDNN(nn.Module):

    # The class constructor
    def __init__(self):
        # We call init of the super class
        nn.Module.__init__(self)
        # Two linear transform layers
        self.fc1 = nn.Linear(5, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 3)
        # The batch normalization layers
        self.batch_norm_1 = nn.BatchNorm1d(512)
        self.batch_norm_2 = nn.BatchNorm1d(128)
        # We check which device is available to run the neural network - GPU else CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Uninitialized optimizer and criterion(loss function) used to train the neural network
        self.optimizer = None
        self.criterion = None
        # Uninitialized min and max values of the dynamics dataset are stored here
        self.min_dataset_values = None
        self.max_dataset_values = None

    # This function specifies the working of the forward pass
    def forward(self, x):
        # linear transform layer one
        x = self.fc1(x)
        # ReLU activation unit
        x = tf.leaky_relu(x)
        # Applies 1D batch normalization
        x = self.batch_norm_1(x)
        # Linear transform layer two
        x = self.fc2(x)
        # ReLU activation unit
        x = tf.leaky_relu(x)
        # Applies 1D batch normalization
        x = self.batch_norm_2(x)
        # Linear transform layer four
        x = self.fc3(x)
        # We return the output of the neural network
        return x

    # This functions adjust the learning rate of the optimizer to the specified percentage
    def adjust_learning_rate(self, learning_drop=0.5):
        # We move through the parameter groups of the optimizer
        for param_group in self.optimizer.param_groups:
            # The learning rate drops to the specified percentage of its original value
            param_group['lr'] *= learning_drop

    # This is a function to get predictions using the trained network. Input is a numpy matrix of 5 columns and any rows
    def get_dynamics_predictions(self, input_dataset):
        # Making the network to be in evaluation mode so that losses won't be accumulated, thereby saving memory
        self.eval()
        # We normalize the test dataset based on the recorded parameters of the dynamics dataset
        input_dataset = ((input_dataset - self.min_dataset_values[0:5]) /
                         (self.max_dataset_values[0:5] - self.min_dataset_values[0:5]))
        # We convert the dataset into a tensor
        input_dataset = torch.from_numpy(input_dataset)
        # We obtain the dimensionality of the dataset
        data_dimension = len(input_dataset.shape)
        # We check if the input dataset is a 1D tensor
        if data_dimension == 1:
            # We make it a 2D tensor
            input_dataset = torch.unsqueeze(input_dataset, 0)
        # Convert to nn variable. cuda() creates another Variable that isn’t a leaf node in the computation graph
        input_dataset = Variable(input_dataset).to(self.device)
        # The output obtained from the neural network
        output_dataset = self(input_dataset)
        # We check if the input dataset was originally a 1D tensor
        if data_dimension == 1:
            # We make it a 1D tensor again
            output_dataset = output_dataset[0]
        # We convert the dataset back to a numpy matrix and return
        output_dataset = output_dataset.detach().cpu().numpy()
        # We de-normalize the output from the neural network
        output_dataset = (output_dataset * (self.max_dataset_values[5:8] - self.min_dataset_values[5:8]) +
                          self.min_dataset_values[5:8])
        # We return the output dataset
        return output_dataset

    # This function loads the network state dictionary from the path specified
    def load_network_state_dictionary(self, network_state_dict_path):
        # Loads the pretrained dictionary
        self.load_state_dict(torch.load(network_state_dict_path))

    # This function loads the minimum and maximum values of the datasets. Helps normalize data during application
    def load_dynamics_dataset_parameters(self, dynamics_dataset_params_path):
        # The minimum and maximum dataset values
        [self.min_dataset_values, self.max_dataset_values] = pickle.load(open(dynamics_dataset_params_path, "rb"))

    # This function automatically handles training of the neural network using normalized data and labels
    def norm_train_network(self, train_data_loader, epochs):
        # Converting the network back into training mode
        self.train()
        # loop over the dataset multiple times till we cover the maximum number of epochs
        for epoch in range(epochs):
            # The running loss is initialized to zero
            running_loss = 0.0
            # Enumerating through the training dataset from the beginning
            for i, train_data in enumerate(train_data_loader, start=0):
                # get the inputs and corresponding labels
                inputs, labels = train_data
                # wrap them in a variable. It is same as tensors but helps make gradient calculation faster
                inputs, labels = Variable(inputs).to(self.device), Variable(labels).to(self.device)
                # zero the parameter gradients
                self.optimizer.zero_grad()
                # obtain the outputs of the neural network
                outputs = self(inputs)
                # Calculate the loss between outputs and labels
                loss = self.criterion(outputs, labels)
                # Perform backward propagation to calculate the gradients at every layer
                loss.backward()
                # Optimizer changes the weights at every weight according to stochastic gradient descent
                self.optimizer.step()
                # We update the running loss
                running_loss += loss.data
                # print every 500 mini-batches
                if i % 500 == 499:
                    print('    Epochs Complete: %2d, Batches Complete: %5d, Average Batch Loss: %.8f'
                          % (epoch, i + 1, running_loss / 500))
                    # Resetting the running loss
                    running_loss = 0.0
            # We drop the learning rate every epoch
            self.adjust_learning_rate(optimizer_learning_drop_percentage)

    # This is a function to evaluate the neural network using normalized data and labels
    def norm_evaluate_network(self, test_data_loader):
        # The running loss is initialized to zero
        running_loss = 0.0
        # Making the network to be in evaluation mode so that losses won't be accumulated, thereby saving memory
        self.eval()
        # Iterating through the data is data_loader
        for test_data in test_data_loader:
            # Splitting into images and labels
            inputs, labels = test_data
            # Convert to nn variable. cuda() creates another Variable that isn’t a leaf node in the computation graph
            inputs, labels = Variable(inputs).to(self.device), Variable(labels).to(self.device)
            # The output obtained from the neural network
            outputs = self(inputs)
            # Calculate the loss between outputs and labels
            loss = self.criterion(outputs, labels)
            # We update the running loss
            running_loss += loss.data
        # Prints the average loss over the normalized testing dataset
        print('    Average Loss Over the Training Dataset is: %.8f' % (running_loss/len(test_data_loader)))

# --------------- DATASET CLASS FOR HANDLING THE DYNAMICS DATA ----------------


# We create this class that inherits from dataset class to simplify access to train and test datasets
class DatasetHandler(data.Dataset):

    # Initializes the constructor for the dataset class
    def __init__(self, dataset):
        # We get the length of the dataset
        self.length = len(dataset)
        self.inputs = torch.from_numpy(dataset[:, 0:5])
        self.labels = torch.from_numpy(dataset[:, 5:8])

    # This function returns the length of the dataset
    def __len__(self):
        return self.length

    # Getter function for accessing the elements of the dataset
    def __getitem__(self, index):
        # Returns the inputs and labels of the dataset
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
    training_row_list = np.random.choice(dynamics_data_samples_count, training_samples_count, replace=False)
    # We obtain the list of rows we will treat as test samples
    test_row_list = list(set(range(dynamics_data.shape[0])) - set(training_row_list))

    # ----- Normalizing the dataset -----
    # We obtain the minimum value of each column
    min_data = np.min(dynamics_data, axis=0)
    # We obtain the maximum value of each column
    max_data = np.max(dynamics_data, axis=0)
    # We normalize the data to between -0.5 and 0.5
    dynamics_data = (dynamics_data - min_data)/(max_data - min_data)
    # We dump the min and max data into a pickle file for use post training
    with open(dir_path + "\\trained_network\dynamics_data_parameters.pkl", 'wb') as pickle_file:
        pickle.dump([min_data, max_data], pickle_file)

    # ----- Obtaining the training and testing datasets along with their loaders-----
    # extracting the training dataset
    training_dataset = DatasetHandler(dynamics_data[training_row_list, :])
    # extracting the training dataset
    testing_dataset = DatasetHandler(dynamics_data[test_row_list, :])
    # Combines a dataset and sampler to provide single or multi-process iterators over the dataset (2 sub-processes)
    train_loader = torch.utils.data.DataLoader(training_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    # Combines a dataset and sampler to provide single or multi-process iterators over the dataset (2 sub-processes)
    test_loader = torch.utils.data.DataLoader(testing_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # ----- Building the model -----
    print('Building model...')
    # Build the neural network model
    net = DynamicsModelDNN().cuda()
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
        net.load_network_state_dictionary(dir_path + "//trained_network//" + trained_network_state_dict_file)

    # ----- Training the model -----

    print('Start training the network...')
    # We train the neural network for the number of epochs specified
    net.norm_train_network(train_loader, training_epochs)

    # ----- Evaluating the model -----

    print('Start evaluating the network...')
    # We evaluate the neural network for the number of epochs specified
    net.norm_evaluate_network(test_loader)

    print('Saving the network state dictionary...')
    # Save the neural network model to the specified path
    torch.save(net.state_dict(), dir_path + '\\trained_network\\trained_network_dict.pth')
    print('Execution Completed')

# --------------- END OF CODE ----------------
