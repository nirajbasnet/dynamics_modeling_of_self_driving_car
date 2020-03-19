# Created by Ashwin Vinoo
# Date 2/27/2020

# This code simply demos how to use the trained neural network to obtain predictions of movement
# Its Very Simple - Just 4 steps!!!

# ----- Step 1: Import the dynamics neural network model class -----

# We import the dynamics model class from the other python file as well as other libraries
from dynamics_network_rnn import DynamicsModelRNN
import os
import numpy as np

# ----- Step 2: Create a CUDA based object of that class -----

# We create a CUDA (GPU) instance of the neural network model class but without .cuda a cpu instance is created
net = DynamicsModelRNN().cuda()

# ----- Step 3: Load the model parameters from the 2 files in trained_network folder -----

# The name of the file from which we can load the state dictionary
trained_network_state_dict_file = 'trained_network_dict_rnn.pth'
# The name of the file from which we can load the dynamics data parameters
dynamics_data_parameters_file = 'dynamics_data_parameters_rnn.pkl'
# Gets the directory of this python file
dir_path = os.path.dirname(os.path.realpath(__file__))
# We load both the state dictionary file and the dynamics data parameters file
net.load_network_state_dictionary(dir_path + "//trained_network//" + trained_network_state_dict_file)
net.load_dynamics_dataset_parameters(dir_path + "//trained_network//" + dynamics_data_parameters_file)

# ----- Step 4: Run the function to get predictions -----

# Example One: 2D numpy input (12 sequences of 62  actions + observations are needed for prediction)
input_1 = np.random.rand(12, 62)
print("output is:\n", net.get_dynamics_predictions(input_1))

# ----- Hope this clarifies stuff :p -----
