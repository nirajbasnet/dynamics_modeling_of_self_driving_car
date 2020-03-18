# Created by Ashwin Vinoo
# Date 2/27/2020

# This code simply demos how to use the trained neural network to obtain predictions of movement
# Its Very Simple - Just 4 steps!!!

# ----- Step 1: Import the dynamics neural network model class -----

# We import the dynamics model class from the other python file as well as other libraries
from dynamics_network_trainer import DynamicsModelDNN
import os
import numpy as np

# ----- Step 2: Create a CUDA based object of that class -----

# We create a CUDA (GPU) instance of the neural network model class but without .cuda a cpu instance is created
net = DynamicsModelDNN().cuda()

# ----- Step 3: Load the model parameters from the 2 files in trained_network folder -----

# The name of the file from which we can load the state dictionary
trained_network_state_dict_file = 'trained_network_dict.pth'
# The name of the file from which we can load the dynamics data parameters
dynamics_data_parameters_file = 'dynamics_data_parameters.pkl'
# Gets the directory of this python file
dir_path = os.path.dirname(os.path.realpath(__file__))
# We load both the state dictionary file and the dynamics data parameters file
net.load_network_state_dictionary(dir_path + "//trained_network//" + trained_network_state_dict_file)
net.load_dynamics_dataset_parameters(dir_path + "//trained_network//" + dynamics_data_parameters_file)

# ----- Step 4: Run the function to get predictions -----

# Example One: 1D numpy input
input_1 = np.array([-1.45504177, 1.259085278, -1.505907286, -2.229707808, 0.227067457], dtype=np.float32)
print("input one is:\n", input_1)
print("output one is:\n", net.get_dynamics_predictions(input_1))
print("ideal output one is:\n", [-1.462270874, 1.370336041, -1.585165564], "\n")

# Example One: 2D numpy input
input_2 = np.array([[-0.718528424, 0.250674387, -0.783228812, -2.229707808, 0.315297908],
                    [0.431422099, -0.081821809, -0.49981321, 2.217140725, -0.351144536],
                    [0, 0, 0, 0.439988574, 0.193475833]], dtype=np.float32)
print("input two is:\n", input_2)
print("output two is:\n", net.get_dynamics_predictions(input_2))
print("ideal output two is:\n", [-0.797531328, 0.329335263, -0.895118643], "\n",
                                [0.528718227, -0.13495133, -0.624766513], "\n",
                                [0.021999429, 0, 0.013262383])

# ----- Hope this clarifies stuff :p -----
