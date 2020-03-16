# Created by Aashish Adhikari at 2:31 PM 3/15/2020
import torch, numpy as np
import torch.nn as nn, time
from Neural_Blocks import State_Transition_Module, Observation_Encoder, Decoder_Module, Initial_State_Module
import torch.nn.functional as F
from PIL import Image
import os, os.path
import matplotlib.pyplot as plt, copy


class Environment_Model_Architecture(nn.Module):
    def __init__(self):
        super(Environment_Model_Architecture, self).__init__()
        self.observation_encoder = Observation_Encoder()
        self.initial_state_generator = Initial_State_Module()
        self.transition_module = State_Transition_Module()
        self.obseration_and_reward_decoder = Decoder_Module()

        self.next_state = 0 #Initially, nothing.

        # Read the data set
        self.all_obs, self.all_ground_truth_next_obs, self.all_actions_with_rewards = get_training_data()

        self.obs_minus_2 = torch.FloatTensor(np.reshape(self.all_obs[0],(1,1,150,150)))
        self.obs_minus_1 = torch.FloatTensor(np.reshape(self.all_obs[1],(1,1,150,150)))
        self.obs_0 = torch.FloatTensor(np.reshape(self.all_obs[2],(1,1,150,150)))

        #Reshape the observations to 4 dimensions

        self.embedded_obs_1 = self.observation_encoder(self.obs_minus_2)
        self.embedded_obs_2 = self.observation_encoder(self.obs_minus_1)
        self.embedded_obs_3 = self.observation_encoder(self.obs_0)

        self.action_to_take = torch.FloatTensor(self.all_actions_with_rewards[2]) #Start with the third action

        self.reward_list = []
        self.obs_prediction_list = []



    def forward(self):
        #Embed three observations
        self.embedded_obs_1 = self.observation_encoder(self.obs_minus_2)
        self.embedded_obs_2 = self.observation_encoder(self.obs_minus_1)
        self.embedded_obs_3 = self.observation_encoder(self.obs_0)
        #Generate Initial State using the embeddings
        self.current_state = self.initial_state_generator(self.embedded_obs_1,self.embedded_obs_2,self.embedded_obs_3)
        '''Make sure the passed action is reshaped.'''
        #Get the next state
        self.next_state = self.transition_module(self.current_state, self.action_to_take)
        #decode the reward and the next observation for the action taken at current state
        self.reward, self.obs_prediction = self.obseration_and_reward_decoder(self.next_state)

        return self.reward, self.obs_prediction

      #'''Dont forget to set the current state to zero at the end of the last time step of each trajectory of length 3'''

def get_training_data():

    #Read Input Images
    all_training_images = []
    path = "../Data/Input/"
    valid_images_format = [".jpg", ".gif", ".png", ".tga"]
    counter = 0
    for f in os.listdir(path):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_images_format:
            continue
        all_training_images.append(plt.imread(path+"img"+str(counter)+".png"))
        counter +=1


    #Read Ground Truth Images
    all_ground_truth_images = []
    path = "../Data/Ground_Truth/"
    valid_images_format = [".jpg", ".gif", ".png", ".tga"]
    counter = 0
    for f in os.listdir(path):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_images_format:
            continue
        all_ground_truth_images.append(plt.imread(path+"img"+str(counter)+".png"))

    #Read the actions and the corresponding rewards
    velocity_delta_reward = np.genfromtxt("../Data/Actions/train_data.csv", delimiter=",")
    velocity_delta_reward = copy.deepcopy(velocity_delta_reward[1:])  # Removing the column names i.e., the first row from the csv file data

    return all_training_images, all_ground_truth_images, velocity_delta_reward



'''---------------------------------------------main()------------------------------------------------------------------'''
if __name__ == "__main__":


    print("Total cuda-supporting devices count is ", torch.cuda.device_count())
    device = torch.cuda.current_device()
    print("Current cuda device is ", device)

    env_model = Environment_Model_Architecture().to(device)

    timesteps_before_each_update = 1

    seed = 42
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    data_counter = 0



    true_rewards_list = []
    true_predictions_list = []

    #Set all none values from the env_model


    reward_loss_func = nn.MSELoss()
    obs_prediction_loss_func = nn.KLDivLoss()

    optimizer = torch.optim.RMSprop(env_model.parameters(), lr=0.005)


    print("Training starts. t = ",timesteps_before_each_update,"\n")
    while data_counter <= 19650:

        env_model.obs_minus_2, env_model.obs_minus_1, env_model.obs_0 = torch.FloatTensor(env_model.all_obs[data_counter]), torch.FloatTensor(env_model.all_obs[data_counter+1]), torch.FloatTensor(env_model.all_obs[data_counter+2])
        env_model.action_to_take = torch.FloatTensor(np.array(env_model.all_actions_with_rewards[data_counter][0]))

        predicted_reward_of_action_taken, predicted_next_obs = env_model.forward()

        env_model.rewards_list = env_model.rewards_list.append(copy.deepcopy(predicted_reward_of_action_taken))
        env_model.obs_prediction_list = env_model.obs_prediction_list.append(copy.deepcopy(predicted_next_obs))

        true_rewards_list.append(env_model.all_actions_with_rewards[data_counter][1])
        true_predictions_list.append(env_model.all_ground_truth_next_obs[data_counter])



        if data_counter%timesteps_before_each_update == 2: #At every third step
            optimizer.zero_grad()

            reward_loss = reward_loss(env_model.rewards_list, true_rewards_list)
            obs_pred_loss = obs_prediction_loss_func(env_model.obs_prediction_list, true_predictions_list)
            total_loss = reward_loss + obs_pred_loss

            total_loss.backward()

            optimizer.step()

            true_rewards_list = []
            true_predictions_list = []
            env_model.rewards_list = []
            env_model.obs_prediction_list = []



        #Update necesssary variables
        data_counter += 1
        '''Update all nones from the env_model'''
        env_model.obs_minus_2 = np.reshape(env_model.all_obs[data_counter],(1,1,150,150))
        env_model.obs_minus_1 = np.reshape(env_model.all_obs[data_counter+1],(1,1,150,150))
        env_model.obs_0 = np.reshape(env_model.all_obs[data_counter+2],(1,1,150,150))

        env_model.embedded_obs_1 = env_model.observation_encoder(env_model.obs_minus_2)
        env_model.embedded_obs_2 = env_model.observation_encoder(env_model.obs_minus_1)
        env_model.embedded_obs_3 = env_model.observation_encoder(env_model.obs_0)

        env_model.action_to_take = env_model.all_actions_with_rewards[data_counter+2]  # When data counter = 1, action to take from the file is 3

        #Save the model if necessary
        if data_counter%9 == 0:
            torch.save(env_model.state_dict(), '../Saved_Models/Env_Model_' + str(data_counter) + '.pth')













