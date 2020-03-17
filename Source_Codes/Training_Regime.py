# Created by Aashish Adhikari at 2:31 PM 3/15/2020
import torch, numpy as np
import torch.nn as nn, time
from Neural_Blocks import State_Transition_Module, Observation_Encoder, Decoder_Module, Initial_State_Module
import torch.nn.functional as F
from PIL import Image
import os, os.path
import matplotlib.pyplot as plt, copy

possible_velocity = [0.5,0.75,1,1.25,1.5]
possible_deltas = [-0.4188, -0.36652, -0.31416, -0.2618, -0.20944, -0.157080, -0.10472, -0.05236, 0, 0.05236, 0.10472, 0.15708, 0.20944, 0.2618, 0.31416, 0.36652, 0.4188]
# one_hot_velocity = torch.nn.functional.one_hot(torch.cuda.FloatTensor(possible_velocity))
# one_hot_deltas = torch.nn.functional.one_hot(torch.cuda.FloatTensor(possible_deltas))


class Environment_Model_Architecture(nn.Module):
    def __init__(self):
        super(Environment_Model_Architecture, self).__init__()
        self.observation_encoder = Observation_Encoder().to(device)
        self.initial_state_generator = Initial_State_Module().to(device)
        self.transition_module = State_Transition_Module().to(device)
        self.obseration_and_reward_decoder = Decoder_Module().to(device)

        self.next_state = 0 #Initially, nothing.

        # Read the data set and move to cuda device
        print("Reading the files from the disk now.")
        self.all_obs, self.all_ground_truth_next_obs, self.all_actions_with_rewards = get_training_data()
        print("Completed reading the files from the disk.\n")


        self.obs_minus_2 = torch.cuda.FloatTensor(np.reshape(self.all_obs[0],(1,1,150,150)))
        self.obs_minus_1 = torch.cuda.FloatTensor(np.reshape(self.all_obs[1],(1,1,150,150)))
        self.obs_0 = torch.cuda.FloatTensor(np.reshape(self.all_obs[2],(1,1,150,150)))

        #Reshape the observations to 4 dimensions

        self.embedded_obs_1 = self.observation_encoder(self.obs_minus_2)
        self.embedded_obs_2 = self.observation_encoder(self.obs_minus_1)
        self.embedded_obs_3 = self.observation_encoder(self.obs_0)
        self.action_to_take_original = torch.cuda.FloatTensor(self.all_actions_with_rewards[2]) #Start with the third action
        self.velocity = self.action_to_take_original[0]
        self.delta = self.action_to_take_original[1]
        self.one_hot_action = action_one_hot_encoding(self.delta,self.velocity)
        self.rewards_list = []
        self.obs_prediction_list = []



    def forward(self):
        #Embed three observations
        self.embedded_obs_1 = self.observation_encoder(self.obs_minus_2)
        self.embedded_obs_2 = self.observation_encoder(self.obs_minus_1)
        self.embedded_obs_3 = self.observation_encoder(self.obs_0)
        #Generate Initial State using the embeddings
        self.current_state = self.initial_state_generator(self.embedded_obs_1,self.embedded_obs_2,self.embedded_obs_3)
        #Get the next state
        self.next_state = self.transition_module(self.current_state, self.one_hot_action)
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

def action_one_hot_encoding(delta, velocity): #17 distinct deltas and 5 distince velocities
    #[-0.4188, -0.36652, -0.31416, -0.2618, -0.20944, -0.157080, -0.10472, -0.05236, 0, 0.05236, 0.10472, 0.15708, 0.20944, 0.2618, 0.31416, 0.36652, 0.4188]
    #[0.5,0.75,1,1.25,1.5]'''

    delta_encoding = np.zeros(17)
    idx = int(delta / 0.05)
    delta_encoding[idx + 8] = 1


    vel_encoding = np.zeros(5)
    idx_vel = int(velocity / 0.25) - 2
    vel_encoding[idx_vel] = 1
    return (torch.cuda.FloatTensor(np.concatenate((delta_encoding, vel_encoding), axis=0)))






'''---------------------------------------------main()------------------------------------------------------------------'''
if __name__ == "__main__":

    torch.cuda.empty_cache()
    print("Total cuda-supporting devices count is ", torch.cuda.device_count())
    device = torch.cuda.current_device()
    print("Current cuda device is ", device)



    env_model = Environment_Model_Architecture().to(device)
    print("Environment instance created.\n")

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
        print("\nIteration: ", str(data_counter))

        #env_model.obs_minus_2, env_model.obs_minus_1, env_model.obs_0 = torch.FloatTensor(env_model.all_obs[data_counter].reshape()), torch.FloatTensor(env_model.all_obs[data_counter+1]), torch.FloatTensor(env_model.all_obs[data_counter+2])
        #env_model.action_to_take = torch.FloatTensor(np.array(env_model.all_actions_with_rewards[data_counter][0]))

        predicted_reward_of_action_taken, predicted_next_obs = env_model.forward()

        # env_model.rewards_list = env_model.rewards_list.append(copy.deepcopy(predicted_reward_of_action_taken))
        # env_model.obs_prediction_list = env_model.obs_prediction_list.append(copy.deepcopy(predicted_next_obs))

        true_reward = torch.cuda.FloatTensor([env_model.all_actions_with_rewards[data_counter][2]])
        true_observation = torch.cuda.FloatTensor([env_model.all_ground_truth_next_obs[data_counter]])

        # true_rewards_list.append(env_model.all_actions_with_rewards[data_counter][1])
        # true_predictions_list.append(env_model.all_ground_truth_next_obs[data_counter])



        #if data_counter%timesteps_before_each_update == 2: #At every step
        optimizer.zero_grad()

        #reward_loss = reward_loss(env_model.rewards_list, true_rewards_list)
        reward_pred_loss = reward_loss_func(predicted_reward_of_action_taken,true_reward)
        #obs_pred_loss = obs_prediction_loss_func(env_model.obs_prediction_list, true_predictions_list)
        true_observation = torch.flatten(true_observation)
        obs_pred_loss = obs_prediction_loss_func(predicted_next_obs, true_observation)

        total_loss = reward_pred_loss + obs_pred_loss
        print("Total loss: ", str(total_loss))
        print("Calculating the gradients.")
        total_loss.backward()
        print("Now, backpropagating.")
        optimizer.step()

        # true_rewards_list = []
        # true_predictions_list = []
        # env_model.rewards_list = []
        # env_model.obs_prediction_list = []

        del total_loss, reward_pred_loss, obs_pred_loss


        #Update necesssary variables
        data_counter += 1
        '''Update the env_model obs'''
        env_model.obs_minus_2 = torch.cuda.FloatTensor(np.reshape(env_model.all_obs[data_counter],(1,1,150,150)))
        env_model.obs_minus_1 = torch.cuda.FloatTensor(np.reshape(env_model.all_obs[data_counter+1],(1,1,150,150)))
        env_model.obs_0 = torch.cuda.FloatTensor(np.reshape(env_model.all_obs[data_counter+2],(1,1,150,150)))

        '''State updated in the reward func, need not do anything here.'''

        env_model.action_to_take_original = torch.cuda.FloatTensor(env_model.all_actions_with_rewards[data_counter+2])  # When data counter = 1, action to take from the file is 3
        env_model.velocity = env_model.action_to_take_original[0]
        env_model.delta = env_model.action_to_take_original[1]
        env_model.one_hot_action = action_one_hot_encoding(env_model.delta, env_model.velocity)

        #Save the model if necessary
        if data_counter%5 == 0:
            print("Saving model now.")
            torch.save(env_model.state_dict(), '../Saved_Models/Env_Model_' + str(data_counter) + '.pth')













