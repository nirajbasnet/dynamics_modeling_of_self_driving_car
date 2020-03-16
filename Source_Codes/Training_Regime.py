# Created by Aashish Adhikari at 2:31 PM 3/15/2020

import torch.nn as nn, time
from Neural_Blocks import State_Transition_Module, Observation_Encoder, Decoder_Module, Initial_State_Module
import torch.nn.functional as F

class Environment_Model_Architecture(nn.Module):
    def __init__(self):
        super(Environment_Model_Architecture, self).__init__()
        self.observation_encoder = Observation_Encoder()
        self.initial_state_generator = Initial_State_Module()
        self.transition_module = State_Transition_Module()
        self.obseration_and_reward_decoder = Decoder_Module()

        self.obs_minus_2 = None
        self.obs_minus_1 = None
        self.obs_0 = None

        self.embedded_obs_1 = None
        self.embedded_obs_2 = None
        self.embedded_obs_3 = None

        self.current_state = 0#At first there is no state information and needs to be encoded from the first three observations
        self.next_state = 0

        self.action_to_take = None

        self.reward = None
        self.obs_prediction = None
        self.reward_list = []
        self.obs_prediction_list = []

    def forward(self, input):
        #Embed three observations
        self.embedded_obs_1 = self.observation_encoder(self.obs_minus_2)
        self.embedded_obs_2 = self.observation_encoder(self.obs_minus_1)
        self.embedded_obs_3 = self.observation_encoder(self.obs_0)
        #Generate Initial State using the embeddings
        self.current_state = self.initial_state_generator(self.embedded_obs_1,self.embedded_obs_2,self.embedded_obs_3)
        '''Make sure the passed action is reshaped.'''
        self.next_state = self.transition_module(self.current_state, self.action_to_take)
        self.reward, self.obs_prediction = self.obseration_and_reward_decoder(self.next_state)

        return self.reward, self.obs_prediction





        '''Dont forget to set the current state to zero at the end of the last time step of each trajectory of length 3'''






#if __name__ == "__main__":
