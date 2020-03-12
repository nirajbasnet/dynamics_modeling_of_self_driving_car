# Created by Aashish Adhikari at 5:23 PM 3/11/2020
#The modules here are inspired from the paper "Learning and Querying Fast Generative Models for Reinforcement Learning"

import torch.nn as nn, time

class Conv_Stack(nn.Module):

    def __init__(self):

        super(Conv_Stack, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding= 0)
        self.leakyrelu_1 = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(in_channels=16, out_channels= 16, kernel_size=5, stride=1, padding=2) #Size-Expanding convolution
        self.leakyrelu_2 = nn.LeakyReLU()
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=64, kernel_size=5, stride=1, padding=0) #op dimension not matched with anything for now

    def forward(self, input): #input dimension 150 X 150 X 1

        input = self.conv1(input)
        input = self.leakyrelu_1(input)
        input_ = self.conv2(input)
        input = self.leakyrelu_2(input+input_)
        input = self.conv3(input)

        return input #Output dimension is 144 X 144 X 64


