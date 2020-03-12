# Created by Aashish Adhikari at 5:23 PM 3/11/2020
#The modules here are inspired from the paper "Learning and Querying Fast Generative Models for Reinforcement Learning"

import torch.nn as nn, time

class Conv_Stack(nn.Module):

    def __init__(self, k1=3,c1=1,k2=3,c2=1,k3=3,c3=1,s1=1,s2=1,s3=1,p1=0,p2=0,p3=0):

        super(Conv_Stack, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=c1, kernel_size=k1, stride=s1, padding= p1)
        self.leakyrelu_1 = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(in_channels=c1, out_channels= c2, kernel_size=k2, stride=s2, padding=p2) #Size-Expanding convolution
        self.leakyrelu_2 = nn.LeakyReLU()
        self.conv3 = nn.Conv2d(in_channels=c2, out_channels=c3, kernel_size=k3, stride=s3, padding=p3) #op dimension not matched with anything for now

    def forward(self, input):

        input = self.conv1(input)
        input = self.leakyrelu_1(input)
        input_ = self.conv2(input)
        input = self.leakyrelu_2(input+input_)
        input = self.conv3(input)

        return input

# space_to_depth operation outputs a copy of the input tensor where values from the height and width dimensions are moved to the depth dimension.
# It transfers an input size of (None, 38,38,64) to (None, 19,19, 256).
#referred from https://discuss.pytorch.org/t/is-there-any-layer-like-tensorflows-space-to-depth-function/3487/15
class DepthToSpace(nn.Module):

    def __init__(self, block_size):
        super().__init__()
        self.blocksize = block_size

    def forward(self, input): # input should be 4-dimensional of the formal (None, Channel-Depth, Height,Width)
        no_dimen, channel_depth, height, width = input.size()
        input = input.view(no_dimen, self.blocksize, self.blocksize, channel_depth // (self.blocksize ** 2), height, width)
        input = input.permute(no_dimen, channel_depth//(self.blocksize^2), height, self.blocksize, width, self.blocksize).contiguous()
        input = input.view(no_dimen, channel_depth // (self.blocksize ** 2), height * self.blocksize, width * self.blocksize)
        return input


class SpaceToDepth(nn.Module):

    def __init__(self, block_size):
        super().__init__()
        self.block_size = block_size

    def forward(self, x):
        no_dimen, channel_depth, height, width = x.size()
        x = x.view(no_dimen, channel_depth, height // self.block_size, self.block_size, width // self.block_size, self.block_size)
        x = x.permute(no_dimen, self.block_size, self.block_size, channel_depth, height//self.block_size, width//self.block_size).contiguous()
        x = x.view(no_dimen, channel_depth * (self.block_size ** 2), height // self.block_size, width // self.block_size)
        return x

class Observation_Encoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.space_to_depth_1 = SpaceToDepth(4)
        self.conv_stack_1 = Conv_Stack(k1=3, c1=16, k2=5, c2=16, k3=3, c3=64, s1=1, s2=2, s3=1, p1=0,p2=2,p3=0)
        self.space_to_depth_2 = SpaceToDepth(2)
        self.conv_stack_2 = Conv_Stack(k1=3, c1=32, k2=5, c2=32, k3=3, c3=64, s1=1, s2=2, s3=1, p1=0, p2=2, p3=0)
        self.leaky_relu = nn.LeakyReLU()


    def forward(self, input):
        input = self.space_to_depth_1(input)
        input = self.conv_stack_1(input)
        input = self.space_to_depth_2(input)
        input = self.space_to_depth_2(input)
        input = self.conv_stack_2(input)
        input = self.leaky_relu(input)

        return input
