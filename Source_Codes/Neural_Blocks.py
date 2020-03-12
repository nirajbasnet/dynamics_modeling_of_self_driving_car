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

