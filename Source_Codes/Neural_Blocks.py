# Created by Aashish Adhikari at 5:23 PM 3/11/2020
#The modules here are inspired from the paper "Learning and Querying Fast Generative Models for Reinforcement Learning"

import torch.nn as nn, time, torch
import numpy as np


class Conv_Stack(nn.Module):

    def __init__(self, in_1=16,k1=3,c1=1, in_2=16, k2=3,c2=1,k3=3,c3=1,s1=1,s2=1,s3=1,p1=0,p2=0,p3=0):
        super(Conv_Stack, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_1, out_channels=c1, kernel_size=k1, stride=s1, padding= p1)
        self.leakyrelu_1 = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(in_channels=in_2, out_channels= c2, kernel_size=k2, stride=s2, padding=p2) #Size-Expanding convolution
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

''' torch.nn.functional.pixel_shuffle does exactly what tf.nn.depth_to_space does,
PyTorch doesn't have any function to do the inverse operation similar to tf.nn.space_to_depth'''
# class DepthToSpace(nn.Module):
#
#     def __init__(self, block_size):
#         super(DepthToSpace,self).__init__()
#         self.blocksize = block_size
#
#     def forward(self, input): # input should be 4-dimensional of the formal (None, Channel-Depth, Height,Width)
#         no_dimen, channel_depth, height, width = input.shape
#         input = input.view(no_dimen, self.blocksize, self.blocksize, channel_depth // (self.blocksize ** 2), height, width)
#         input = input.permute(no_dimen, channel_depth//(self.blocksize^2), height, self.blocksize, width, self.blocksize).contiguous()
#         input = input.view(no_dimen, channel_depth // (self.blocksize ** 2), height * self.blocksize, width * self.blocksize)
#         return input
#


class SpaceToDepth(nn.Module): #space_to_depth is a convolutional practice used very often for lossless spatial dimensionality reduction

    def __init__(self, block_size):
        super(SpaceToDepth,self).__init__()
        self.block_size = block_size

    def forward(self, x):
        n, c, h, w = x.size()
        unfolded_x = torch.nn.functional.unfold(x, self.block_size, stride=self.block_size)
        N_C_H_W = unfolded_x.view(n, c * self.block_size ** 2, h // self.block_size, w // self.block_size)
        return N_C_H_W # N , C, H, W --> 1 16 37 37

class Observation_Encoder(nn.Module):

    def __init__(self):
        super(Observation_Encoder, self).__init__()
        self.space_to_depth_1 = SpaceToDepth(4)
        self.conv_stack_1 = Conv_Stack(in_1=16,in_2=16, k1=3, c1=16, k2=5, c2=16, k3=3, c3=64, s1=1, s2=1, s3=1, p1=0,p2=2,p3=0)
        self.space_to_depth_2 = SpaceToDepth(2)
        self.conv_stack_2 = Conv_Stack(in_1 = 256, in_2=32 ,k1=3, c1=32, k2=5, c2=32, k3=3, c3=64, s1=1, s2=1, s3=1, p1=0, p2=2, p3=0)
        self.leaky_relu = nn.LeakyReLU()


    def forward(self, input): #1,1,150,150
        input = self.space_to_depth_1(input) # 1,16,37,37
        input = self.conv_stack_1(input)
        input = self.space_to_depth_2(input) #[1, 256, 16, 16]
        input = self.conv_stack_2(input)
        input = self.leaky_relu(input)

        return input #1, 64, 12, 12

class Residual_Conv_Stack(nn.Module):

    def __init__(self, in_dimen = 1, k1=3, c1=32, k2=5, c2=32, k3=3, c3=64, s1=1, s2=1, s3=1, p1=0, p2=0, p3=0):
        super(Residual_Conv_Stack, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_dimen, out_channels= c1, kernel_size=k1, stride= s1, padding= p1)
        self.conv2 = nn.Conv2d(in_channels=c1, out_channels= c2, kernel_size=k2, stride= s2, padding= p2)
        self.conv3 = nn.Conv2d(in_channels=c2, out_channels= c3, kernel_size=k3, stride= s3, padding= p3)

    def forward(self, input):
        input_ = self.conv1(input)
        input_ = nn.LeakyReLU(input_)
        input_ = self.conv2(input_)
        input_ = nn.LeakyReLU(input_)
        input_ = self.conv3(input_)
        input_ = input + input_

        return input_


# class Pool_and_Inject(nn.Module):
#
#     def __init__(self):
#         super(Pool_and_Inject, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels=in_dimen, out_channels= c1, kernel_size=k1, stride= s1, padding= p1)
#         self.maxpool = nn.MaxPool2d(kernel_size=5) #kernel size not specified in the paper, so using 5
#
#     def forward(self, input):
#         input_ = self.conv1(input)
#         input_ = self.maxpool(input_)
#
#         input_ = np.tile(input_,())
#
#         #Check if the tiling worked or not
#         assert input.shape[0] = input_.shape[0]
#

if __name__ == "__main__":

    sample_input = np.random.randn(1,1,150,150)
    a= torch.FloatTensor(sample_input)
    encoder = Observation_Encoder()
    o = encoder.forward(a)
    print(o.shape)


    '''
    The .view() operation gives you a new view on the tensor without copying any data.
This means view is cheap to call and you don’t think too much about potential performance issues.
    '''