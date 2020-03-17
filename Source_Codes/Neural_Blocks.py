# Created by Aashish Adhikari at 5:23 PM 3/11/2020
#The modules here are inspired from the paper "Learning and Querying Fast Generative Models for Reinforcement Learning"

import torch.nn as nn, time, torch
import numpy as np

device = torch.cuda.current_device()


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

class DepthToSpace(nn.Module):

    def __init__(self, block_size):
        super(DepthToSpace,self).__init__()
        self.blocksize = block_size

    def forward(self, input): # input should be 4-dimensional of the formal (None, Channel-Depth, Height,Width)
        no_dimen, channel_depth, height, width = input.shape
        input = input.view(no_dimen, self.blocksize, self.blocksize, channel_depth // (self.blocksize ** 2), height, width)
        input = input.permute(no_dimen, channel_depth//(self.blocksize^2), height, self.blocksize, width, self.blocksize).contiguous()
        input = input.view(no_dimen, channel_depth // (self.blocksize ** 2), height * self.blocksize, width * self.blocksize)
        return input

class SpaceToDepth(nn.Module): #space_to_depth is a convolutional practice used very often for lossless spatial dimensionality reduction

    def __init__(self, block_size):
        super(SpaceToDepth,self).__init__()
        self.block_size = block_size

    def forward(self, x):
        n, c, h, w = x.shape[0],x.shape[1],x.shape[2],x.shape[3]
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
        assert input.shape[0] == 1
        assert input.shape[1] == 1
        assert input.shape[2] == 150

        input = self.space_to_depth_1(input) # 1,16,37,37
        input = self.conv_stack_1(input)
        input = self.space_to_depth_2(input) #[1, 256, 16, 16]
        input = self.conv_stack_2(input)
        input = self.leaky_relu(input)
        return input #1, 64, 12, 12

class Residual_Conv_Stack(nn.Module):

    def __init__(self, in_channelss = 1, k1=3, c1=32, k2=5, c2=32, k3=3, c3=64, s1=1, s2=1, s3=1, p1=1, p2=2, p3=1,final_op=118): #paddings to preserve the original H,W
        super(Residual_Conv_Stack, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channelss, out_channels= c1, kernel_size=k1, stride= s1, padding= p1) #p1 = 1
        self.leakyrelu_1 = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(in_channels=c1, out_channels= c2, kernel_size=k2, stride= s2, padding= p2) #p2 = 2
        self.leakyrelu_2 = nn.LeakyReLU()
        self.conv3 = nn.Conv2d(in_channels=c2, out_channels= c3, kernel_size=k3, stride= s3, padding= p3)#p3 =  diff 1, 3
        #Extra convolution added to make uniform dimension between input and input_ in forward()
        self.conv4 = nn.Conv2d(in_channels=c3, out_channels= final_op, kernel_size=3, stride=1, padding=1)
    def forward(self, input): #input shape: 1, X, 10, 10
        input_ = self.conv1(input)
        input_ = self.leakyrelu_1(input_)
        input_ = self.conv2(input_)
        input_ = self.leakyrelu_2(input_)
        input_ = self.conv3(input_)
        # To make uniform such that they can be added, pass input_ thru a convoulution
        input_ = self.conv4(input_)
        input_ = input + input_

        return input_


class Initial_State_Module(nn.Module):
    def __init__(self): #Each embedded observation is #1, 64, 12, 12 dimensions
        super(Initial_State_Module, self).__init__()
        self.conv_stack = Conv_Stack(in_1=192,k1=1,c1=64,in_2=64,k2=3,c2=64,k3=3,c3=64,p1=0,p2=1,p3=0) #64 to 192 hunuparxa ki/?????????

    def forward(self, obs_t_minus_2, obs_t_minus_1, obs_t_0):#Each embedding has a shape [1, 64, 12, 12] at first.
        input = torch.cat((obs_t_minus_2,obs_t_minus_1),dim = 1)#Concatenating horizontally gives a shape [1, 64, 12, 24]
        input = torch.cat((input,obs_t_0),dim = 1)#Again concatenating gives [1, 128, 12, 12]
        input = self.conv_stack(input)
        return input #returns the first state, s0 with dimensions 1, 192, 10, 10, wrong not 192 but 64


class Pool_and_Inject(nn.Module):

    def __init__(self):
        super(Pool_and_Inject, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=86, out_channels= 32, kernel_size=3, stride= 1, padding= 0)
        self.maxpool = nn.MaxPool2d(kernel_size=3) #kernel size not specified in the paper, so using 3
        #An added layer to make dimensions correct
        self.added_conv = nn.Conv2d(in_channels=118,out_channels=64, kernel_size=3, stride=1,padding=1)


    def forward(self, input): #input shape: [1, 86, 10, 10]
        input_ = self.conv1(input)
        input_ = self.maxpool(input_) #input_ shape [1, 32, 2, 2]
        input_to_concat = torch.repeat_interleave(input_,repeats = 5, dim = 2) #Tile operation
        input_to_concat = torch.repeat_interleave(input_to_concat, repeats=5, dim=3)  # Tile operation
        input_ = torch.cat((input,input_to_concat), dim=1)
        input_ = self.added_conv(input_)

        return  input_ #returns [1, 64, 10, 10]



class State_Transition_Module(nn.Module):

    def __init__(self):
        super(State_Transition_Module, self).__init__()
        self.res_conv_1 = Residual_Conv_Stack(in_channelss= 86, k1=3, c1=32, k2=5, c2=32, k3=3, c3=64, s1=1, s2=1, s3=1, p1=1, p2=2, p3=1, final_op=86) #gives 1, 86, 10, 10
        self.leaky_relu = nn.LeakyReLU()
        self.pool_and_inject = Pool_and_Inject()
        self.res_conv_2 = Residual_Conv_Stack(in_channelss=64, k1=3, c1=32, k2=5, c2=32, k3=3, c3=64, s1=1, s2=1, s3=1, p1=1, p2=0, p3=3, final_op=64)

    def forward(self, last_state, not_reshaped_last_action):
        reshaped_last_action = not_reshaped_last_action.reshape((1,22,1,1)) #converting into 22 channels ffrom a vecroe to make it suitable for tile operation
        action_to_concat = torch.repeat_interleave(reshaped_last_action,repeats = last_state.shape[2], dim = 2) #Tile operation
        action_to_concat = torch.repeat_interleave(action_to_concat, repeats = last_state.shape[3], dim = 3)
        input = torch.cat((last_state, action_to_concat),dim=1) #state shape: 1, 64, 10, 10, action shape: 1,22,10,10

        input = self.res_conv_1(input) #input fed to resconv shape 1, 86, 10, 10
        input = self.leaky_relu(input)

        input = self.pool_and_inject(input)

        input = self.res_conv_2(input) #input fed to resconv shape 1, 118, 10, 10

        return input #returns a state of dimensions 1, 64, 10, 10

class Decoder_Module(nn.Module):
    def __init__(self):
        super(Decoder_Module, self).__init__()

        # LHS
        self.conv1 = nn.Conv2d(in_channels=64, out_channels=24, kernel_size=3)
        self.leaky_relu = nn.LeakyReLU()
        # will flatten after this to feed to a linear layer
        self.linear_1 = nn.Linear(in_features=1*24*8*8, out_features=128) #input shape is 1, 24, 8, 8
        ################For Predicted Reward#############################
        self.linear_2 = nn.Linear(in_features=128, out_features=1)
        #RHS
        self.conv_stack_1 = Conv_Stack(in_1=64,in_2=32, k1=1, c1=32, k2=5, c2=32, k3=3, c3=64, s1=1, s2=1, s3=1, p1=0,p2=2,p3=0)
        self.leaky_relu_1 = nn.LeakyReLU()
        self.conv_stack_2 = Conv_Stack(in_1=16,in_2=64, k1=3, c1=64, k2=3, c2=64, k3=1, c3=48, s1=1, s2=1, s3=1, p1=0,p2=1,p3=0)
        self.linear_added = nn.Linear(in_features=1*3*56*56, out_features=150*150)
        ################For Predicted Observation########################




    def forward(self, input): #gets a state as the input with dimensions 1, 192, 10, 10
        #LHS
        input_1 = self.conv1(input)
        input_1 = self.leaky_relu(input_1) #output shape is 1, 24, 8, 8
        input_1 = self.linear_1(input_1.flatten())
        input_1 = self.linear_2(input_1)
        #RHS
        input_2 = self.conv_stack_1(input)
        input_2 = self.leaky_relu_1(input_2)
        input_2 = torch.nn.functional.pixel_shuffle(input_2,2) #output is 1,16,16,16
        input_2 = self.conv_stack_2(input_2)
        input_2 = torch.nn.functional.pixel_shuffle(input_2,2) #output is [1, 3, 56, 56], so flattening gives 1*3*56*56
        input_2 = (self.linear_added(input_2.flatten())) #output is 150*150

        return input_1, input_2



    '''
    The .view() operation gives you a new view on the tensor without copying any data.
    This means view is cheap to call and you don’t think too much about potential performance issues.
    '''