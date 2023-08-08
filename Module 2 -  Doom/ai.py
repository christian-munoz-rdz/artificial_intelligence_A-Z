# AI for Doom



# Importing the libraries
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

# Importing the packages for OpenAI and Doom
import gym
from gym.wrappers import SkipWrapper
from ppaquette_gym_doom.wrappers.action_space import ToDiscrete

# Importing the other Python files
import experience_replay, image_preprocessing

# Part 1 - Building the AI

# Making the brain
class CNN(nn.Module):

    def __init__(self, number_actions):
        super(CNN, self).__init__()
        # Convolutional layers
        self.convolution1 = nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 5)
        self.convolution2 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3)
        self.convolution3 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 2)
        # Full connection
        self.fc1 = nn.Linear(in_features = self.count_neurons((1,80,80)), out_features = 40)
        self.fc2 = nn.Linear(in_features = 40, out_features = number_actions)
    
    def count_neurons(self,image_dim):
        x = Variable(torch.rand(1, *image_dim))
        x = F.relu(F.max_pool2d(self.convolution1(x), 3, 2))
        x = F.relu(F.max_pool2d(self.convolution2(x), 3, 2))
        x = F.relu(F.max_pool2d(self.convolution3(x), 3, 2))

        return x.data.view(1, -1).size(1)

    def forward(self, x):
        # Convolutional layers
        x = F.relu(F.max_pool2d(self.convolution1(x), 3, 2))
        x = F.relu(F.max_pool2d(self.convolution2(x), 3, 2))
        x = F.relu(F.max_pool2d(self.convolution3(x), 3, 2))
        # Flattening
        x = x.view(x.size(0), -1)
        # Full connection
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

# Making the body
class SoftmaxBody(nn.Module):

    def __init__(self, T):
        super(SoftmaxBody, self).__init__()
        self.T = T
    
    def forward(self, outputs):
        probs = F.softmax(outputs * self.T)
        actions = probs.multinomial()
        return actions

# Making the AI

# Part 2 - Training the AI with Deep Convolutional Q-Learning

# Getting the Doom environment

# Building an AI

# Setting up Experience Replay
    
# Implementing Eligibility Trace

# Making the moving average on 100 steps

# Training the AI
