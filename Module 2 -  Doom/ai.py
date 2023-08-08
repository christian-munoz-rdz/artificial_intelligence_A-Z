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
class CNN(nn.Module):

    def __init__(self, number_actions):
        super(CNN, self).__init__()
        # Convolutional layers
        self.convolution1 = nn.Conv2d(in_channels = 1, out_channels = 32, kernel_size = 5)
        self.convolution2 = nn.Conv2d(in_channels = 32, out_channels = 32, kernel_size = 3)
        self.convolution3 = nn.Conv2d(in_channels = 32, out_channels = 64, kernel_size = 2)
        # Full connection
        self.fc1 = nn.Linear(in_features = number_neurons, out_features = 40)
        self.fc2 = nn.Linear(in_features = 40, out_features = number_actions)

# Making the brain

# Making the body

# Making the AI

# Part 2 - Training the AI with Deep Convolutional Q-Learning

# Getting the Doom environment

# Building an AI

# Setting up Experience Replay
    
# Implementing Eligibility Trace

# Making the moving average on 100 steps

# Training the AI
