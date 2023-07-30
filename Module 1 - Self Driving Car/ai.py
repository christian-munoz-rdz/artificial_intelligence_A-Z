# AI for Self Driving Car

# Importing the libraries

import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

#Creating the architecture of the Neural Network
class Network(nn.Module):

    #Initializing the Neural Network
    def __init__(self, input_size, nb_action):
        super(Network, self).__init__()
        self.input_size = input_size
        self.nb_action = nb_action
        self.fc1 = nn.Linear(input_size,30)
        self.fc2 = nn.Linear(30,nb_action)
    
    #Forward propagation of the neural network
    def forward(self, state):
        x = F.relu(self.fc1(state))
        q_values = self.fc2(x)
        return q_values

#Implementing Experience Replay
class ReplayMemory(object):

    #Initializing the memory
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    #Pushing a new event in the memory
    def push(self, event):
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]
    
    #Drawing a random sample from the memory
    def sample(self, batch_size):
        #if list = [(1,2,3),(4,5,6),(7,8,9)] then zip(*list) = [(1,4,7),(2,5,8),(3,6,9)]
        samples = zip(*random.sample(self.memory, batch_size))
        return map(lambda x: Variable(torch.cat(x,0)), samples)

