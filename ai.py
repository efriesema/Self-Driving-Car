###########################################################################
#
#    Ed Friesema     December 17, 2019
#    Udemy AI course -Self-Driving car Module
#    Objective:  to create a reinforcement learning neural network, Dqn
#                that will learn how to properly navigate around a map
#                where sand is drawn on the map by the user and  two goals
#                are set first in the upper left cornee and then in the
#                lower right. Base don PyTorch
#
###########################################################################

import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable
from datetime import  datetime

# Create neural net architecture
class Network(nn.Module):
    HIDDEN = 30  # number of nodes in the hidden layer

    def __init__(self, input_size, nb_action):
        super(Network, self).__init__()
        self.input_size = input_size
        self.nb_action = nb_action
        self.fc1 = nn.Linear(input_size, Network.HIDDEN)  # hidden layer is fully connected to input layer
        self.fc2 = nn.Linear(Network.HIDDEN, nb_action)  # hidden layer-output layer

    def forward(self, state):  # Overrides base class
        x = F.relu(self.fc1(state))  # hidden neurons layer with relu activation function
        q_values = self.fc2(x)  # output neurons
        return q_values


# Implement Experience Replay
class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity  # Maximum number of transitions stored in memory
        self.memory = []  # actually list of memory values initialized to null

    def push(self, event):  # event =(last_state, new state, last action ,last reward)
        self.memory.append(event)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    def sample(self, batch_size):
        # zip(*) reformats the memory from a list of tuples of state ,action and reward samples
        # to 4 separate lists of all the state samples, reward samples,action sample
        samples = zip(*random.sample(self.memory, batch_size))
        return map(lambda x: Variable(torch.cat(x, 0)),
                   samples)  # concatenate samples into a tensor based on last state


# Implement Deep Q Learning

class Dqn():
    MEMORY_SIZE = 100000  # Memory size
    LEARNING_RATE = 0.001
    BATCH_SIZE = 100
    WINDOW_SIZE = 1000
    T = 100   # multiplier of q-probabilities (T>=0) T=0 in effect turn off the AI
            # higher T means it's more likely that largest Q value will be selected

    def __init__(self, input_size, nb_action, gamma):
        self.gamma = gamma
        self.reward_window = []
        self.model = Network(input_size, nb_action)
        self.memory = ReplayMemory(Dqn.MEMORY_SIZE)
        self.optimizer = optim.Adam(self.model.parameters(), lr=Dqn.LEARNING_RATE)
        # format as a tensor using state x batch_size
        self.last_state = torch.Tensor(input_size).unsqueeze(0)  # 0 because we are unsqueezing states
        self.last_action = 0  # action represented three possibilities 0,1,2
        self.last_reward = 0  # reward is a numeric value from -1.0 to 1.0

    def select_action(self, state):  # state = (signal1, signal2, signal3, orientation, -orientation)
        # Select your choice of one of three actions based on  Q probability values using softmax function
        # volatile=True keeps the gradient part of tensor from being included
        # when not needed and thus saves memory and improves performance
        probs = F.softmax(self.model(Variable(state, volatile=True)) * Dqn.T)
        action = probs.multinomial()  # possible need to add (1) if 'no modulenamed Torch' error received
        return action.data[0, 0]

    def learn(self, batch_state, batch_next_state, batch_action, batch_reward):
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)  # 1 because we are unsqueezing actions
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        target = batch_reward + self.gamma * next_outputs
        # Hoover loss function
        td_loss = F.smooth_l1_loss(outputs, target)  # used to backprorpagate andmodify network weights
        self.optimizer.zero_grad()  # reinitializes optimizer
        td_loss.backward(retain_graph=True)  # might need to change to retain_graph if 'no module named Torch' erro
        self.optimizer.step()

    def update(self, reward, new_signal):
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        self.memory.push(
            (self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward])))
        action = self.select_action(new_state)
        if len(self.memory.memory) > Dqn.BATCH_SIZE:
            batch_state, batch_next_state, batch_reward, batch_action = self.memory.sample(Dqn.BATCH_SIZE)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.reward_window.append(reward)
        if len(self.reward_window) > Dqn.WINDOW_SIZE:
            del self.reward_window[0]
        return action

    def score(self):
        return sum(self.reward_window) / (len(self.reward_window) + 1.)

    def save(self):
        torch.save({'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict()
                    }, 'last_brain.pth')
        print("brain saved....for now")

    def load(self):
        if os.path.isfile('last_brain.pth'):
            print("=> loading checkpoint . . .")
            checkpoint = torch.load('last_brain.pth')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict((checkpoint['optimizer']))
            print("Done !")
        else:
            print("last_brain.pth does not exist")
