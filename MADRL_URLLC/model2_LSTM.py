import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import pdb


# Actor and Critic Networks
class SPN(nn.Module):
    def __init__(self, input_size, hidden_size, fc_output_size):
        super(SPN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.fc_output_size = fc_output_size
        
        self.lstm = nn.LSTM(input_size, self.hidden_size, num_layers=1, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, self.fc_output_size)
        
        
        
        
    def forward(self, x):
        out, _ = self.lstm(x)
        x = self.fc(out)
        return x

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        input_size, lstm_hidden_size, fc_output_size = state_dim, 200, 240
        self.SPN = SPN(input_size, lstm_hidden_size, fc_output_size)
        self.layer1 = nn.Linear(fc_output_size, 512)
        self.layer2 = nn.Linear(512, 256)
        self.layer3 = nn.Linear(256, 128)
        self.layer4 = nn.Linear(128, action_dim)
        
    def forward(self, state):
        #pdb.set_trace()
        x = self.SPN(state)
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        return torch.tanh(self.layer4(x))

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        input_size, lstm_hidden_size, fc_output_size = state_dim, 200, 240
        self.SPN = SPN(input_size, lstm_hidden_size, fc_output_size)
        
        self.layer1 = nn.Linear(fc_output_size + action_dim, 512)
        self.layer2 = nn.Linear(512, 256)
        self.layer3 = nn.Linear(256, 128)
        self.layer4 = nn.Linear(128, 1)
        
    def forward(self, state, action):
        #pdb.set_trace()
        x = self.SPN(state)
        #pdb.set_trace()
        x = torch.cat([x, action], 1)
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.relu(self.layer3(x))
        return self.layer4(x)
    
   