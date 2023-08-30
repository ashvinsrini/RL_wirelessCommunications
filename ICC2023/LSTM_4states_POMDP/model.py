# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 10:45:31 2023

@author: sriniva3
"""

import math
import random

import gym
import numpy as np
import pdb
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
# Import math Library
import math
from system_model import *

N, Lc, Lu, lamb_bh, lamb_ue = 40, 6, 3, 0.05, 0.05
# '''


class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size, Lc, std=0.0, lstm_hidden_size=4):
        super(ActorCritic, self).__init__()

        self.lstm_hidden_size = lstm_hidden_size
        self.projection = nn.Sequential(
            nn.Linear(num_inputs, 128),
            nn.Tanh(),
        )

        self.lstm_layer = nn.LSTM(
            input_size=4, hidden_size=self.lstm_hidden_size, num_layers=1)
        # 2 for UE plus fp, 4 for 4 LSTM cells, 40 for number of files.
        self.fc_1 = nn.Linear(2*4*40, 128)
        self.relu = nn.ReLU()

        self.critic = nn.Sequential(
            nn.Linear(128, hidden_size),
            nn.Linear(hidden_size, 1),
            nn.ReLU(),


        )

        self.actor = nn.Sequential(
            nn.Linear(128, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, num_outputs),
            nn.Softplus(),
        )

    def forward(self, x):
        #################### LSTM ######################
        #FC = self.projection(x)
        FC = x
        # pdb.set_trace()
        FC = torch.reshape(FC, (80, 1, 4))
        # pdb.set_trace()
        lstm_out, (hn, cn) = self.lstm_layer(FC)
        lstm_out = torch.reshape(lstm_out.flatten(), (1, 2*4*40))
        # pdb.set_trace()
        # reshaping the data for Dense layer next
        hn = hn.view(-1, self.lstm_hidden_size)
        out = self.relu(lstm_out)
        out = self.fc_1(out)  # first Dense
        # pdb.set_trace()

        ######################## Actor and Critic networks ##########################

        value = self.critic(out)
        probs = self.actor(out)
        # pdb.set_trace()
        probs = torch.distributions.Dirichlet(probs)
        probs_sample = probs.sample()
        # pdb.set_trace()
        log_prob = probs.log_prob(probs_sample)
        entropy = probs.entropy()

        t1, t2, t3 = Lc * probs_sample[0][0:N]/sum(probs_sample[0][0:N]), probs_sample[0][N:2*N]/sum(probs_sample[0][N:2*N]),\
            Lu * probs_sample[0][2*N:3*N]/sum(probs_sample[0][2*N:3*N])
        # pdb.set_trace()
        probs_sample = torch.cat((t1, t2, t3), 0)
        # pdb.set_trace()

        #rho = 1 * probs_sample[0][0:N]/sum(probs_sample[0][0:N])
        #w = probs_sample[0][N:]/sum(probs_sample[0][N:])
        #probs_sample= torch.cat((rho, w), 0)
        #dist  = Categorical(self.actor(x))
        # pdb.set_trace()
        #output = dist.probs
        # pdb.set_trace()
        return probs_sample, value, log_prob, entropy
# '''


def compute_returns(next_value, rewards, masks, gamma=0.98):
    R = next_value
    returns = []
    for i in reversed(range(len(rewards))):
        R = rewards[i] + gamma * R * masks[i]
        returns.insert(0, R)
        # pdb.set_trace()
    return returns



def test_env1(model, device, state, fp_state, ue_cache_prob, rho, w, prev_rho, num_steps, prev_ue, fp_table):
    state = state.flatten()
    state = torch.FloatTensor(state).to(device)
    state = np.reshape(state, [1, len(state)])
    total_reward, tot_rqos, tot_ruep, tot_rbh = 0, 0, 0, 0
    for time_step in range(num_steps):

        probs, _, _, _ = model(state)
        action_vect = probs.detach().numpy()
        rho, w, tgt_prob = action_vect[0:N], action_vect[N:2 *
                                                         N], action_vect[2*N:]
        # pdb.set_trace()
        # print('----------------------------------------')
        # print(np.sum(tgt_prob))
        print(time_step)
        next_state, r, next_fp_state, next_ue_cache_prob, rqos, rbh, ruep = env1(fp_state, ue_cache_prob, rho, w, prev_ue,
                                                                                 prev_rho, N, Lc, Lu, lamb_bh, Q, fp_table, tgt_prob)

        ue_cache_prob_cpy = ue_cache_prob.copy()
        state, fp_state, ue_cache_prob, prev_rho, prev_ue = next_state, next_fp_state, next_ue_cache_prob, rho, ue_cache_prob_cpy
        # pdb.set_trace()
        state = state.flatten()
        state = torch.FloatTensor(state).to(device)
        state = np.reshape(state, [1, len(state)])

        total_reward += r
        tot_rqos += rqos
        tot_ruep += rbh
        tot_rbh += ruep

    # total_reward+=N
    #print(total_reward, tot_rqos, tot_rbh, tot_ruep)
    return total_reward, tot_rqos, tot_rbh, tot_ruep


def plot(frame_idx, rewards):
    plt.plot(rewards, 'b-')
    plt.title('frame %s. reward: %s' % (frame_idx, rewards[-1]))
    plt.pause(0.0001)
    pass


def model_test(state):
    model = torch.load(
        'model_final_OMPMC_Diff_UEPminimization_hiddensize128.pt')
    probs, value, log_prob, entropy_timestep = model(state)
    action_arr = probs.detach().numpy()
    rho, w, tgt = action_arr[0:N], action_arr[N:2*N], action_arr[2*N:]
    return rho, w, tgt, value, log_prob, entropy_timestep
