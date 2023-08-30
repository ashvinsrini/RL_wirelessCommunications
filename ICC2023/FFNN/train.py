# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 10:43:51 2023

@author: sriniva3
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
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
from model import *
from system_model import *
import pickle
from astropy.io import fits
from fp_generation import *
from torchsummary import summary
use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")

save_pth = r'C:\Users\sriniva3\OneDrive - Aalto University\Simulations\RL framework OMPMC\OMPMC_diffusion_based_UEpower_mathmetical_formulation_update_equation_newFP\models'

################ Environment for OMPMC #############################
Q = np.array([[0.9,0.033,0.033,0.034],[0.2, 0.7, 0.05, 0.05],[0.05, 0.05, 0.85, 0.05],[0.1, 0.1, 0.1, 0.7]])
skewness = np.array([2, 2.5, 3, 3.5])
fp_states = np.array([0,1,2,3])
N, Lc, Lu, lamb_bh, lamb_ue = 40, 6, 3, 0.05, 0.05
#init_fp_state = random.choice(fp_states)
init_fp_state = 1
data = fits.open(r'C:\Users\sriniva3\OneDrive - Aalto University\Simulations\RL framework OMPMC\OMPMC_diffusion_based_UEpower\mathematica_fp\fp_2.fits')
fp_table = np.transpose(np.array([data[i].data for i in range(1)])[0])
#fp_table = np.load('file_popularity.npy')

############# getting first 16 time slots UE cache probabilities ############
init_ue_cache_prob_list = []
for i in range(4):
    init_ue_cache_prob = get_init_ue_cache_prob(N = N)
    init_ue_cache_prob_list.append(init_ue_cache_prob)
init_ue_cache_prob_list = np.array(init_ue_cache_prob_list)
##############################################################################

#init_fp_zipf = get_init_file_popularity(init_fp_state, N)
init_fp_zipf = fp_table[init_fp_state]
#state = get_present_state(init_fp_zipf, init_ue_cache_prob)


########### generate next skewness for file popularity #########
fp_state = init_fp_state


#w, rho = get_init_channel_coeff(N), get_init_bs_cache_prob(N)
########### Initialization ############
#num_inputs = np.prod(state.shape)
num_inputs = 2*4*N #### 2 for UE caching probability and file popularity, 16 for previous time slots plus present time slot
num_outputs = 3*N
#Hyper params:
hidden_size = 128
lr = 7e-2
n = 1
num_steps   = n*256
Nu = 256
done = False


model = ActorCritic(num_inputs, num_outputs, hidden_size, Lc).to(device)
#model = torch.load('model_final_OMPMC_Diff_UEPminimization_hiddensize128_unsorted_file.pt')



optimizer = optim.Adam(model.parameters())

all_acc_rewards, all_acc_rewardsqos, all_acc_rewardsbh, all_acc_rewardsuep = [], [], [], []
#acc_reward = 0
entropy_epochs = []

rho_arr, w_arr, tgt_prob_arr = [], [], []
for _ in range(4):
    t1, t2, t3 = np.random.uniform(0,1,N), np.random.uniform(0,1,N), np.random.uniform(0,1,N)
    rho_arr.append(Lc*t1/np.sum(t1))
    w_arr.append(t2/np.sum(t2))
    tgt_prob_arr.append(Lu*t3/np.sum(t3))

rho_arr,w_arr,tgt_prob_arr = np.array(rho_arr), np.array(w_arr), np.array(tgt_prob_arr)


for epoch in range(0, 2_500):
      #print('epoch', epoch)
      #init_fp_state = np.random.choice(np.array([i for i in range(2*950 - num_steps)]))
      init_fp_state = np.random.choice(np.array([i for i in range(1,20,1)]))
      fp_state, ue_cache_prob = init_fp_state, init_ue_cache_prob_list
    
      if epoch == 0:
            ws, Mvals, T0vals = return_ws_Mvals_T0vals(F,tau,lim, EpochLength)
            all_popus, popus_acrss_slots, MostPopulars, PopularitiesNow = slot1_up1(ws, Mvals, T0vals, Nn)
            all_popus, popus_acrss_slots, MostPopulars, PopularitiesNow = slot2_257_up1(ws, Mvals, T0vals, Nn, all_popus, popus_acrss_slots, MostPopulars, PopularitiesNow)
            popus_acrss_slots = np.array(popus_acrss_slots)
            _, fp_directed_walk = directed_walk_for_1update(popus_acrss_slots)
            #pdb.set_trace()
            Popus_acrss_slots = fp_directed_walk
        
            
      #'''
      else:
            all_popus, popus_acrss_slots, MostPopulars, PopularitiesNow, ws, Mvals, T0vals = slot1_upn(ws, Mvals, T0vals, Nn, MostPopulars)
            all_popus, popus_acrss_slots, MostPopulars, PopularitiesNow = slot2_257_upn(ws, Mvals, T0vals, Nn, all_popus, popus_acrss_slots, MostPopulars, PopularitiesNow) 
            popus_acrss_slots = np.array(popus_acrss_slots)
            _, fp_directed_walk = directed_walk_for_1update(popus_acrss_slots)
            pass

      #'''
      full_epi_rewards = []
      full_epi_rewardsqos, full_epi_rewardsbh, full_epi_rewardsuep = [], [], []
    
      log_probs = []
      values    = []
      rewards   = []
      rewards_qos, rewards_bh, rewards_uep = [], [], []
      masks     = []
      entropy = 0
      prev_rho = np.zeros(N)
      prev_ue = np.zeros(N)
      done = 0
      counter = 1
      acc_reward, acc_reward_qos, acc_reward_bh, acc_reward_uep = 0, 0, 0, 0
    
    
      for itera, time_step in enumerate(range(3, num_steps)):
        #print('time_step',time_step)
        #fp_zipf = fp_directed_walk[time_step]  
        #state = get_present_state(fp_zipf, init_ue_cache_prob_list, popus_acrss_slots, time_step)
        
        if time_step == 3:
            ###### for intial 4 time slots, LSTM 4 states #########
            splus = tgt_prob_arr[3] # splus is the latest target distribution, for 4 LSTM, it is 3.
            s = tgt_prob_arr[2] # s is the previous target distribution
            outage = get_outage_prob(rho_arr[3],w_arr[3])
            _,fp_mdp = directed_walk_mdp(popus_acrss_slots, splus, s, outage, ts = time_step, k = 0.9)

            ue_cache_prob_lst = tgt_prob_arr
            prev_fp_lst = popus_acrss_slots[0:time_step]

            state = get_present_state(fp_mdp, ue_cache_prob_lst, prev_fp_lst)
            #pdb.set_trace()
        else:
            ######### for 5th state onwards###############
            new_tgt = tgt_prob ########## new_tgt is output from the model 
            new_rho = rho
            new_w = w

            #print('time_step',time_step, 'tgt_prob_arr_shape', tgt_prob_arr.shape)
            #pdb.set_trace()
            tgt_prob_arr = np.vstack((tgt_prob_arr[1:4], new_tgt))
            
            rho_arr = np.vstack((rho_arr[1:4], new_rho))
            w_arr = np.vstack((w_arr[1:4], new_w))
            prev_fp_lst = np.vstack((popus_acrss_slots[1:3], fp_mdp))

            splus = tgt_prob_arr[3] # splus is the latest target distribution, for 4 LSTM, it is 3.
            s = tgt_prob_arr[2] # s is the previous target distribution
            outage = get_outage_prob(rho_arr[3],w_arr[3])
            _,fp_mdp = directed_walk_mdp(popus_acrss_slots, splus, s, outage, ts = time_step, k = 0.9)

            ue_cache_prob_lst = tgt_prob_arr

            state = get_present_state(fp_mdp, ue_cache_prob_lst, prev_fp_lst)
            

            
        #pdb.set_trace()
        #'''

        #'''
        
        #pdb.set_trace()
        state = state.flatten()
        state = torch.FloatTensor(state.astype(np.float32)).to(device)
        state = np.reshape(state, [1, len(state)])
        #pdb.set_trace()

        probs, value,log_prob, entropy_timestep = model(state)
        #print(log_prob, entropy_timestep)
        #pdb.set_trace()
        #action_vect = dist.probs.detach().numpy()[0]
        action_vect = probs.detach().numpy()
        
        rho, w, tgt_prob = action_vect[0:N], action_vect[N:2*N], action_vect[2*N:]
        #pdb.set_trace()
        #print('----------------------------------------')
        #print(np.sum(tgt_prob))
        #pdb.set_trace()
        next_state, r, next_fp_state, next_ue_cache_prob, rqos, rbh, ruep = env(time_step, ue_cache_prob, rho, w, prev_ue,
                                                                          prev_rho, N, Lc, Lu, lamb_bh, Q, popus_acrss_slots, 
                                                                                tgt_prob, fp_mdp, ue_cache_prob_lst, prev_fp_lst)
        next_ue_cache_prob.shape
        #pdb.set_trace()
        ######## computing losses #####################
        #log_prob = np.log(action_vect) ####### np.log (chosen action)
        #entropy+= np.mean(-action_vect*np.log(action_vect)) ########## -summation((p*np.log(p)))
        #print(log_prob, np.mean(-action_vect*log_prob))
        entropy+= entropy_timestep
        #pdb.set_trace()
        log_probs.append(torch.Tensor(log_prob))
        values.append(torch.Tensor(value))
        #pdb.set_trace()
        rewards.append(torch.FloatTensor(np.reshape(r,[1])).unsqueeze(1).to(device))
        rewards_qos.append(torch.FloatTensor(np.reshape(rqos,[1])).unsqueeze(1).to(device))
        rewards_bh.append(torch.FloatTensor(np.reshape(rbh,[1])).unsqueeze(1).to(device))
        rewards_uep.append(torch.FloatTensor(np.reshape(ruep,[1])).unsqueeze(1).to(device))


        full_epi_rewards.append(torch.FloatTensor(np.reshape(r,[1])).unsqueeze(1).to(device))
        full_epi_rewardsqos.append(torch.FloatTensor(np.reshape(rqos,[1])).unsqueeze(1).to(device))
        full_epi_rewardsbh.append(torch.FloatTensor(np.reshape(rbh,[1])).unsqueeze(1).to(device))
        full_epi_rewardsuep.append(torch.FloatTensor(np.reshape(ruep,[1])).unsqueeze(1).to(device))
        
        
        #if np.mod(counter, 2*time_step) == 0:
          #done = True
          #state = get_present_state(init_fp_zipf, init_ue_cache_prob)
        #else: 
        #masks.append(1 - torch.FloatTensor(1 - done).unsqueeze(1).to(device))
        masks.append(1 - done)
        ue_cache_prob_cpy = ue_cache_prob[-1].copy()
        fp_state, ue_cache_prob, prev_rho, prev_ue =  next_fp_state, next_ue_cache_prob, rho, ue_cache_prob_cpy
        #pdb.set_trace()


        if counter+3 == Nu:
          next_state = next_state.flatten()
          next_state = torch.FloatTensor(next_state.astype(np.float32)).to(device)
          next_state = np.reshape(next_state, [1, len(next_state)])
          _, next_value, _, _ = model(next_state)
          #pdb.set_trace()
          returns = compute_returns(next_value, rewards, masks)
          #print('entering update mode')
          log_probs = torch.cat(log_probs)
          returns   = torch.cat(returns).detach()
          values    = torch.cat(values)

          #pdb.set_trace()
          ################# computing advantage function ##############
          advantage = returns - values

          ################# computing losses and summing ##############
          actor_loss  = -(log_probs * advantage.detach()).sum()
          #actor_loss = -np.sum(np.array([np.array(log_probs[i*2*N:(i+1)*2*N]*advantage[i].detach().numpy()) for i in range(len(advantage))]))/len(log_probs)
          #actor_loss = torch.tensor(actor_loss)
          #critic_loss = advantage.pow(2).mean()
          critic_loss = (values*advantage.detach()).sum()

          loss = actor_loss +  critic_loss - 0.05*entropy
          loss.backward()
          optimizer.step()
          log_probs, returns, values, masks, rewards, entropy, rewards_qos, rewards_bh, rewards_uep = [], [], [], [], [], 0, [], [], []
          #print('Finished updating', counter, Nu, time_step)
          counter = 1

        counter+=1 
        if itera == num_steps: done = 1
        
      if epoch%1 ==0: 
        #pdb.set_trace()
        #acc_reward+= np.sum(full_epi_rewards).detach().numpy()[0][0]
        acc_reward+= np.sum(full_epi_rewards)
        acc_reward_qos+= np.sum(full_epi_rewardsqos)
        acc_reward_bh+= np.sum(full_epi_rewardsbh)
        acc_reward_uep+= np.sum(full_epi_rewardsuep)
        
        all_acc_rewards.append(acc_reward)
        all_acc_rewardsqos.append(acc_reward_qos)
        all_acc_rewardsbh.append(acc_reward_bh)
        all_acc_rewardsuep.append(acc_reward_uep)
        
        entropy_epochs.append(entropy)
        
        #init_state = get_present_state(init_fp_zipf, init_ue_cache_prob)
        #total_reward = test_env(model, device, init_state, fp_state, ue_cache_prob, rho, w, prev_rho, num_steps)
        #all_acc_rewards.append(total_reward)
        print(epoch, acc_reward, acc_reward_qos, acc_reward_bh, acc_reward_uep)
        #print(loss, log_prob, entropy)        
        if epoch%100==0:
            plt.figure(1)
            plt.plot(-1*np.array(all_acc_rewardsqos)/256,label = 'qos')
            plt.legend()
            plt.show()
            plt.figure(2)
            plt.plot(-1*np.array(all_acc_rewardsbh)/256,label = 'bh')
            plt.legend()
            plt.show()
            plt.figure(3)
            plt.plot(-1*np.array(all_acc_rewardsuep)/256,label = 'uep')
            plt.legend()
            plt.show()
            #print(epoch, acc_reward, acc_reward_qos, acc_reward_bh, acc_reward_uep)
            #print(loss, log_prob, entropy)
            #pdb.set_trace()
            #if acc_reward>=-75:
                #torch.save(model, os.path.join(save_pth,'model_{}.pt'.format(acc_reward.detach().numpy()[0][0]/256)))
            pass


