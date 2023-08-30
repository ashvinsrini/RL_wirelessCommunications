# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 10:45:49 2023

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

Q = np.array([[0.9,0.033,0.033,0.034],[0.2, 0.7, 0.05, 0.05],[0.05, 0.05, 0.85, 0.05],[0.1, 0.1, 0.1, 0.7]])
skewness = np.array([2, 2.5, 3, 3.5])
fp_states = np.array([0,1,2,3])
N, Lc, Lu, lamb_bh, lamb_ue = 40, 6, 3, 0.05, 0.05



def generate_zipf(tau, N):
        #N = 100 # number of files
        denom = np.sum([np.power(1/i, tau) for i in range(1,N + 1)])
        zipf = np.array([np.power(1/n,tau)/denom for n in range(1, N+1)])
        #plt.plot(zipf, '*')
        return zipf

def get_present_state(fp_mdp, ue_cache_prob_lst, prev_fp_lst):
    state = np.vstack((prev_fp_lst[0], ue_cache_prob_lst[0], prev_fp_lst[1], ue_cache_prob_lst[1],
                          prev_fp_lst[2], ue_cache_prob_lst[2],fp_mdp, ue_cache_prob_lst[3]))
    
    #state = np.vstack((prev_fp_lst[0], prev_fp_lst[1], prev_fp_lst[2], fp_mdp,
                      #ue_cache_prob_lst[0], ue_cache_prob_lst[1], ue_cache_prob_lst[2], ue_cache_prob_lst[3]))
    return state


def get_init_file_popularity(init_fp_state, N):
  fp_zipf = generate_zipf(skewness[init_fp_state], N)
  return fp_zipf

def get_init_ue_cache_prob(N = N):
  Lu = 3
  x = np.random.uniform(size = N)
  probs = Lu*x/np.sum(x)
  return probs

def get_next_file_popularity(next_fp_state, N):
  fp_zipf = generate_zipf(skewness[next_fp_state], N)
  return fp_zipf

'''
file_popularities = []
file_popularities.append([fp_state,1,init_fp_zipf])
for i in range(0,10):
  fp_dist = Q[fp_state,] 
  fp_chosen_prob = np.random.choice(fp_dist, size=1, p=fp_dist)[0]
  next_fp_state = np.argwhere(fp_dist == fp_chosen_prob)
  next_fp_state = [next_fp_state[i][0] for i in range(len(next_fp_state))]
  next_fp_state = np.random.choice(next_fp_state)
  #print(fp_chosen_prob, next_fp_state)
  #print(next_fp_state)
  fp_zipf = get_next_file_popularity(next_fp_state, N)
  fp_state = next_fp_state
  file_popularities.append([fp_state,fp_chosen_prob,fp_zipf])
  #plt.figure(i)
  #plt.plot(fp_zipf, '*')
  #plt.title('for {}'.format(next_fp_state))
  #print(fp_zipf)
''';


def get_next_fp(fp_state, Q, N):
  ######## based on Q Markovian table #############
  fp_dist = Q[fp_state,] 
  fp_chosen_prob = np.random.choice(fp_dist, size=1, p=fp_dist)[0]
  next_fp_state = np.argwhere(fp_dist == fp_chosen_prob)
  next_fp_state = [next_fp_state[i][0] for i in range(len(next_fp_state))]
  next_fp_state = np.random.choice(next_fp_state)
  fp_zipf = get_next_file_popularity(next_fp_state, N)
  fp_state = next_fp_state
  return fp_state,fp_chosen_prob,fp_zipf


################### get next UE caching probability ##############
#'''
def get_init_channel_coeff(N):
  w = np.random.uniform(2, size=N)
  return w

def get_init_bs_cache_prob(N):
  rho = np.random.uniform(size = N)
  return rho
#'''
def get_outage_prob(rho,w):
  lambdaa, sigma, avg_pow, alpha =  300, 1, 1, 1/10
  
  #w_th = np.array([1e-2 for _ in range(len(w))])
  #w = np.minimum(w_th,w)
  #w/=np.sum(w)
  channel_gain_th = (np.power(sigma,2)/avg_pow)*(np.power(2,alpha/w) - 1)
  temp_result = (np.power(np.pi,2)*lambdaa*rho)/(4*np.sqrt(2*channel_gain_th))
  outage = [math.erfc(temp) for temp in temp_result]
  #pdb.set_trace()
  return np.array(outage)
#init_ue_cache_prob

def get_next_ue_cache_prob(target_prob, outage, Lu, prev_cache_prob, N):
    next_ue_cache_prob = target_prob
    #pdb.set_trace()
    return next_ue_cache_prob

def get_qos_reward(fp_prob, prev_ue, outage, N):
  #rqos = 1 - np.sum(fp_prob*(ue_cache_prob + (1 - ue_cache_prob)*(1 - outage)))
  request = np.multiply(fp_prob, 1 - prev_ue)
  #request = fp_prob
  #rqos = 1/np.sum(request * outage)
  rqos = -np.sum(request * outage)
  return rqos

def get_bh_reward(Lc, present_rho, previous_rho, N):
  val = present_rho - previous_rho
  val_plus = 0.5*(val + np.abs(val))
  #rbh = 1/np.sum(val_plus)
  rbh = -np.sum(val_plus)
  return rbh


def get_uep_reward(Lu, target_prob, previous_ue, N, outage):
  inds = np.where(target_prob > previous_ue)
  cost = np.sum(np.minimum((target_prob[inds] - previous_ue[inds])/((1 - previous_ue[inds])*(1 - outage[inds])),1))
  ruep = -cost  
  #cost = np.sum((target_prob[inds] - previous_ue[inds])/((1 - previous_ue[inds])*(1 - outage[inds])))
  #ruep = -np.minimum(cost,1)
  #pdb.set_trace()
    
  return ruep


def env(fp_state, ue_cache_prob, rho, w, prev_ue, prev_rho, N, Lc, Lu, lamb_bh, Q, fp_table, tgt_prob, 
       fp_mdp, ue_cache_prob_lst, prev_fp_lst):
  ############# get outage, next state, and rewards ########

  outage = get_outage_prob(rho,w)
  #pdb.set_trace()
  #print(np.sum(outage))
  next_ue_cache_prob = get_next_ue_cache_prob(tgt_prob, outage, Lu, prev_ue, N)
  
  '''
  present_file_popularity = generate_zipf(skewness[fp_state], N)

  next_fp_state, fp_chosen_prob , _ = get_next_fp(fp_state, Q, N)
  next_file_popularity = generate_zipf(skewness[next_fp_state], N)
  '''
  
  present_file_popularity = fp_table[fp_state]
  next_fp_state = fp_state + 1
  if next_fp_state == 256: next_fp_state = fp_state
  next_file_popularity = fp_table[next_fp_state]


   
  #pdb.set_trace()
  rqos = get_qos_reward(present_file_popularity, prev_ue, outage, N)

  rbh = get_bh_reward(Lc, rho, prev_rho, N)
  #pdb.set_trace()
  ruep = get_uep_reward(Lu, next_ue_cache_prob, prev_ue, N, outage)

  r = rqos + lamb_bh * rbh + lamb_ue * ruep
    
  next_ue_cache_prob = np.concatenate((ue_cache_prob[1:] , np.reshape(next_ue_cache_prob,(1,-1))))
  #pdb.set_trace()
  next_state = get_present_state(fp_mdp, ue_cache_prob_lst, prev_fp_lst)
  #done = False
  #print('current fp state', fp_state, 'next fp state', next_fp_state, 'chose fp prob', fp_chosen_prob)
  #pdb.set_trace()
  return next_state, r, next_fp_state, next_ue_cache_prob, rqos, rbh, ruep 




