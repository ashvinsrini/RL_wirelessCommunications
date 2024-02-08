import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from tqdm import tqdm
import gym
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from sklearn.neighbors import KDTree

from model2_LSTM import *
from itertools import product
import pickle
import joblib
import random
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
import warnings
warnings.filterwarnings("ignore")
import pdb
from scipy.special import jv
from scipy import special

class env:
    
    def __init__(self, N, U, B = 2):
        self.N = N
        self.U = U
        self.B = B
        pass
    
    
    def get_jakes_coeffcients(self, complex_samples, prev_h, corr_0, mu_0):
        
        #'''
        rnd_samples = np.random.choice(complex_samples, size = (self.N,self.U))
        h = []
        for i in range(self.U):
            h.append(corr_0[i]*prev_h[:,i] + mu_0[i]*rnd_samples[:,i])
        h = np.transpose(np.array(h))
        h/=np.sqrt(2)
        #'''
        
        return h
        
    def get_jakes_coeffcients_actual(self, TimeSequences, all_pathGains, time_ind = 2, BS_ind = 0):
        
        '''
        rnd_samples = np.random.choice(complex_samples, size = (self.N,self.U))
        h = []
        for i in range(self.U):
            h.append(corr_0[i]*prev_h[:,i] + mu_0[i]*rnd_samples[:,i])
        h = np.transpose(np.array(h))
        h/=np.sqrt(2)
        '''
        ############ For jakes coefficient ###############
        if BS_ind == 0:
            h = TimeSequences[:,time_ind][0:self.N*self.U]
            ch_pg = all_pathGains[:,time_ind][0:self.N*self.U]
        else: 
            h = TimeSequences[:,time_ind][BS_ind*(self.N*self.U):(BS_ind+1)*(self.N*self.U)]
            ch_pg = all_pathGains[:,time_ind][BS_ind*(self.N*self.U):(BS_ind+1)*(self.N*self.U)]
            
        h = np.reshape(h,(self.N,self.U))
        ch_pg = np.reshape(ch_pg,(self.N,self.U))
        #h/=np.sqrt(2)
        h = np.sqrt(ch_pg)*h
        h/=np.sqrt(2)
        return h   
    
    def get_state(self, h, all_h, all_A, serve_BS_ind = 0):
        
        #p = 1
        B = self.B
        i_BS_inds = [i for i in range(B) if i!=serve_BS_ind]
        other_h = all_h[i_BS_inds]
        other_A = all_A[i_BS_inds]
        #pdb.set_trace()
        
        #p_r = np.abs(np.random.normal(9.790121497600237e-06, 3.26163510385863e-06, (self.N,self.U)))
        #p_i1 = np.abs(np.random.normal(8.405267245958328e-09, 2.499866701653188e-09, (self.N,self.U)))
        #p_i2 = np.abs(np.random.normal(8.405267245958328e-09, 2.499866701653188e-09, (self.N,self.U)))
        
        
        inds_r = np.where(all_A[serve_BS_ind] == 1.0)
        inds_i1 = np.where(other_A[0] == 1.0)
        #inds_i2 = np.where(other_A[1] == 1.0)
        #v_r = np.random.uniform(5.216186024347601e-09, 1.9575026809176127e-09, len(inds_r[0]))
        #v_i1 = np.random.uniform(9.056671409987651e-10, 1.590486735091789e-08, len(inds_i1[0]))
        #v_i2 = np.random.uniform(9.056671409987651e-10, 1.590486735091789e-08, len(inds_i2[0]))

        #p_r, p_i1, p_i2 = np.zeros((N,U)), np.zeros((N,U)), np.zeros((N,U))
        #p_r[inds_r], p_i1[inds_i1],  p_i1[inds_i2] = v_r, v_i1, v_i2

        #inter_power = p_i1*np.abs(other_h[0])**2*other_A[0] + p_i2*np.abs(other_h[1])**2*other_A[1]
        
        #p_r, p_i1 = np.zeros((N,U)), np.zeros((N,U))
        #p_r[inds_r], p_i1[inds_i1] = v_r, v_i1

        #inter_power = p_i1*np.abs(other_h[0])**2*other_A[0]
        inter_power = np.abs(other_h[0])**2*other_A[0]
        
        inter_power+=1e-14
        #SNR = (p_r*np.abs(all_h[0])**2*all_A[serve_BS_ind])/inter_power

        SNR = (np.abs(all_h[0])**2*all_A[serve_BS_ind])/inter_power
        
        
        
        #inter_power = p_i1*np.abs(other_h[0])**2*other_A[0] + p_i2*np.abs(other_h[1])**2*other_A[1]
        #inter_power+=1e-14
        #SNR = (p_r*np.abs(all_h[0])**2*all_A[serve_BS_ind])/inter_power
        #pdb.set_trace()
        
        #l, h = np.power(10,5/10), np.power(10,20/10)
        #inds_l = np.where((SNR!=0) & (SNR < l))
        #inds_h = np.where((SNR!=0) & (SNR > h))
        #SNR[inds_l] = np.random.normal(l+0.2,0.2,len(inds_l[0]))
        #SNR[inds_h] = np.random.normal(h-0.2,0.2,len(inds_h[0]))
        
        '''
        SNR= SNR/10
        SNR = np.reshape(SNR,(N*U))
        ind_h = np.where(SNR>0.5)
        ind_l = np.where((SNR!=0)&(SNR<=0.3))
        SNR[ind_l] = np.random.normal(0.3,0.0005,len(ind_l[0])) 
        SNR[ind_h] = np.random.normal(0.5,0.0005,len(ind_h[0])) 
        SNR = np.reshape(SNR, (N,U))
        ''';
        
        
        #SNR = np.where(np.isnan(SNR) | np.isinf(SNR), 0, SNR)
        #ind = np.where(SNR >1)
        #SNR[ind] = 1
        #pdb.set_trace()
        return SNR
        
    
    
    def get_rewards(self, SNR, action):
        N, U = self.N, self.U
        k, n = 400, 1000
        p_o = 1e-5
        lambda_1, lambda_2 = 1000, 1
        SNR = np.reshape(SNR,(N,U))
        r_ber = []
        r_res = []
        ########## Compute reward block error rate, channel resources used for every user ################
        for i in range(0,U):
            ind = np.nonzero(action[:,i])[0]
            #pdb.set_trace()
            UE_SNR = SNR[:,i][ind]
            gamma_sum = np.sum(UE_SNR)
            m = len(UE_SNR)
            
            V_gamma = 1 - (1/(1 + gamma_sum)**2)
            C_gamma = np.log2(1 + gamma_sum)

            #Q_term = (2*m*n_m*C_gamma - 2*B + np.log(m*n_m))/(2*np.sqrt(m*n_m)*np.sqrt(V_gamma))
            #pdb.set_trace()
            #if V_gamma!=0:
            Q_term = (2*n*C_gamma - 2*k + np.log2(n))/(2*np.sqrt(n)*np.sqrt(V_gamma))
                #print(Q_term, C_gamma, V_gamma, gamma_avg, m)

            P_e = 0.5-0.5*special.erf(Q_term/np.sqrt(2))
            P_e = np.min((P_e,2*1e-5*np.abs(np.random.normal(0,1,1)[0])))
            #else:
                #P_e = 0
            #pdb.set_trace()
            r_ber.extend([P_e])
            #print('---------Q_term, Pe------------')
            #print(Q_term, P_e)
            r_res.extend([m])

        #pdb.set_trace()
        r_ber = np.array(r_ber)
        #ind_high_ber  = np.where(r_ber>p_o)
        #r_ber[ind_high_ber] = 2*p_o
        #ind_low_ber  = np.where(r_ber==0)
        #r_ber[ind_low_ber] = np.abs(np.random.normal(np.random.choice([1e-7,1e-8,1e-9]),np.random.choice([1e-7,1e-8,1e-9]),1)[0])
        
        r_ber = np.where(np.isnan(r_ber) | np.isinf(r_ber), 0, r_ber)
        
        reward_ber = -np.sum(np.exp(np.array(r_ber)/p_o - 1))

        #reward_res = self.N - np.sum(np.array(r_res))
        #pdb.set_trace()
        #reward_res =  np.min((N, np.sum(np.array(r_res))))
        reward_res =  -np.sum(np.array(r_res))
        
        #pdb.set_trace()
        #print('--------------------reward_ber, reward_res-----------------')
        #print(reward_ber, reward_res)
        weighted_reward = lambda_1*reward_ber+lambda_2*reward_res
        
        #if np.isnan(weighted_reward): pdb.set_trace()
        #if np.isinf(weighted_reward): pdb.set_trace()
        return weighted_reward, r_ber, r_res, reward_res
        
        pass
    