# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 10:45:05 2023

@author: sriniva3
"""

###################################### fp generation get time slot per batch update #################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ipywidgets import interact, interactive, fixed, interact_manual
import plotly.express as px
import plotly.graph_objects as go
import random
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from tqdm import tqdm
import pdb, os
import plotly.express as px
import plotly.graph_objects as go
import math
from tqdm import tqdm
from scipy.stats import rice
from astropy.io import fits

######## initialization values ############
Nn = 40; 
F = 1000; 

EpochLength = 256

frac = 3;
lim = frac*4*np.log(1 + np.sqrt(2))/EpochLength;

tau = 0.6;  

################# function to generate ntnat values ################

def ntnat(t,w = 0, Mm = 0, T0 = 0):
    C = 4*np.log(1+np.sqrt(2))
    #pdb.set_trace()
    
    ntnat_val = 2*Mm/(1+math.cosh((t -T0)/(w/C)))
    #print(C/w)
    return ntnat_val
    

############### function to generate ws, Mvals, T0vals #############
def return_ws_Mvals_T0vals(F,tau,lim, EpochLength):
    Mvals = []
    Mvals.extend(np.arange(1,F+1,1)**-tau)
    Mvals.extend(np.arange(1,F+1,1)**-tau)
    Mvals.extend(np.arange(1,F+1,1)**-tau)

    Nbatches = 3;
    ws, T0vals = [], []
    for kk in range(1,Nbatches+1):
        ws_var = (4*np.log(1 + np.sqrt(2))/(np.random.uniform(0,1,F))/lim)
        #print(np.max(ws_var), np.min(ws_var), np.mean(ws_var), np.median(ws_var))
        T0vals_var = EpochLength*(kk - 2) + EpochLength*np.random.uniform(0,1,F)
        ws.extend(ws_var)
        T0vals.extend(T0vals_var)
        
    return  ws, Mvals, T0vals
    
    
    
    
############ function to generate PopuFunsPerBatch for the same ws, Mvals, T0vals per update ###########

def generate_PopuFunsPerBatch(t, ws, Mvals, T0vals):  
    PopuFunsPerBatch = [];
    for i in [0,1000,2000]:
        ws_temp, Mvals_temp, T0vals_temp = ws[i:i+1000], Mvals[i:i+1000], T0vals[i:i+1000]
        ntnat_vals = [ntnat(t = t, w = w, Mm = Mm, T0 = T0) for (w, Mm, T0) in zip(ws_temp, Mvals_temp, T0vals_temp)]
        PopuFunsPerBatch.append(ntnat_vals)
    PopuFunsPerBatch = np.array(PopuFunsPerBatch)
    return PopuFunsPerBatch

############# func to flush old files but keep already cached files(end of previous update) 
############# and add new files ###########################################################

def flush_old_fls_add_new_fls(nbatchs, ws, Mvals, T0vals, MostPopulars, st_ind, ed_ind):
    #nbatch, st_ind, ed_ind = 4, 0, 1000
    total_len = len(T0vals)
    inds = MostPopulars[MostPopulars < ed_ind] ######## keep files that are already cached 
    flush_inds = [i for i in range(st_ind, ed_ind) if i not in inds]

    #print('inds', inds)
    #print('most popular', MostPopulars)
    ws, Mvals, T0vals = np.array(ws), np.array(Mvals), np.array(T0vals)

    len_new = total_len - (total_len - 1000 + len(inds))

    ws_new = 4*np.log(1 + np.sqrt(2))/(np.random.uniform(0,1,len_new))/lim
    Mvals_new = [(f+1)**-tau for f in range(1, len_new+1)]
    T0vals_new = EpochLength*(nbatchs - 2) + EpochLength*np.random.uniform(0,1,len_new)



    r1 = np.concatenate((ws[inds], ws[total_len-2000:total_len-1000], ws[total_len-1000:total_len], ws_new))
    r2 = np.concatenate((Mvals[inds], Mvals[total_len-2000:total_len-1000], Mvals[total_len-1000:total_len], Mvals_new))

    T0vals_mid = EpochLength*(nbatchs -1 - 2) + EpochLength*np.random.uniform(0,1,ed_ind)
    T0vals_init = EpochLength*(nbatchs - 2  - 2) + EpochLength*np.random.uniform(0,1,ed_ind)
    r3 = np.concatenate((T0vals[inds], T0vals_init, T0vals_mid, T0vals_new))
    #pdb.set_trace()
    return list(r1), list(r2), list(r3)


########## generate fp for slot 1, update 1 ######################
def slot1_up1(ws, Mvals, T0vals, Nn, slot = 1, plot = False):
    all_popus = []
    slot = 1
    popus_acrss_slots = []
    PopuFunsPerBatch = generate_PopuFunsPerBatch(slot, ws, Mvals, T0vals)

    popus = PopuFunsPerBatch.flatten()
    #print(np.max(popus), np.min(popus), np.mean(popus), np.median(popus))
    MostPopulars = np.argsort(popus)[::-1][0:Nn]

    PopularitiesNow = popus[MostPopulars]
    PopularitiesNow/= np.sum(PopularitiesNow)
    popus_acrss_slots.append(PopularitiesNow)
    all_popus.append(popus)
    #plt.hist(popus, range=[0, 0.06], bins = 20)
    #plt.show()
    if plot:
        plt.plot(PopularitiesNow,'*')
        plt.show()
        print('previous',len(MostPopulars[MostPopulars<1000]))
        print('next',len(MostPopulars[MostPopulars>2000]))
        print('middle', 40 - (len(MostPopulars[MostPopulars<1000]) + len(MostPopulars[MostPopulars>2000])))
    
    return all_popus, popus_acrss_slots, MostPopulars, PopularitiesNow 
########## generate fp from slots 2 to 256, update 1 ######################

def slot2_257_up1(ws, Mvals, T0vals, Nn, all_popus, popus_acrss_slots, MostPopulars, PopularitiesNow, plot = False):
    
    for slot in range(2,257):
        #slot = slot + 1
        #print(slot)
        #pdb.set_trace()
        MostPopularsPrevious = MostPopulars

        PopuFunsPerBatch = generate_PopuFunsPerBatch(slot, ws, Mvals, T0vals)
        popus = PopuFunsPerBatch.flatten()
        #print(np.max(popus), np.min(popus), np.mean(popus), np.median(popus))
        MostPopulars = np.argsort(popus)[::-1][0:Nn]

        #print('previous files')
        #print(MostPopularsPrevious)
        #print('present files')
        #print(MostPopulars)


        #MostPopularsPrevious == MostPopulars
        DroppedFiles = [f for f in MostPopularsPrevious if f not in MostPopulars]
        NewFiles = [f for f in MostPopulars if f not in MostPopularsPrevious]
        #print('dropped files','new files')
        #print(DroppedFiles, NewFiles)
        DropPositions = [i for d in DroppedFiles for i in range(len(MostPopularsPrevious)) if d == MostPopularsPrevious[i]]
        MostPopulars = MostPopularsPrevious
        #print('DropPositions', DropPositions)
        MostPopulars[DropPositions] = NewFiles
        #print(MostPopulars)
        PopularitiesNow = popus[MostPopulars]
        PopularitiesNow/= np.sum(PopularitiesNow)

        #plt.plot(PopularitiesNow,'*')
        #plt.show()
        popus_acrss_slots.append(PopularitiesNow)
        all_popus.append(popus)
        if plot:
            print('previous',len(MostPopulars[MostPopulars<1000]))
            print('next',len(MostPopulars[MostPopulars>2000]))
            print('middle', 40 - (len(MostPopulars[MostPopulars<1000]) + len(MostPopulars[MostPopulars>2000])))
        
    return all_popus, popus_acrss_slots, MostPopulars, PopularitiesNow

########## generate fp for slot 1, update n ######################
def slot1_upn(ws, Mvals, T0vals, Nn, MostPopulars, slot = 1, plot = False):
    ws, Mvals, T0vals = flush_old_fls_add_new_fls(3, ws, Mvals, T0vals, MostPopulars, st_ind = 0 , ed_ind = 1000)
    all_popus = []
    slot = 1
    popus_acrss_slots = []
    PopuFunsPerBatch = generate_PopuFunsPerBatch(slot, ws, Mvals, T0vals)

    popus = PopuFunsPerBatch.flatten()
    #print(np.max(popus), np.min(popus), np.mean(popus), np.median(popus))
    MostPopulars = np.argsort(popus)[::-1][0:Nn]

    PopularitiesNow = popus[MostPopulars]
    PopularitiesNow/= np.sum(PopularitiesNow)
    popus_acrss_slots.append(PopularitiesNow)
    all_popus.append(popus)
    #plt.hist(popus, range=[0, 0.06], bins = 20)
    #plt.show()
    if plot:
        plt.plot(PopularitiesNow,'*')
        plt.show()
        print('previous',len(MostPopulars[MostPopulars<1000]))
        print('next',len(MostPopulars[MostPopulars>2000]))
        print('middle', 40 - (len(MostPopulars[MostPopulars<1000]) + len(MostPopulars[MostPopulars>2000])))
    
    return all_popus, popus_acrss_slots, MostPopulars, PopularitiesNow, ws, Mvals, T0vals 
########## generate fp from slots 2 to 256, update n ######################

def slot2_257_upn(ws, Mvals, T0vals, Nn, all_popus, popus_acrss_slots, MostPopulars, PopularitiesNow, plot = False):
    for slot in range(2,257):
        #slot = slot + 1
        #print(slot)
        MostPopularsPrevious = MostPopulars

        PopuFunsPerBatch = generate_PopuFunsPerBatch(slot, ws, Mvals, T0vals)
        popus = PopuFunsPerBatch.flatten()
        #print(np.max(popus), np.min(popus), np.mean(popus), np.median(popus))
        MostPopulars = np.argsort(popus)[::-1][0:Nn]

        #print('previous files')
        #print(MostPopularsPrevious)
        #print('present files')
        #print(MostPopulars)


        #MostPopularsPrevious == MostPopulars
        DroppedFiles = [f for f in MostPopularsPrevious if f not in MostPopulars]
        NewFiles = [f for f in MostPopulars if f not in MostPopularsPrevious]
        #print('dropped files','new files')
        #print(DroppedFiles, NewFiles)
        DropPositions = [i for d in DroppedFiles for i in range(len(MostPopularsPrevious)) if d == MostPopularsPrevious[i]]
        MostPopulars = MostPopularsPrevious
        #print('DropPositions', DropPositions)
        MostPopulars[DropPositions] = NewFiles
        #print(MostPopulars)
        PopularitiesNow = popus[MostPopulars]
        PopularitiesNow/= np.sum(PopularitiesNow)

        #plt.plot(PopularitiesNow,'*')
        #plt.show()
        popus_acrss_slots.append(PopularitiesNow)
        all_popus.append(popus)
        if plot:
            print('previous',len(MostPopulars[MostPopulars<1000]))
            print('next',len(MostPopulars[MostPopulars>2000]))
            print('middle', 40 - (len(MostPopulars[MostPopulars<1000]) + len(MostPopulars[MostPopulars>2000])))
        
    return all_popus, popus_acrss_slots, MostPopulars, PopularitiesNow

########################### adding directed walk to the file popularity evolution ##########################

def rician_sampling(alpha, beta):
    y = rice.rvs(b = alpha, size = 1)
    sample = np.maximum(np.random.normal(alpha, beta),0)
    return sample

def directed_walk_func1(ts, ntnat_vals, k1 = 3/4, variance_scale = 10):
    p = ntnat_vals[0]
    #TimesOfInterest = np.arange(-100,100,1)
    TimesOfInterest = ts
    trendValues = ntnat_vals
    all_vals = []
    alphas, betas = [], []
    for trendValue in trendValues:
        alpha = (p*k1 + trendValue*(1-k1))
        beta = trendValue/variance_scale
        p = rician_sampling(alpha, beta)
        #pdb.set_trace()
        #if p<0:pdb.set_trace()
        #x = np.arange(alpha-2*beta,alpha+2*beta,0.001)
        #p = np.random.choice(x)
        all_vals.append((trendValue, p))
        alphas.append(alpha)
        betas.append(beta)
    trend = [v[0] for v in all_vals]
    directed_walk = [v[1] for v in all_vals]
    return np.array(trend), np.array(directed_walk)


def directed_walk_func2(ts, trend, alpha_t = 0.5, sf_num1 = 25, sf_num2 = 30):
    
    trend_tilde = []
    trend_tilde.append(trend[0])
    prev_t = 0
    for i in range(1,len(trend)):
        alpha = np.random.uniform(0,alpha_t)    
        mu = alpha*(trend[i-1]) + (1-alpha)*trend[i]
        #std = trend[i]/10
        scaling_factor = np.random.uniform(sf_num1, sf_num2)
        #scaling_factor = 20
        std = np.maximum(np.random.normal(trend[i]/scaling_factor,trend[i]/scaling_factor),0)
        #std =trend[i]/scaling_factor
        val = np.maximum(np.random.normal(mu, std),0)
        trend_tilde.append(val)
    trend_tilde = np.array(trend_tilde)
    
    if np.max(trend_tilde>1):
        trend_tilde/=np.max(trend_tilde)
        
    return np.array(trend), np.array(trend_tilde)



def directed_walk_for_1update(fp):
    ts = np.arange(0,256,1)
    fp_directed_walk = []
    return_trend = []
    for i in range(fp.shape[1]):
        ntnat_vals = fp[:,i]
        #trend, directed_walk = directed_walk_func2(ts, ntnat_vals, k1 = np.random.uniform(0.5,.75),
                                                 #variance_scale = np.random.uniform(10,20))
            
        trend, directed_walk = directed_walk_func2(ts, ntnat_vals, alpha_t = 0.5, sf_num1 = 25, sf_num2 = 30)  
        
        fp_directed_walk.append(directed_walk)
        return_trend.append(trend)
    fp_directed_walk = np.transpose(np.array(fp_directed_walk))
    return_trend = np.transpose(np.array(return_trend))

    return return_trend, fp_directed_walk



def directed_walk_mdp(popus_acrss_slots, splus, s, outage, ts, k = 0.9):
    
    if ts == 0:
        
        delta_rt = popus_acrss_slots[ts]
        alpha = k*delta_rt 
    else: 
        delta_rt = popus_acrss_slots[ts] - popus_acrss_slots[ts-1]
        #dt = (splus - s)/((1 - s)*(1 - outage))
        dt = splus - s
        dt = np.minimum(dt,1)
        dt = np.maximum(dt,-1)
        alpha = popus_acrss_slots[ts] + k*delta_rt + (1 - k)*dt
        
    
    beta = np.abs(delta_rt)/0.4;
    #pdb.set_trace()
    fp_directed_walk = np.maximum(np.random.normal(alpha,beta),0)
    fp_directed_walk/= np.sum(fp_directed_walk)
    
    #pdb.set_trace()
    return popus_acrss_slots, fp_directed_walk










