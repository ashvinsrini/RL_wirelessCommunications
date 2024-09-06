import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
#import gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import random
import pdb
import plotly.express as px
import plotly.graph_objects as go
import pickle
import scipy.io
from matplotlib.font_manager import FontProperties
import pandas as pd
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from tqdm import tqdm
import matplotlib.animation as animation
import glob
from natsort import natsorted
from PIL import Image
from scipy.special import jv
from scipy import special





class env:
    
    def __init__(self, N = 4, J = 3, M = 10, Ts = 100, f_c = 1.3):
        self.N = N
        self.J = J
        self.M = M
        self.Ts = Ts
        self.f_c = f_c
        self.TxRxds = np.zeros(( self.Ts,  self.M,  self.M* self.J))
        self.d_intra = np.abs(np.random.normal(0.5,0.1, self.M* self.J))
        self.d_intra = np.reshape( self.d_intra, ( self.M, self.J))       
        self.compute_gamma0() 
        
    def _generate_matrix(self, N, J):
        # Initialize an NxJ matrix filled with zeros
        matrix = np.zeros((N, J), dtype=int)

        # Ensure each column has at least one 1
        for j in range(J):
            # Randomly select a row to place the 1 in column j
            while True:
                row_index = np.random.randint(N)
                # Place the 1 only if that row currently contains no 1
                if np.sum(matrix[row_index]) == 0:
                    matrix[row_index, j] = 1
                    break

        # Fill remaining rows with either a single 1 or all 0s
        for i in range(N):
            if np.sum(matrix[i]) == 0:  # If the row is all zeros
                if np.random.rand() > 0.5:  # 50% chance to add a 1
                    col_index = np.random.randint(J)
                    matrix[i, col_index] = 1

        return matrix
    def compute_gamma0(self):
        ## transmit power ###
        Pt_dBm = 10  # Transmit power in dBm

        #### Noise power ####
        k = 1.38e-23  # Boltzmann's constant
        T = 290       # Temperature in Kelvin
        B = 10e6       # Bandwidth in Hz

        # Calculate noise spectral density in Watts/Hz
        Noise_spectral_density = k * T

        # Convert noise spectral density to dBm/Hz
        # Convert to Watts first then to dBm (1 mW = 0.001 W)
        Noise_spectral_density_W = Noise_spectral_density * 1000  # Convert to mW
        Noise_spectral_density_dBm = 10 * np.log10(Noise_spectral_density_W)
        #print('Noise_spectral_density_dBm', Noise_spectral_density_dBm)

        # Calculate total noise power in Watts then convert to dBm
        N_thermal = Noise_spectral_density * B  # Total noise power in Watts
        N_thermal_dBm = 10 * np.log10(N_thermal * 1000)  # Convert noise power to dBm
        #print('thermal_noise_power(dBm)', N_thermal_dBm)

        ##### gamma_0 in dB ############
        gamma_0_dB = Pt_dBm - N_thermal_dBm  # Calculate SNR in dB
        #print('gamma_0 (dB)', gamma_0_dB)

        ##### gamma_0 ######
        gamma_0 = np.power(10, gamma_0_dB/10)
        #print('1/gamma_0', 1/gamma_0)        
        self.gamma_0 = gamma_0
        
    def return_jakes_coeffcients(self, fd_max, TimeVaris, n_links = 5, plot = True):
        ff_gains, TimeSequences = [], []
        rays = 100
        for i in tqdm(range(n_links)):    
            #TimeVaris = np.arange(0,50,0.0005)
            #TimeVaris = np.arange(0,.2,0.005)
            frequs = np.sort(np.array([np.round(fd_max*np.cos(2*np.pi*np.random.uniform(0,1))) for _ in range(rays)]))
            phases = np.array([np.exp(1j*2*np.pi*np.random.uniform(0,1)) for _ in range(rays)])

            TimeSequence = []
            for t in TimeVaris:
                tab = np.exp(1j*2*np.pi*frequs*t)
                tabrot = tab*phases
                fun = np.sum(tabrot)
                TimeSequence.append(fun)
            TimeSequence = np.array(TimeSequence)

            #TimeSequence = TimeSequence/np.linalg.norm(TimeSequence)*np.sqrt(len(TimeSequence))
            PowerSequence1 =  np.abs(TimeSequence)**2;
            #plt.plot(TimeVaris[0:200], 10*np.log10(PowerSequence1)[0:200])
            ff_gains.append(PowerSequence1)
            TimeSequences.append(TimeSequence)
        ff_gains = np.array(ff_gains)/rays
        TimeSequences = np.array(TimeSequences)
        if plot:
            plt.plot(TimeVaris[0:200], 10*np.log10(ff_gains[0])[0:200])
            plt.show()
        return ff_gains, TimeSequences
    
    
    def fast_fading_channel_coefficients(self):
        M, J, N, Ts, f_c  = self.M, self.J, self.N, self.Ts, self.f_c
        f_c = 1.3 #GHz
        v = 40 #kmph 
        v_ms = v*5/18
        c = 3*1e8
        tau = .01
        fd_max = v_ms*f_c*1e9/c
        TimeVaris = np.arange(0,5,0.0005)
        ff_gains, TimeSequences = self.return_jakes_coeffcients(fd_max, TimeVaris, n_links = M*(M-1)*J*N, plot = False)
        
        FastFadingChannels = np.random.normal(0,1/np.sqrt(2), M*J*N) + 1j*np.random.normal(0,1/np.sqrt(2), M*J*N)
        FastFadingChannels = np.reshape(FastFadingChannels,(M,J,N))
        FadingGains = np.abs(FastFadingChannels)**2
        alltime_fast_fading_gains = []
        for ts in range(Ts):
            all_fast_fading_gains = np.zeros((M,M*J,N))
            for m in range(M):
                jakes_coeffs = ff_gains[:,ts]
                jakes_coeffs = np.reshape(jakes_coeffs,(M,(M-1)*J,N))
                all_fast_fading_gains[m] = np.concatenate([FadingGains[m], jakes_coeffs[m]])
            all_fast_fading_gains = np.array(all_fast_fading_gains)
            alltime_fast_fading_gains.append(all_fast_fading_gains)
        alltime_fast_fading_gains = np.array(alltime_fast_fading_gains)
        return alltime_fast_fading_gains
    
    ######## functions to return large scale fading gains ##########
    #### compute intra and inter distances across all time slots #####
    
    def generate_random_point(self,grid_size):
        return np.random.uniform(0, grid_size, 2)

    def generate_random_velocity(self):
        angle = np.random.uniform(0, 2 * np.pi)
        velocity = 11.11  # 40 km/h in m/s
        return np.array([velocity * np.cos(angle), velocity * np.sin(angle)])

    def is_within_grid(self,point, grid_size):
        return all(0 <= coord <= grid_size for coord in point)

    def handle_boundary_collisions(self,point, grid_size):
        angle = np.random.uniform(0, 2 * np.pi)
        velocity = 11.11
        if point[0] <= 0 or point[0] >= grid_size:
            angle = np.pi - angle if point[0] <= 0 else -angle
        if point[1] <= 0 or point[1] >= grid_size:
            angle = -np.pi / 2 - angle if point[1] <= 0 else np.pi / 2 - angle
        return np.array([velocity * np.cos(angle), velocity * np.sin(angle)])

    def handle_ap_collisions(self,points, velocities, min_distance):
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                if np.linalg.norm(points[i] - points[j]) < min_distance:
                    # Adjust direction randomly for both APs
                    velocities[i] = self.generate_random_velocity()
                    velocities[j] = self.generate_random_velocity()
        return velocities

    def update_positions(self,points, velocities, tau, grid_size, min_distance):
        new_points = points + velocities * tau
        for i, point in enumerate(new_points):
            if not self.is_within_grid(point, grid_size):
                velocities[i] = self.handle_boundary_collisions(point, grid_size)
            new_points[i] = points[i] + velocities[i] * tau  # Recalculate with new velocity
        velocities = self.handle_ap_collisions(new_points, velocities, min_distance)
        return new_points, velocities
    
    
    def return_euclid_dist(self, device_x_coord, device_y_coord, AP_x_coord, AP_y_coord):
        device_coords = np.array([device_x_coord, device_y_coord])
        AP_coords = np.array([AP_x_coord, AP_y_coord])
        return np.linalg.norm(device_coords - AP_coords)
    
    def compute_TxRX(self,grid_size = 20):
        M, J, N, Ts  = self.M, self.J, self.N, self.Ts
        TxRxds = self.TxRxds
        #grid_size = 20
        num_points = self.M
        min_distance = 2
        tau = 0.01  # time interval in seconds

        # Initialize points and velocities
        points = np.array([self.generate_random_point(grid_size) for _ in range(num_points)])
        velocities = np.array([self.generate_random_velocity() for _ in range(num_points)])
        init_cents = points
        sub_net_cents = np.zeros((Ts+1, num_points, 2))
        sub_net_cents[0] = init_cents
        for step in tqdm(range(Ts-1)):
            points, velocities = self.update_positions(points, velocities, tau, grid_size, min_distance)
            sub_net_cents[step+1] = points
        
        d_relative_locs = {}
        for i in range(M):
            d_angle = np.random.uniform(0,2*np.pi,J) 
            d_r = np.random.uniform(0, 1, J)
            d_relative_locs[i] = np.vstack([d_r*np.cos(d_angle), d_r*np.sin(d_angle)])
            
        x_coords_ts, y_coords_ts = {}, {}
        for ts in range(Ts):
            coords = sub_net_cents[ts]
            point_xs, point_ys = [], []
            for k in d_relative_locs.keys():
                point_x = d_relative_locs[k][0] + coords[k][0]
                point_y = d_relative_locs[k][1] + coords[k][1]
                point_xs.append(point_x)
                point_ys.append(point_y)
            #break    
            x_coords_ts[ts] = point_xs
            y_coords_ts[ts] = point_ys       
            
        for ts in range(Ts):
            device_x_coords, device_y_coords = np.array(x_coords_ts[ts]), np.array(y_coords_ts[ts])
            device_x_coords, device_y_coords = device_x_coords.flatten(), device_y_coords.flatten()
            AP_x_coords, AP_y_coords = sub_net_cents[ts][:,0], sub_net_cents[ts][:,1]

            #for dx in device_x_coords 
            dists = np.zeros((M,M*J))
            for i in range(AP_x_coords.shape[0]):
                dist = []
                for j in range(len(device_x_coords.flatten())):
                    #dist = []
                    #print(i,j)
                    dist.append(self.return_euclid_dist(device_x_coords[j], device_y_coords[j], AP_x_coords[i], AP_y_coords[i]))
                    #print(dist)
                dists[i] = np.array(dist)
            TxRxds[ts] = dists
        return TxRxds
    
    def large_scale_fading_channel_coefficients(self, TxRxds):
        Ts, M, J, N = self.Ts, self.M, self.J, self.N
        f_c = self.f_c
        alltime_PathGains = []
        for ts in range(Ts):
            PL_los = 31.84 +21.5*np.log10(TxRxds[ts]) + 19*np.log10(f_c)
            PL = 33+25.5*np.log10(TxRxds[ts])+20*np.log10(f_c)
            PL_nlos = np.max((PL_los, PL), axis = 0)
            PathGains = np.sqrt(np.power(10, -PL_nlos/10))
            PathGains = np.repeat(PathGains[:, :, np.newaxis], N, axis=2)
            alltime_PathGains.append(PathGains)
        alltime_PathGains = np.array(alltime_PathGains)
        return alltime_PathGains
    
    
    def compute_SINRs_freq_reuse1(self, alltime_PathGains, alltime_fast_fading_gains):
        Ts, M, J, N = self.Ts, self.M, self.J, self.N
        gamma_0 = self.gamma_0
        all_SINRsdB = np.zeros((Ts, M,J,N))
        for ts in range(Ts):
            PathGainsTot = alltime_PathGains[ts,:,:,:]*alltime_fast_fading_gains[ts,:,:,:]

            #### Compute WantedSigPerDev ######
            WantedSigPerDev = np.zeros((M,J,N))
            for m in range(M):
                WantedSigPerDev[m] = PathGainsTot[m,m*J:(m+1)*J]

            InterfPowsPerDev = np.zeros((M,J,N))
            for m in range(M):
                Interferers = [i for i in range(M) if i!=m]
                Devs = np.arange(m*J,(m+1)*J)
                #print(Interferers, Devs)
                InterfPowGains = PathGainsTot[np.ix_(Interferers, Devs)]
                InterfPowsPerDev[m,] = np.sum(InterfPowGains, axis = 0)

            SINRs = WantedSigPerDev/(InterfPowsPerDev + 1/gamma_0)
            SINRsdB = 10*np.log10(SINRs)
            all_SINRsdB[ts,:] = SINRsdB            
        return all_SINRsdB
        
        
    ###### compute rewards ###############
    def compute_rewards(self, alltime_PathGains, alltime_fast_fading_gains, 
                        ts = 0, b = None, interfers_actions = None, b_actions = None):
        # interfers_actions shape (M-1, J,N), b_actions shape J x N
        M, J, N = self.M, self.J, self.N
        epsilon = 1e-15        
        SINR_combined_lin = []
        
        
        if ((b is None) and (interfers_actions is None) and (b_actions is None)):
                b = 0
                interfers_actions = np.zeros((M-1, J, N))
                b_actions = self._generate_matrix(N, J)
                for i, m in enumerate(range(M-1)):
                    interfers_actions[i,:,:] = np.transpose(self._generate_matrix(N, J))

        PathGainsTot = alltime_PathGains[ts,:,:,:]*alltime_fast_fading_gains[ts,:,:,:]
        
        channel_gain_interested_subnw = PathGainsTot[0,:,:]
        channel_gain_interested_subnw = channel_gain_interested_subnw.reshape(M,J,N)
        WantedSigPerDev = channel_gain_interested_subnw[0,:,:]
        interfers = np.array([i for i in range(M) if i!=b])
        InterfPowGains = channel_gain_interested_subnw[interfers,:,:]

        for i in range(InterfPowGains.shape[0]):
                InterfPowGains[i,:,:] = np.multiply(InterfPowGains[i,:,:], interfers_actions[i,:,:])
        InterfPowsPerDev = np.sum(InterfPowGains, axis = 0)
        #dimension of each SINR should be a matrix of N x J for an agent
        #print(np.max(WantedSigPerDev), np.max(InterfPowsPerDev))
        SINR =   np.transpose(WantedSigPerDev/(InterfPowsPerDev + 1/self.gamma_0))
        #SINR =   np.transpose(WantedSigPerDev/(InterfPowsPerDev))
        #inds = np.where(SINR == np.Inf)
        #SINR[inds] = 0
        SINR = np.multiply(SINR, b_actions)
        SINR_combined = np.max(SINR, axis = 0)
        SINR_combined_lin.extend(SINR_combined)
        k,n,P_o = 0.4,1, 1e-5
        V_gamma = 1 - (1/(1 + SINR_combined)**2)
        C_gamma = np.log2(1 + SINR_combined)
        Q_term = (2*n*C_gamma - 2*k + np.log2(n))/(2*np.sqrt(n)*np.sqrt(V_gamma))
        P_e = 0.5-0.5*special.erf(Q_term/np.sqrt(2))
        #reward = -np.exp((P_e/P_o) - 1)
        reward = -np.log10(np.abs(((P_e/P_o)+epsilon)))
        return SINR_combined_lin, P_e, reward
        
    ####### get next state ##############
    def get_next_state(self, alltime_PathGains, alltime_fast_fading_gains, 
                        ts = 1, b = None, interfers_actions = None, b_actions = None):
        ### ts should be next time-slot relative to compute rewards #########
        M, J, N = self.M, self.J, self.N        
        ###### if all are None, then just enerate random actions for intial time-slot and for testing ####
        if ((b is None) and (interfers_actions is None) and (b_actions is None)):
                b = 0
                interfers_actions = np.zeros((M-1, J, N))
                b_actions = self._generate_matrix(N, J)
                for i, m in enumerate(range(M-1)):
                    interfers_actions[i,:,:] = np.transpose(self._generate_matrix(N, J))
                    
        ###### the below code is the samething as in compute rewards method #########
        PathGainsTot = alltime_PathGains[ts,:,:,:]*alltime_fast_fading_gains[ts,:,:,:]
        
        channel_gain_interested_subnw = PathGainsTot[0,:,:]
        channel_gain_interested_subnw = channel_gain_interested_subnw.reshape(M,J,N)
        WantedSigPerDev = channel_gain_interested_subnw[0,:,:]
        interfers = np.array([i for i in range(M) if i!=b])
        InterfPowGains = channel_gain_interested_subnw[interfers,:,:]

        for i in range(InterfPowGains.shape[0]):
                InterfPowGains[i,:,:] = np.multiply(InterfPowGains[i,:,:], interfers_actions[i,:,:])
        InterfPowsPerDev = np.sum(InterfPowGains, axis = 0)
        
        #dimension of each SINR should be a matrix of N x J for an agent
        SINR =   np.transpose(WantedSigPerDev/(InterfPowsPerDev + 1/self.gamma_0))
        SINR = np.multiply(SINR, b_actions)
        SINR = np.multiply(SINR, b_actions)
        next_state = np.stack((SINR, b_actions), axis=0)
        return next_state

