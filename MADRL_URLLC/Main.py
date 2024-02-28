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
from Env import *




def ret_matrix(N,U):
    #N = 30  # Number of rows
    #U = 4  # Number of columns
    num_samples = np.power(10,2)  # Number of samples

    # Initialize a list to store the matrices
    matrices = []

    for _ in tqdm(range(num_samples)):
        # Initialize a zero matrix
        matrix = np.zeros((N, U))

        for i in range(N):
            # Randomly select a column to set to 1 for each row
            one_col = np.random.randint(0, U)
            matrix[i, one_col] = 1

        matrices.append(matrix)

    # Convert to numpy array for easier manipulation
    matrices = np.array(matrices)
    return matrices



if __name__ == "__main__":
    
    plot = True
    B = 2
    serve_BS_ind = 0
    N, U = 30, 4
    # Hyperparameters
    BUFFER_SIZE = 10000
    BATCH_SIZE = 64
    GAMMA = 0.98
    TAU = 0.005
    LR_ACTOR = 0.001
    LR_CRITIC = 0.001
    #N_ACTIONS = 51  # Number of discrete actions
    K = 1  # Number of neighbors for KNN
    num_episodes, time_steps = 100, 512
    p_0 = 1e-5
    lamda_1 = 1000
    
    ######## Load the path gains and fading gains, if not found then generate by Non stationary channel script ###########
    TimeSequences= np.load('TimeSequences.npy')
    all_pathGains = np.load('all_pathGains.npy')
    all_pathGains = np.transpose(all_pathGains)
    print(TimeSequences.shape, all_pathGains.shape)
    
    
    
    all_wtd_rewards = []
    ############# place holder to allocate channels, this is given by the action vector  ############# 
    all_A = []
    for _ in range(B):
            array = np.zeros((N, U), dtype=int)

            # For each row, place a single 1 in a random column
            for i in range(N):
                array[i, np.random.randint(U)] = 1

            all_A.append(array)
    all_A = np.array(all_A)

    ##################################################################################################

    #all_h = np.random.normal(loc=0, scale=1, size=(B,N,U))+ 1j*np.random.normal(loc=0, scale=1, size=(B,N,U))
    #all_h = all_h/np.sqrt(2)
    time_ind = B
    all_h = TimeSequences[:,time_ind]
    all_h = np.reshape(all_h, (B,N,U))


    prev_h = all_h[serve_BS_ind]
    #prev_h_1 = np.random.normal(loc=0, scale=1, size=(N,U))+ 1j*np.random.normal(loc=0, scale=1, size=(N,U))
    prev_h_1 = TimeSequences[:,0][0:N*U]
    prev_h_1 = np.reshape(prev_h_1,(N,U))


    num_samples = 10000  # Number of samples
    # Generate complex Gaussian samples
    real_part = np.random.normal(loc=0, scale=1, size=num_samples)
    imaginary_part = np.random.normal(loc=0, scale=1, size=num_samples)
    complex_samples = real_part + 1j * imaginary_part
    #rnd_samples = np.random.choice(complex_samples, size = (N,U))
    '''
    v = np.random.normal(10,1,U) # randomly drawn from a Gaussian with mean 10m/s
    fc = 1.3e9 # 1.3GHz
    tau = 0.5e-3 # time delay
    c = 3e8 # Veloctiy of light
    f_Dmax = 2*v*fc/c 
    '''

    environ = env(N,U)
    #h = environ.get_jakes_coeffcients(complex_samples, prev_h, corr_0, mu_0)
    h = environ.get_jakes_coeffcients_actual(TimeSequences, all_pathGains, time_ind = 2)
    SNR_init = environ.get_state(h, all_h, all_A, serve_BS_ind = 0)
    #h_1 = environ.get_jakes_coeffcients(complex_samples, prev_h_1, corr_0, mu_0)
    h_1 = environ.get_jakes_coeffcients_actual(TimeSequences,  all_pathGains, time_ind = 1)
    SNR_init_1 = environ.get_state(h_1, all_h, all_A, serve_BS_ind = 0)
    #weighted_reward, r_ber, r_res, reward_res = environ.get_rewards(SNR_init)
    #print(weighted_reward)
    #all_wtd_rewards.extend([weighted_reward])    
    
    
    
    
    ########### Actor and Critic networks instantiation #######################
    state_dim, action_dim = 2*2*np.prod(SNR_init.shape), np.prod(SNR_init.shape) 
    environ = env(N,U)
    #matrices = np.load('KNN_matrices.npy')
    matrices = ret_matrix(N,U)
    action_space = np.array([np.reshape(matrices[i], (N*U)) for i in range(len(matrices))])
    #knn = joblib.load('knn.pkl' , mmap_mode ='r')
    knn = KDTree(action_space)
    # Create actor and critic networks
    actor = Actor(state_dim, action_dim)
    critic = Critic(state_dim, action_dim)
    actor_target = Actor(state_dim, action_dim)
    critic_target = Critic(state_dim, action_dim)

    # Copy initial weights to target networks
    actor_target.load_state_dict(actor.state_dict())
    critic_target.load_state_dict(critic.state_dict())

    # Initialize optimizers
    optimizer_actor = optim.Adam(actor.parameters(), lr=LR_ACTOR)
    optimizer_critic = optim.Adam(critic.parameters(), lr=LR_CRITIC)

    # Create action space
    #action_space = np.linspace(-2, 2, N_ACTIONS).reshape(-1, 1)

    # Replay Buffer
    replay_buffer = deque(maxlen=BUFFER_SIZE)
    

    
    ##### Start training process #######



    ########### for each epoch ############
    SNR_init = np.reshape(SNR_init,((np.prod(SNR_init.shape),1)))
    SNR_init = np.array([SNR_init[i][0] for i in range(len(SNR_init))])

    SNR_init_1 = np.reshape(SNR_init_1,((np.prod(SNR_init_1.shape),1)))
    SNR_init_1 = np.array([SNR_init_1[i][0] for i in range(len(SNR_init_1))])

    prev_action = matrices[1,:,:] #Intializing previous action for time-slot 0  with random channel allocation matrix
    prev_action = prev_action.flatten()

    prev_action_1 = matrices[2,:,:] #Intializing previous action for time-slot 0  with random channel allocation matrix
    prev_action_1 = prev_action_1.flatten()


    epi_rewards, epi_a_losses, epi_c_losses = [], [], []
    for epi in range(num_episodes):

        SNR = SNR_init
        total_reward = 0
        #state = SNR
        if epi == 0:
            state = np.concatenate((SNR, prev_action, SNR_init_1, prev_action_1))
        else:
            state = np.concatenate((SNR_next.flatten(), action.flatten(), SNR.flatten(), prev_action.flatten()))


        for t in tqdm(range(time_steps)):
            #print(t)
            #pdb.set_trace()
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            #pdb.set_trace()
            continuous_action = actor(state_tensor)
            continuous_action = continuous_action + torch.normal(0, 0.1, size=continuous_action.shape) # Adding noise for exploration
            continuous_action = continuous_action.detach().numpy().squeeze()
            #knn = KDTree(action_space)
            distances, indices = knn.query(continuous_action.reshape(1, -1), k=K)
            #action_idx = np.random.choice(indices.squeeze())
            #action = action_space[action_idx]  

            #'''
            if len(indices[0])>1: 
                indices = indices.squeeze()
            else:
                indices = indices[0]
            # Evaluate each of the K actions using the critic network
            state_tensor_repeated = state_tensor.repeat(len(indices), 1)
            action_candidates = torch.FloatTensor(action_space[indices])
            q_values = critic(state_tensor_repeated, action_candidates).detach().numpy().squeeze()
            #pdb.set_trace()
            #Select the action that maximizes the Q-value
            best_action_idx = indices[np.argmax(q_values)]
            action = action_space[best_action_idx]    
            #''';
            #print(best_action_idx)
            action = np.reshape(action, (N,U))
            all_A[0] = action

            # assign random actions for other BSs for every t and update ch. coffeiceints of other BSs.
            #all_A[1:] =  [matrices[int(np.random.choice(np.arange(1,10e3,1)))]  for _ in range(1)] 
            all_A[1:] =  [matrices[10]  for _ in range(1)] 
            #all_h[1:] = [environ.get_jakes_coeffcients(complex_samples, all_h[cnt], corr_0, mu_0) for cnt in range(len(all_h)) if cnt!= 0 ]

            all_h[1:] = [environ.get_jakes_coeffcients_actual(TimeSequences,  all_pathGains, time_ind = 1,BS_ind = b) for b in range(1,2)]

            #h = environ.get_jakes_coeffcients(complex_samples, all_h[0], corr_0, mu_0)
            #time_ind = np.mod(t+2,9999)
            time_ind = 3
            h = environ.get_jakes_coeffcients_actual(TimeSequences,  all_pathGains, time_ind = time_ind)


            SNR_next = environ.get_state(h, all_h, all_A, serve_BS_ind = 0) #getting next state since all_A is updated with q values
            #pdb.set_trace()
            weighted_reward, _ , _, reward_res = environ.get_rewards(SNR, action)
            total_reward += weighted_reward

            # Store transition
            #state_v, action_v, next_state_v =  np.reshape(SNR,(N*U)), np.reshape(action,(N*U)), np.concatenate((SNR_next.flatten(), action.flatten()))
            #next_state_v = torch.FloatTensor(np.concatenate((SNR_next.flatten(), action.flatten())))
            state_v, action_v =  state, np.reshape(action,(N*U))
            next_state_v = np.concatenate((SNR_next.flatten(), action.flatten(), SNR.flatten(), prev_action.flatten()))
            SNR = SNR_next
            prev_action = action
            #pdb.set_trace()    
            replay_buffer.append((state_v, action_v, weighted_reward, next_state_v)) 

            # Sample mini-batch
            if len(replay_buffer) >= BATCH_SIZE:
                    #pdb.set_trace()
                    mini_batch =  random.sample(replay_buffer, BATCH_SIZE)
                    states, actions, rewards, next_states = zip(*mini_batch)

                    states = torch.FloatTensor(states)
                    actions = torch.FloatTensor(actions)
                    rewards = torch.FloatTensor(rewards).unsqueeze(1)
                    next_states = torch.FloatTensor(next_states)
                    #dones = torch.FloatTensor(dones).unsqueeze(1)

                    # Compute targets
                    with torch.no_grad():
                        next_actions = actor_target(next_states)
                        target_values = critic_target(next_states, next_actions)
                        if t<time_steps:
                            targets = rewards +  GAMMA * target_values
                        else:
                            targets = rewards

                    # Update Critic
                    #pdb.set_trace()
                    values = critic(states, actions)
                    loss_critic = nn.MSELoss()(values, targets)
                    optimizer_critic.zero_grad()
                    loss_critic.backward()
                    optimizer_critic.step()

                    # Update Actor
                    loss_actor = -critic(states, actor(states)).mean()
                    optimizer_actor.zero_grad()
                    loss_actor.backward()
                    optimizer_actor.step()

                    # Soft update target networks
                    for param, target_param in zip(critic.parameters(), critic_target.parameters()):
                        target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
                    for param, target_param in zip(actor.parameters(), actor_target.parameters()):
                        target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

            state = next_state_v
            #pdb.set_trace()   

        epi_rewards.append(total_reward)
        epi_a_losses.append(loss_actor.detach().numpy())
        epi_c_losses.append(loss_critic.detach().numpy())
        print('epoch',epi,'rewards', total_reward/t, 'actor_loss', loss_actor.detach().numpy(), 'cricitc_loss', loss_critic.detach().numpy())
            #pass
    epi_rewards = np.array(epi_rewards)/time_steps
    plt.figure(1)
    plt.plot(epi_rewards)
    plt.grid()
    plt.title('rewards vs epochs')
    plt.savefig('./imgs/rewards.png', dpi = 300)
    
    
    plt.figure(2)
    p_e = p_0*(np.log(-epi_rewards/(lamda_1*U)) + 1)
    plt.semilogy(np.array(p_e))
    plt.grid()
    plt.title('BLER vs epochs')
    plt.savefig('./imgs/bler.png', dpi = 300)
