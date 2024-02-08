import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pickle
import random
from scipy.special import jn
import pdb

from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from tqdm import tqdm

import glob
from natsort import natsorted
from PIL import Image


################## ALL functions ########################

def return_jakes_coeffcients(fd_max, TimeVaris, n_links = 5, plot = True):
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
    ff_gains = np.array(ff_gains)
    TimeSequences = np.array(TimeSequences)
    if plot:
        plt.plot(TimeVaris[0:200], 10*np.log10(ff_gains[0])[0:200])
        plt.show()
    return ff_gains, TimeSequences



def return_circlepositions(n, circle_radius, square_side, min_distance):
    # Initialize an empty list to store circle positions
    circle_positions = []

    # Function to check if a new circle position is valid
    def is_valid_position(new_position):
        for existing_position in circle_positions:
            distance = np.linalg.norm(new_position - existing_position)
            if distance < min_distance:
                return False
        return True

    # Generate positions for circles ensuring minimum distance
    while len(circle_positions) < n:
        new_position = np.random.rand(2) * (square_side - 2 * circle_radius) + circle_radius

        if is_valid_position(new_position):
            #pdb.set_trace()
            circle_positions.append(new_position)
            #circle = plt.Circle(new_position, circle_radius, fill=True, edgecolor='black', facecolor='blue', alpha=0.5)
    return circle_positions



##### generate sub networks in an indoor factory environment ###########
def ret_circle_positions(plot = True):
        #### plot square and sub networks #######
        plots = []
        # Create a square with the specified side length
        square = plt.Rectangle((0, 0), square_side, square_side, fill=None, edgecolor='black')

        # Create a subplot
        fig, ax = plt.subplots()

        # Add the square to the plot
        ax.add_patch(square)

        circle_positions = return_circlepositions(n, circle_radius, square_side, min_distance)

        angles = {}
        distances = {}

        all_x_cords, all_y_cords = {}, {}

        for i, pos in enumerate(circle_positions):
            circle = plt.Circle(pos, circle_radius, fill=True, edgecolor='black', facecolor='blue', alpha=0.5)
            ax.add_patch(circle)
            angs, dists = [], []
            x_coords, y_coords = [], []
            for _ in range(UEs_per_subnet):
                        angle = np.random.uniform(0, 2 * np.pi)
                        distance_from_center = np.random.uniform(0, circle_radius)
                        point_x = pos[0] + distance_from_center * np.cos(angle)
                        point_y = pos[1] + distance_from_center * np.sin(angle)
                        plt.scatter(point_x, point_y, color='red', marker='.')
                        plt.scatter(pos[0], pos[1], color='black', marker='.')
                        angs.append(angle)
                        dists.append(distance_from_center)
                        x_coords.append(point_x)
                        y_coords.append(point_y)

            x_coords.append(pos[0])
            y_coords.append(pos[1])

            angles[i] = angs
            distances[i] = dists
            all_x_cords[i] = x_coords
            all_y_cords[i] = y_coords

        # Set axis limits
        ax.set_xlim(0, square_side)
        ax.set_ylim(0, square_side)

        # Set aspect ratio to 'equal' to ensure a square plot
        ax.set_aspect('equal', adjustable='box')
        plots.append([ax])
        # Set plot title
        #plt.title(f'Square with {n} circles (time_instance={0} units)')
        plt.savefig('./imgs/1.png')
        # Show the plot
        if plot:
            plt.show()

        all_circle_positions[0] = circle_positions
        return plots, angles, distances, circle_positions
    
    
def return_next_circlepositions(n, circle_radius, square_side, min_distance, circle_positions):
    # Initialize an empty list to store next circle positions
    next_circle_positions = []

    # Function to check if a new circle position is valid
    def is_valid_position(new_position):
        for existing_position in next_circle_positions:
            distance = np.linalg.norm(new_position - existing_position)
            if distance < min_distance:
                return False
        return True

    # Generate positions for circles ensuring minimum distance
    cnt = 0
    while len(next_circle_positions) < n:
        #new_position = np.random.rand(2) * (square_side - 2 * circle_radius) + circle_radius
        #pdb.set_trace()
        new_position = circle_positions[cnt] + np.random.uniform(-0.5, 0.5, (1, 2))[0] 
        if is_valid_position(new_position):
            next_circle_positions.append(new_position)
            cnt+=1
    #pdb.set_trace()
    return next_circle_positions

def check_boundar_conds(n, next_circle_positions, circle_radius, square_side):
    #pdb.set_trace()
    new_circle_positions = []
    for j in range(n):
        cp = next_circle_positions[j]
        for i in range(2):
            while cp[i] - circle_radius < 0 or cp[i] + circle_radius > square_side:
                    if  cp[i] + circle_radius > square_side:
                        diff  = cp[i] + circle_radius -  square_side 
                        cp[i] = cp[i] - 1*diff
                    else:
                        diff = circle_radius - cp[i]
                        cp[i] = cp[i] + 1*diff
                    #pdb.set_trace()
                    #cp[i] = cp[i]-3*diff
        new_circle_positions.append(cp)
    #pdb.set_trace()
    return new_circle_positions

def return_channel_coefficients(t, fd_max, tau, all_x_cords, all_y_cords, UEs_per_subnet):
    
    ############## channel coffecients ####################

    ##### For a single sample ########
    '''
    arg = 2*np.pi*fd_max*tau
    rho = jn(0, arg)
    mu = np.sqrt(1-rho**2)

    #h_0 = np.random.normal(0,1,n*UEs_per_subnet*(n-1)*N)+1j*np.random.normal(0,1,n*UEs_per_subnet*(n-1)*N)
    rand = np.random.normal(0,1,n*UEs_per_subnet*(n-1)*N)+1j*np.random.normal(0,1,n*UEs_per_subnet*(n-1)*N)
    #h_0 = np.reshape(h_0,(n, UEs_per_subnet, n-1, N))
    rand = np.reshape(rand,(n, UEs_per_subnet, n-1, N))

    ####### For interference, Create jakes coefficient for n sub networks #######
    #h = np.zeros((n, UEs_per_subnet, n-1, N))

    h = rho*prev_h + mu*rand
    '''
    #h = ff_gains[:,t]
    #h = np.reshape(h,(n, UEs_per_subnet, n-1, N))


    ### interference with path loss #########
    L0 = 128.1
    alf = 3.75

    # compute distance of each device of each subnetwork w.r.t other APs #

    TxRxdistances = []
    for k in all_x_cords.keys():
        other_keys = [key for key in all_x_cords.keys() if key!=k]
        TxRx_dist = []
        for k1 in other_keys:
            for i in range(UEs_per_subnet):
                pos1 = np.hstack((all_x_cords[k][i],all_y_cords[k][i]))
                pos2 = np.hstack((all_x_cords[k1][UEs_per_subnet],all_y_cords[k1][UEs_per_subnet]))
                dist = np.linalg.norm(pos1 - pos2)
                TxRx_dist.append(dist)

        TxRxdistances.extend(TxRx_dist)

    TxRxdistances = np.tile(np.array(TxRxdistances),N)

    #### get path gains total #####
    #pathlossesdB = L0 + 10*alf*np.log10(TxRxdistances/10000)
    #PL_los = 31.84+21.50*np.log10(f_c)
    PL_loss = 31.84+21.50*np.log10(TxRxdistances)+ 19*np.log10(f_c*1e-9)
    PL = 33+25.5*np.log10(TxRxdistances)+ 20*np.log(f_c*1e-9)
    pathlossesdB = np.max((PL,PL_loss), axis = 0)
    pathGains = np.power(10, pathlossesdB/10)    
    
    
    

    #FadingGains = np.abs(h.flatten())**2
    #pdb.set_trace()
    #PathGainsTot = pathGains*FadingGains
    
    
    ### Within subnet static channels #####
    #h_static = np.random.normal(0,1,n*UEs_per_subnet*(n-1)*N)+1j*np.random.normal(0,1,n*UEs_per_subnet*(n-1)*N)
    #h_static_gain = np.abs(h_static)**2
    #h_static = np.reshape(h_static,(n, UEs_per_subnet, n-1, N))
    #SINRs = np.abs(h_static)**2/PathGainsTot
    #SINRsdB = 10*np.log10(SINRs)
    #plt.hist(SINRsdB)
    #plt.show()
    #pdb.set_trace()
    #return SINRs, SINRsdB, h, pathGains
    return pathGains


def ret_PathGains_all_time_inds(plots, angles, distances, circle_positions,time_ind = 2, plot = False):
    counter = 1
    all_pathGains = []
    for time_instance in tqdm(range(time_ind)):

        square = plt.Rectangle((0, 0), square_side, square_side, fill=None, edgecolor='black')

        # Create a subplot
        fig, ax = plt.subplots()

        # Add the square to the plot
        ax.add_patch(square)

        #circle_velocities = np.random.uniform(-10, 10, (n, 2))    

        next_circle_positions = return_next_circlepositions(n, circle_radius, square_side, min_distance, circle_positions)

        next_circle_positions = check_boundar_conds(n, next_circle_positions, circle_radius, square_side)

        all_x_cords, all_y_cords = {}, {}
        for i, pos in enumerate(next_circle_positions):
            circle = plt.Circle(pos, circle_radius, fill=True, edgecolor='black', facecolor='blue', alpha=0.5)
            ax.add_patch(circle)
            x_coords, y_coords = [], []
            for j, _ in enumerate(range(UEs_per_subnet)):
                        point_x = pos[0] + distances[i][j] * np.cos(angles[i][j])
                        point_y = pos[1] + distances[i][j] * np.sin(angles[i][j])
                        plt.scatter(point_x, point_y, color='red', marker='.')
                        plt.scatter(pos[0], pos[1], color='black', marker='.')
                        x_coords.append(point_x)
                        y_coords.append(point_y)

            x_coords.append(pos[0])
            y_coords.append(pos[1])

            all_x_cords[i] = x_coords
            all_y_cords[i] = y_coords


        # Set axis limits
        ax.set_xlim(0, square_side)
        ax.set_ylim(0, square_side)

        # Set aspect ratio to 'equal' to ensure a square plot
        ax.set_aspect('equal', adjustable='box')
        plots.append([ax])
        # Set plot title
        plt.title(f'Square with {n} circles (time_instance={counter} units)')
        plt.savefig('./imgs/{}.png'.format(counter+1))
        #plt.close()
        # Show the plot
        if not plot:
            plt.close()
        else:        
            plt.show()
        #all_circle_positions.append(next_circle_positions)
        all_circle_positions[counter] = next_circle_positions
        circle_positions = next_circle_positions 
        counter+=1

        pathGains = return_channel_coefficients(time_instance, fd_max, tau, all_x_cords, all_y_cords, UEs_per_subnet)
        all_pathGains.append(pathGains)
        #plt.figure(2)
        #plt.hist(SINRsdB, bins = 100)
    all_pathGains = np.array(all_pathGains)    
    return all_pathGains




def create_gif(time_ind):
    files = glob.glob(r"./imgs/*.png")
    files = natsorted(files)
    image_array = []
    
    def update(i):
        im.set_array(image_array[i])
        return im, 
    
    for my_file in files:

        image = Image.open(my_file)
        image_array.append(image)
    
    # Create the figure and axes objects
    fig, ax = plt.subplots()

    # Set the initial image
    im = ax.imshow(image_array[0], animated=True)
    
    animation_fig = animation.FuncAnimation(fig, update, frames=len(image_array), interval=100, blit=True,repeat_delay=10,)

    # Show the animation
    #plt.show()

    animation_fig.save("./imgs/animated_{}.gif".format(time_ind))
    
    pass
    
if __name__ == "__main__":
    ###### Intialization to save path gains and fading coefficeints for MADRL training #######
    GIF_flag = True
    time_ind = 20
    N = 30 # Number of channel resources
    UEs_per_subnet = 4
    Tot_area = 20
    #Nsub_nets = 8
    N_dev_subnet = 10
    v = 40 #kmph 
    v_ms = v*5/18
    f_c = 1.3*1e9
    c = 3*1e8
    tau = .001
    fd_max = v_ms*f_c/c
    all_circle_positions = {}
    #ff_gains= np.load('ff_gains.npy')



    # Define the side length of the square
    square_side = 20

    min_distance = 4

    # Set the number of circles(sub networks)
    n = 2

    # Calculate the radius of the circles based on the given area
    #circle_radius = np.sqrt(2 / np.pi)
    circle_radius = 1

    TimeVaris = np.arange(0,5,0.0005)
   
    
    ff_gains, TimeSequences = return_jakes_coeffcients(fd_max, TimeVaris, n_links = 240, plot = True)
    plots, angles, distances, circle_positions = ret_circle_positions()
    all_pathGains = ret_PathGains_all_time_inds(plots, angles, distances, circle_positions, time_ind = time_ind)
    np.save('TimeSequences.npy',TimeSequences)
    np.save('all_pathGains.npy',all_pathGains)
    if GIF_flag:
        create_gif(time_ind = time_ind)
    
    