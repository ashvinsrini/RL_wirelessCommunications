# -*- coding: utf-8 -*-
"""
Created on Sat May 20 13:58:17 2023

@author: sriniva3
"""
import matplotlib.pyplot as plt
import numpy as np
import pickle
import tarfile
# open a .spydata file
filename = 'LSTM_4_states.spydata'
tar = tarfile.open(filename, "r")
# extract all pickled files to the current working directory
tar.extractall()
extracted_files = tar.getnames()
for f in extracted_files:
    if f.endswith('.pickle'):
         with open(f, 'rb') as fdesc:
             data = pickle.loads(fdesc.read())
# or use the spyder function directly:
#from spyderlib.utils.iofuncs import load_dictionary
#data_dict = load_dictionary(filename)


plt.figure(1)
plt.plot(-1*np.array(data['all_acc_rewardsqos'])/256,label = 'qos')
plt.legend()
plt.show()
plt.figure(2)
plt.plot(-1*np.array(data['all_acc_rewardsbh'])/256,label = 'bh')
plt.legend()
plt.show()
plt.figure(3)
plt.plot(-1*np.array(data['all_acc_rewardsuep'])/256,label = 'uep')
plt.legend()
plt.show()