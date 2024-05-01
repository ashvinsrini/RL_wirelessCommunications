import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import numpy as np
import matplotlib.pyplot as plt
import pickle
import random
from scipy.special import jn
import pdb
import json
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from tqdm import tqdm
from itertools import combinations

'''

a. The first step is to select resources based on having maximum values in Uhat and such that all users are assigned atleast one resource

b. Remaining unused resources are assigned as per having the decision closest to one in AllRowAlterns, matrix containing U+1 scheduling decisions (give it to no-one, or give it to one of the U users).

c. You get one channel allocation matrix by performing a. and b. 

d. Repeat steps a, b, c for K unique channel allocation matrices

''';


Nn = 8
Dd = 3
K = 8
with open('Uhat.json', 'r') as file:
    Uhat = json.load(file)
Uhat = np.array(Uhat)


Zers = np.zeros((Dd))
tmp = np.eye(Dd)
AllRowAlterns = np.vstack((Zers, tmp))


Scheduled = np.zeros((Nn, Dd))
Usched = (1/2)*np.ones((Nn, Dd))
Ones = np.ones((1,Dd))
inds = [np.where(-Uhat == k) for k in np.sort(-Uhat.flatten())]


########## PreferencesToGet has the index of highest to lowest values of Uhat (decreasing order) ################
PreferencesToGet = [(inds[i][0][0], inds[i][1][0]) for i in range(len(inds))]


Prefs = [P[::-1] for P in PreferencesToGet]
# Initialize : construct subsets of 1, ... Dd + 1 possible decisions
T = Dd + 1
alts = Prefs[0:T]

SubsetsOfUpToD = []
for leng in range(Dd):
    subs = [s for s in combinations(np.arange(T), leng+1)]
    reasubs = []
    for s in subs: 
        alto = np.array(np.array(alts)[np.array(s)])
        resos = alto[:,1]
        devs = alto[:,0]
        #print(alto, resos, devs, len(resos), len(devs))
        if (len(set(resos)) == leng+1 and len(set(devs)) == leng+1):
            #pdb.set_trace()
            reasubs.append(s)
    #print(reasubs)
    SubsetsOfUpToD.append(reasubs)
    
'''
alts = [(0, 5), (0, 7), (0, 6), (1, 6)]

SubsetsOfUpToD has indices corresponding to those sets in alts such that combination of tuples in alts is unique.
For e.g., for tuple length 1, all are included since all the rows and columns for each tuple will be unique, i.e., 
(0,5) is repeated only once and so on, so reasubs will be 0,1,2,3
for tuple size length 2 only, [(0,5),(1,6)] and [(0,7),(1,6)] are unique and thus reasubs will have indices [(0,3),(1,3)] 
respectively and so on for higher tuple lengths as well.
''';
print(SubsetsOfUpToD)

############ extend this greedily until you get K alternatives ############
'''
Decsisions are based after SubsOfInterest is not an empty set
1) The variable RowsInSchedule contains the channel scheduling strategy. 
    1A) The Decisions variable has the corresponding subsets from alts variable whose index is pointed in SubsOfInterest
    1B) Loop over each combination of Decisions represented by decisions 
        1Ba) First those resos RowsInSchedule[d[1]] are assigned a value of d[0] +1
        1Bb) Then the list of non scheduled resources are found out 
        1Bc) Finally for each reso in non scheduled resources is assigned that index of that closest vector in AllRowAlterns when 
        compared to Uhat[reso]
2) This is appended to the FullSchedsOfInterest list 
3) Then only the unique entries of FullSchedsOfInterest is computed in DiffScheds
4) This procedure is repeated until this condition 'len(SubsetsOfUpToD[Dd-1]) < K and T < Nn*Dd' is True.
''';


while (len(SubsetsOfUpToD[Dd-1]) < K and T < Nn*Dd):
    print(T)
    T = T + 1
    alts = Prefs[0:T]
    tmp = []
    for leng in np.arange(Dd - 1) + 1:
        #print(leng)
        OneSmaller = SubsetsOfUpToD[leng-1]
        subs = [t + (T-1,) for t in OneSmaller]
        reasubs = []
        for s in subs: 
            alto = np.array(np.array(alts)[np.array(s)])
            resos = alto[:,1]
            devs = alto[:,0]
            #print(alto, resos, devs, len(resos), len(devs))
            if (len(set(resos)) == leng+1 and len(set(devs)) == leng+1):
                #pdb.set_trace()
                reasubs.append(s)
        #print( SubsetsOfUpToD[leng], reasubs)
        #lst = 
        tmp.append(SubsetsOfUpToD[leng]+reasubs)
        #break
    SubsetsOfUpToD = []
    SubsetsOfUpToD.append([s for s in combinations(np.arange(T), 1)])
    SubsetsOfUpToD.extend(tmp)

    SubsOfInterest = SubsetsOfUpToD[Dd-1]
    if len(SubsOfInterest )>0:
        #print(T)
        #updated_nested_lists1 = [[tuple(element + 1 for element in tup) for tup in sublist] for sublist in SubsetsOfUpToD]
        #updated_list2 = [tuple(element + 1 for element in tup) for tup in SubsOfInterest]
        #print(updated_nested_lists1, updated_list2)
        ########## Recreating ExpandToFullSchedule module #########
        Decisions = [np.array(Prefs)[np.array(s)] for s in SubsOfInterest]
        FullSchedsOfInterest = []
        for decisions in Decisions:
            #print(decisions)
            RowsInSchedule = np.zeros((Nn))
            for d in decisions:
                RowsInSchedule[d[1]] = d[0]+1
            lst = decisions[:,1]
            NonScheduledResos = np.array([l for l in range(Nn) if l not in lst ])
            #print(RowsInSchedule, NonScheduledResos+1)
            for reso in NonScheduledResos:
                uvec = Uhat[reso]
                dists = np.array([np.linalg.norm(uvec-AllRowAlterns[i]) for i in range(len(AllRowAlterns))])
                mini = np.min(dists)
                posi = np.where(dists == mini)[0][0]
                RowsInSchedule[reso] = posi
            #print(RowsInSchedule)
            FullSchedsOfInterest.append([(int(r), i+1) for r,i in zip(RowsInSchedule, range(Nn))])
        DiffScheds = set(tuple(x) for x in FullSchedsOfInterest)
        DiffScheds = [list(x) for x in DiffScheds]

        KeepThese = []
        for unique_list in DiffScheds:
                for index, original_list in enumerate(FullSchedsOfInterest):
                    if unique_list == original_list:
                        KeepThese.append(index)
                        break  # Stop after the first match is found
        KeepThese = np.sort(np.array(KeepThese))
        FullSchedsOfInterest = DiffScheds
        SubsOfInterest = np.array(SubsOfInterest)[KeepThese]
        SubsOfInterest = [tuple(S) for S in SubsOfInterest]
        SubsetsOfUpToD[Dd-1] = SubsOfInterest
        #print(DiffScheds)

AlternSchedules = np.zeros((len(DiffScheds), Nn, Dd))
#sched = DiffScheds[0]
for j, sched in enumerate(DiffScheds):
    Usched = 0.5*np.ones((Nn, Dd))
    for i, tup in enumerate(sched):
        if tup[0] == 0:
            Usched[i] = np.zeros(Dd)
        else:
            Usched[i,tup[0]-1] = 1
    inds = np.where(Usched == 0.5)
    Usched[inds] = 0
    AlternSchedules[j] = Usched
AlternSchedules
