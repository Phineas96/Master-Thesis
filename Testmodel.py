#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 12:10:05 2021

@author: MarkusRosenberger
"""

'''
In this script, an already trained model can be loaded 
to test its performance on already created test data
'''

import time
import os
import numpy as np
import matplotlib.pyplot as plt
import kompot as kp
import kvass as kv

# TensorFlow and tf.keras
import tensorflow as tf

#############################################################################################

# define directories
basedir = '/Users/MarkusRosenberger/Documents/Uni Wien/MasterAstro/Machine Learning/Earth/'
traindir = basedir + 'ML_models/3Parameter/Trainingdata/'
testdir = basedir + 'ML_models/3Parameter/Testdata/'

############################################################################################

# insert, which numbers of nodes and layers should be loaded to test them
nodevec = [160]
layervec = [3]


# how many different sets of values you want to test
n = 75


############################################################################################

# load test values
parameters_test = np.load(testdir + 'parameters_test.npy')

Temp_test = np.load(testdir + 'Temp_test.npy')



modelvec = []
labelvec = []

# load all the models
for nodes in nodevec:
    for layers in layervec:
        modelvec.append(tf.keras.models.load_model(basedir + 'ML_models/3Parameter/Models/' + str(nodes) + 'nodes_' + str(layers) + 'layers'))
        
        labelvec.append(str(nodes) + ' nodes, ' + str(layers) + ' layers')
        
        
        
# empty vector for all the RMSE values
diffvec = []

# empty vector for the used indices of the test data
index = []

for i in range(n):
    # random int between 0 and len(Temp_test)
    ind = np.random.randint(0, len(Temp_test))
    index.append(ind)

    # vertical axis as in Kompot Code
    z = np.arange(43.25, 506.75+4.5, 4.5)
    
    # get reference profile from test data
    Tref = Temp_test[ind]
    Tref = 10**Tref
    
    # start with plot of reference profile
    plt.figure(figsize = (10,5))
    plt.plot(Tref, z/1e5, label = 'Reference profile')
    
    diff = []
    labcount = 0
    for model in modelvec:
        Tmod = model.predict(parameters_test[ind])
        Tmod = 10**Tmod[0]
    
        diff.append(np.sqrt(sum((Tmod-Tref)**2))/len(Tmod))
    
        plt.plot(Tmod, z/1e5, label = labelvec[labcount])

    
    # parameters as title in plot
    tit = ["{:.3f}".format(parameters_test[ind][0][0]), "{:.3e}".format(parameters_test[ind][0][1]), "{:.3f}".format(parameters_test[ind][0][2])]
    
    plt.xlabel('log T [K]')
    plt.ylabel('z [km]')
    plt.grid()
    plt.legend()
    plt.title(str(tit).replace("'", "") + '\n')# + str(round(diff[-1], 2)))
    
    plt.savefig(basedir + 'ML_models/3Parameter/test' + str(ind) + '.png', bbox_inches = 'tight')
    plt.close()
    
    diffvec.append(diff)

np.save(basedir + 'ML_models/3Parameter/' + str(n) + 'Differences.npy', diffvec)
np.save(basedir + 'ML_models/3Parameter/Parameters.npy', labelvec)

np.save(basedir + 'ML_models/3Parameter/Indices.npy', index)




















