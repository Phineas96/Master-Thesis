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
outdir = basedir + 'ML_models/3Parameter/Output/Bestmodel/'


############################################################################################

# insert, which numbers of nodes and layers should be loaded to test them 
# in 3 vectors with same length

nodevec = [100, 200, 250]
layervec = [3, 3, 4]
epochvec = [49064, 9789, 22878]


# how many different sets of values you want to test
n = 100


############################################################################################

# load test values
parameters_test = np.load(testdir + 'parameters_test.npy')

Temp_test = np.load(testdir + 'Temp_test.npy')



modelvec = []
labelvec = []

# empty vector for all the RMSE values
diffvec = []

# load all the models
for i in range(len(nodevec)):
    #modelvec.append(tf.keras.models.load_model(basedir + 'ML_models/3Parameter/Models/' + str(nodevec[i]) + 'nodes_' + str(layervec[i]) + 'layers'))
    modelvec.append(tf.keras.models.load_model(outdir + str(nodevec[i]) + 'nodes_' + str(layervec[i]) + 'layers/' + str(epochvec[i]) + 'epochs'))
    
    labelvec.append(str(nodevec[i]) + ' nodes, ' + str(layervec[i]) + ' layers, ' + str(epochvec[i]) + ' epochs')
    
    diffvec.append([])
        

# empty vector for the used indices of the test data
index = []

for i in range(n):
    # random int between 0 and len(Temp_test)
    ind = np.random.randint(0, len(Temp_test))
    while ind in index:
        ind = np.random.randint(0, len(Temp_test))
    index.append(ind)

    # vertical axis as in Kompot Code
    z = np.arange(43.25, 506.75+4.5, 4.5)
    
    # get reference profile from test data
    Tref = Temp_test[ind]
    Tref = 10**Tref
    
    # start with plot of reference profile
    plt.figure(figsize = (10,5))
    
    
    count = 0
    for model in modelvec:
        Tmod = model.predict(parameters_test[ind])
        Tmod = 10**Tmod[0]
    
        diffvec[count].append(np.sqrt(sum((Tmod-Tref)**2))/len(Tmod))
    
        plt.plot(Tmod, z, label = labelvec[count])
        
        count += 1
        
    plt.plot(Tref, z, label = 'Reference profile', color = 'k', linewidth = 3, ls ='--')

    
    # parameters as title in plot
    # order is [Mpl, fC, FxuvIn]
    tit = ["{:.3f}".format(parameters_test[ind][0][0]), "{:.3e}".format(parameters_test[ind][0][1]), "{:.3f}".format(parameters_test[ind][0][2])]
    
    plt.xlabel('log T [K]')
    plt.ylabel('z [km]')
    plt.grid()
    plt.legend()
    plt.title(str(tit).replace("'", ""))# + str(round(diff[-1], 2)))
    
    #plt.savefig(basedir + 'ML_models/3Parameter/Output/Testplots/test' + str(ind) + '.png', bbox_inches = 'tight')
    plt.savefig(outdir + 'Comparison/Version03/test' + str(ind) + '.png', bbox_inches = 'tight')
    plt.close()
        
    

#np.save(basedir + 'ML_models/3Parameter/Output/Testplots/' + str(n) + 'Differences.npy', diffvec)
#np.save(basedir + 'ML_models/3Parameter/Output/Testplots/' + str(n) + 'Parameters.npy', labelvec)
#
#np.save(basedir + 'ML_models/3Parameter/Output/Testplots/' + str(n) + 'Indices.npy', index)
np.save(outdir + 'Comparison/Version03/' + str(n) + 'Differences.npy', diffvec)
np.save(outdir + 'Comparison/Version03/' + str(n) + 'Parameters.npy', labelvec)

np.save(outdir + 'Comparison/Version03/' + str(n) + 'Indices.npy', index)


'''
Plot differences
'''

# possible markers:
# octagon, square, pentagon, plus, star, thin diamond, hexagon, filled x
markers = ['8', 's', 'p', '+', '*', 'd', 'h', 'X']

plt.figure(figsize = (13,6))
for i in range(len(diffvec)):
    p = plt.scatter(index, diffvec[i], label = labelvec[i], marker = markers[i], alpha = 0.5)
    

    plt.axhline(y = np.mean(diffvec[i]), color = p.get_facecolor()[0], ls = '--')


plt.xlabel('Index of Test profile')
plt.ylabel('RMS difference')
plt.title('Comparison of different models')
plt.legend()
#plt.savefig(basedir + 'ML_models/3Parameter/Output/Testplots/Differences.png', bbox_inches = 'tight')
plt.savefig(outdir + 'Comparison/Version03/Differences.png', bbox_inches = 'tight')

