#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 10:59:32 2021

@author: MarkusRosenberger
"""

'''
In this script, several different models are trained 
to compare them and decide which one performs best
In the end, the history of all of them is plotted
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
#define some useful constants

# insert here number of nodes for different models
# trying to set up 6 models at a time
nodevec = [32, 64, 96, 128, 160, 192]

# how many layers you use
# 2 or 3 are possible
nLayers = 1

# for how many epochs you want to train it
nepochs = 50000





parameters_train = np.load(traindir + 'parameters_train.npy')
Temp_train = np.load(traindir + 'Temp_train.npy')


inputTrain = parameters_train


# use log of Temp, results are way better
outputTrain = Temp_train



# define needed parameters
nInput = np.shape(inputTrain[0])
nOutput = len(Temp_train[0])



modelvec = []

if nLayers == 2:
    for i in nodevec:
        modelvec.append(tf.keras.models.Sequential([
                tf.keras.layers.Flatten(input_shape=(nInput)),        
                tf.keras.layers.Dense(i, activation='relu'),
                tf.keras.layers.Dense(i, activation='relu'),
                tf.keras.layers.Dense(nOutput)
                ]))
    

elif nLayers == 3:
    for i in nodevec:
        modelvec.append(tf.keras.models.Sequential([
                tf.keras.layers.Flatten(input_shape=(nInput)),        
                tf.keras.layers.Dense(i, activation='relu'),
                tf.keras.layers.Dense(i, activation='relu'),
                tf.keras.layers.Dense(i, activation='relu'),
                tf.keras.layers.Dense(nOutput)
                ]))
 
# for the case that other numbers are used
else: 
    raise ValueError('Only 2 or 3 layers are allowed!‚')
    
    
# Loss function and optimizer are the same always
loss_fn = tf.keras.losses.MeanSquaredError()
opt = tf.keras.optimizers.Adam()

# compile the models
for i in range(len(modelvec)):
    modelvec[i].compile(optimizer=opt,loss=loss_fn, metrics=['MeanSquaredError'])
    
    
# save the history
training_history = []

for i in range(len(modelvec)):
    training_history.append(modelvec[i].fit(inputTrain, outputTrain, epochs=nepochs))



losshist = [i.history['loss'] for i in training_history]


# Don't need all of this following stuff if only run once

# as many entries as different number of nodes
#losshist = [ [], [], [], [], [], [] ]
# only first time
#losshist_rand = [i.history['loss'] for i in training_history_rand]
#losshist_few = training_history_few
    
# from second time on
#for i in range(len(training_history)):
    
    #losshist[i] = np.hstack((losshist[i], training_history[i].history['loss']))



# save the models and loss histories
i = 0
for model in modelvec:
    model.save(basedir + 'ML_models/3Parameter/Models/' + str(nodevec[i]) + 'nodes_' + str(nLayers) + 'layers')
    i+=1
    
    
i = 0
for hist in losshist:
    # neglect first entry of vector, bc is type history and stuff
    np.save(basedir + 'ML_models/3Parameter/losshist_' + str(nodevec[i]) + 'nodes_' + str(nLayers) + 'layers.npy', np.array(hist[1:]))
    i+=1


# load them again
#modelvec = []
#losshist = []
#for i in nodevec:
#    modelvec.append(tf.keras.models.load_model(basedir + 'ML_models/3Parameter/Models/' + str(nodevec[i]) + 'nodes'))
#    losshist.append(np.load(basedir + 'ML_models/3Parameter/losshist_' + str(nodevec[i]) + 'nodes.npy'))

#####################################################################################################

''' 
Plot loss history
'''
   
# start plot at 100th epoch because values are way to high before
x = np.arange(100, nepochs, 1)
for i in range(len(losshist)):
    plt.figure(figsize = (10,5))
    plt.suptitle(str(nodevec[i]) + ' Nodes per layer')
    
    plt.plot(x, losshist[i][100:], label = str(np.min(losshist[i][100:])))
    plt.xlabel('Epoch')
    plt.xticks([100, int(0.2*nepochs), int(0.4*nepochs), int(0.6*nepochs), int(0.8*nepochs), int(nepochs)])
    plt.yscale('log')
    plt.legend(loc = 'upper right')
    plt.savefig(basedir + 'ML_models/3Parameter/' + str(nepochs) + 'epochs_' + str(nodevec[i]) + 'nodes_' + str(nLayers) + 'layers.png', bbox_inches = 'tight')
    
    plt.show()

