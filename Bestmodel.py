#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 10:52:19 2021

@author: MarkusRosenberger
"""

'''
In this script, several different models are trained 
and compared after a given amount of epochs to 
see which set of values for layers, nodes per layer and epochs performs best
'''

import time
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker


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
#define some useful constants

# insert here number of nodes per layer for different models
# trying to set up several models at a time
nodevec = [50, 100, 150, 200, 250]


# as many empty vectors as len(nodevec)
losshist = [ [], [], [], [], [] ]

if not len(nodevec) == len(losshist):
    raise ValueError('One of the vectors is too short!')


# how many layers you use
# 1, 2, 3 or 4 are possible
nLayers = 1

# for how many epochs you want to train it
nepochs = 200000

# after how many epochs they are compared
deltaepoch = 10000
epochvec = np.arange(deltaepoch, nepochs+deltaepoch, deltaepoch)


# load training values
parameters_train = np.load(traindir + 'parameters_train.npy')
Temp_train = np.load(traindir + 'Temp_train.npy')


# load test values
parameters_test = np.load(testdir + 'parameters_test.npy')
Temp_test = np.load(testdir + 'Temp_test.npy')



inputTrain = parameters_train

# use log of Temp, results are way better
outputTrain = Temp_train



# define needed parameters
nInput = np.shape(inputTrain[0])
nOutput = len(Temp_train[0])



modelvec = []

if nLayers == 1:
    for i in nodevec:
        modelvec.append(tf.keras.models.Sequential([
                tf.keras.layers.Flatten(input_shape=(nInput)),        
                tf.keras.layers.Dense(i, activation='relu'),
                tf.keras.layers.Dense(nOutput)
                ]))


elif nLayers == 2:
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
    
elif nLayers == 4:
    for i in nodevec:
        modelvec.append(tf.keras.models.Sequential([
                tf.keras.layers.Flatten(input_shape=(nInput)),        
                tf.keras.layers.Dense(i, activation='relu'),
                tf.keras.layers.Dense(i, activation='relu'),
                tf.keras.layers.Dense(i, activation='relu'),
                tf.keras.layers.Dense(i, activation='relu'),
                tf.keras.layers.Dense(nOutput)
                ])) 
    
# for the case that other numbers are used
else: 
    raise ValueError('Only 1 - 4 layers are allowed!')
    
    
# Loss function and optimizer are the same always
loss_fn = tf.keras.losses.MeanSquaredError()
opt = tf.keras.optimizers.Adam()

# compile the models
for i in range(len(modelvec)):
    modelvec[i].compile(optimizer=opt,loss=loss_fn, metrics=['MeanSquaredError'])
    
    
# save the history
#training_history = []
trainminvec = np.ndarray((len(nodevec), 1, len(epochvec)))
wherevec = np.zeros_like(trainminvec)
testminvec = np.zeros_like(trainminvec)


for i in range(len(modelvec)):
    
    if not os.path.exists(outdir + '/' + str(nodevec[i]) + 'nodes_' + str(nLayers) + 'layers'):
        os.mkdir(outdir + '/' + str(nodevec[i]) + 'nodes_' + str(nLayers) + 'layers')


    for k in range(len(epochvec)):
        
        '''train the model'''
        training_history = modelvec[i].fit(inputTrain, outputTrain, epochs=deltaepoch)
        losshist[i] = np.hstack((losshist[i], training_history.history['loss']))
       
        trainminvec[i][0][k] = np.min(losshist[i])
        wherevec[i][0][k] = np.argmin(losshist[i])
       
        
        '''test the model'''
        test_loss, _ = modelvec[i].evaluate(parameters_test,  Temp_test, verbose=2)
        testminvec[i][0][k] = test_loss



        modelvec[i].save(outdir + '/' + str(nodevec[i]) + 'nodes_' + str(nLayers) + 'layers/' + str(epochvec[k]) + 'epochs')

    
    
    np.save(outdir + '/' + str(nodevec[i]) + 'nodes_' + str(nLayers) + 'layers/trainminvec_' + str(nepochs) + 'epochs.npy', trainminvec[i][0])
    np.save(outdir + '/' + str(nodevec[i]) + 'nodes_' + str(nLayers) + 'layers/wherevec_' + str(nepochs) + 'epochs.npy', wherevec[i][0])
    np.save(outdir + '/' + str(nodevec[i]) + 'nodes_' + str(nLayers) + 'layers/testminvec_' + str(nepochs) + 'epochs.npy', testminvec[i][0])
    np.save(outdir + '/' + str(nodevec[i]) + 'nodes_' + str(nLayers) + 'layers/losshist_' + str(nepochs) + 'epochs.npy', losshist[i])#, np.array(hist[1:]))

# save the models and loss histories
#i = 0
#for model in modelvec:
#    model.save(basedir + 'ML_models/3Parameter/Models/' + str(nodevec[i]) + 'nodes_' + str(nLayers) + 'layers')
#    i+=1
    
    
    
#i = 0
#for hist in losshist:
#    
#    # neglect first entry of vector, bc is type history and stuff
#    np.save(outdir + '/' + str(nodevec[i]) + 'nodes_' + str(nLayers) + 'layers/losshist_' + str(nepochs) + 'epochs.npy', hist)#, np.array(hist[1:]))
#    
#    i+=1


#for i in range(len(trainminvec)):
#    np.save(outdir + '/' + str(nodevec[i]) + 'nodes_' + str(nLayers) + 'layers/' + 'losshist_'  + str(nepochs) + 'epochs_' + str(nodevec[i]) + 'nodes_' + str(nLayers) + 'layers.npy', np.array(hist[1:]))

    
# load them again
#modelvec = []
#losshist = []
#for i in nodevec:
#    modelvec.append(tf.keras.models.load_model(basedir + 'ML_models/3Parameter/Models/' + str(nodevec[i]) + 'nodes'))
#    losshist.append(np.load(basedir + 'ML_models/3Parameter/losshist_' + str(nodevec[i]) + 'nodes.npy'))


trainminvec = np.ndarray((len(nodevec), 1, len(epochvec)))
wherevec = np.zeros_like(trainminvec)
testminvec = np.zeros_like(trainminvec)


for i in range(len(nodevec)):
    trainminvec[i] = np.load(outdir + '/' + str(nodevec[i]) + 'nodes_' + str(nLayers) + 'layers/' + 'trainminvec_'  + str(nepochs) + 'epochs.npy')
    wherevec[i] = np.load(outdir + '/' + str(nodevec[i]) + 'nodes_' + str(nLayers) + 'layers/' + 'wherevec_'  + str(nepochs) + 'epochs.npy')
    testminvec[i] = np.load(outdir + '/' + str(nodevec[i]) + 'nodes_' + str(nLayers) + 'layers/' + 'testminvec_'  + str(nepochs) + 'epochs.npy')
    
    

plt.figure(figsize= (10,15))
plt.subplots_adjust(hspace = 0.3)


plt.subplot(311)
plt.contourf(epochvec, nodevec, np.reshape(trainminvec, (len(trainminvec),20))*1e5, levels = [0, 0.1, 0.3, 0.5, 1, 3, 5], cmap = 'jet_r', extend = 'max')
plt.colorbar(label = 'Mean Squared Error')
plt.text(nepochs*1.05, nodevec[-1]*1.05, '1e-5', fontsize = 12)
plt.xlabel('Epochs')
plt.ylabel('Nodes per Layer')

plt.subplot(312)
plt.contourf(epochvec, nodevec, np.reshape(testminvec, (len(trainminvec),20))*1e5, levels = [0, 1, 2, 3, 6, 8, 10], cmap = 'jet_r', extend = 'max')
plt.colorbar(label = 'Mean Squared Error')
plt.text(nepochs*1.05, nodevec[-1]*1.05, '1e-5', fontsize = 12)
plt.xlabel('Epochs')
plt.ylabel('Nodes per Layer')

plt.subplot(313)
cs = plt.contourf(epochvec, nodevec, np.reshape(testminvec, (len(trainminvec),20))/np.reshape(trainminvec, (len(trainminvec),20)), levels = [0.5, 1, 3, 5, 10, 20, 30, 40, 50], cmap = 'jet_r', extend = 'both')
cs.cmap.set_under('k')
plt.colorbar(label = u'$ MSE_{test} / MSE_{train}$')
plt.xlabel('Epochs')
plt.ylabel('Nodes per Layer')

plt.savefig(outdir + 'Plots/' + str(nLayers) + 'layers_v01.png', bbox_inches = 'tight')



