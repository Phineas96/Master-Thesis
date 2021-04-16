#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 11:44:03 2021

@author: MarkusRosenberger
"""

'''
In this script, one single model can be trained 
In the end, the history is plotted
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
nNodes = 100

# how many layers you use
# 2, 3 or 4 are possible
nLayers = 3

# for how many epochs you want to train it
nepochs = 50000

checkpoint_filepath = basedir + 'ML_models/3Parameter/Output/Bestmodel/Checkpoints/' + str(nNodes) + 'nodes_' + str(nLayers) + 'layers/{epoch:02d}-{loss:.3e}'


parameters_train = np.load(traindir + 'parameters_train.npy')
Temp_train = np.load(traindir + 'Temp_train.npy')


inputTrain = parameters_train


# use log of Temp, results are way better
outputTrain = Temp_train



# define needed parameters
nInput = np.shape(inputTrain[0])
nOutput = len(Temp_train[0])


if nLayers == 2:
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(nInput)),        
        tf.keras.layers.Dense(nNodes, activation='relu'),
        tf.keras.layers.Dense(nNodes, activation='relu'),
        tf.keras.layers.Dense(nOutput)
        ])
    

elif nLayers == 3:
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(nInput)),        
        tf.keras.layers.Dense(nNodes, activation='relu'),
        tf.keras.layers.Dense(nNodes, activation='relu'),
        tf.keras.layers.Dense(nNodes, activation='relu'),
        tf.keras.layers.Dense(nOutput)
        ])
    
elif nLayers == 4:
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(nInput)),        
        tf.keras.layers.Dense(nNodes, activation='relu'),
        tf.keras.layers.Dense(nNodes, activation='relu'),
        tf.keras.layers.Dense(nNodes, activation='relu'),
        tf.keras.layers.Dense(nNodes, activation='relu'),
        tf.keras.layers.Dense(nOutput)
        ])
    
# for the case that other numbers are used
else: 
    raise ValueError('Only 2, 3 or 4 layers are allowed!')
    
    
loss_fn = tf.keras.losses.MeanSquaredError()
opt = tf.keras.optimizers.Adam()

model.compile(optimizer=opt,loss=loss_fn, metrics=['MeanSquaredError'])


model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    monitor='loss',
    mode='min',
    save_best_only=True)


training_history = model.fit(inputTrain, outputTrain, epochs=nepochs, callbacks=[model_checkpoint_callback])
losshist = training_history.history['loss']


# for the case that the script is run several times
#losshist = []
#losshist = np.hstack((losshist, training_history.history['loss']))


# save the model and the loss history
model.save(basedir + 'ML_models/3Parameter/Models/' + str(nNodes) + 'nodes_' + str(nLayers) + 'layers')
np.save(basedir + 'ML_models/3Parameter/Output/losshist_'  + str(nepochs) + 'epochs_' + str(nNodes) + 'nodes_' + str(nLayers) + 'layers.npy', np.array(losshist[1:]))



''' 
Plot loss history
'''
   
# start plot at 100th epoch because values are way to high before
x = np.arange(100, nepochs, 1)


plt.figure(figsize = (10,5))
plt.plot(x, losshist[100:], label = str(np.min(losshist[100:])) + ' \n at ' + str(np.argmin(losshist[100:])) + ' epochs')
plt.xlabel('Epoch')
plt.xticks([100, int(0.2*nepochs), int(0.4*nepochs), int(0.6*nepochs), int(0.8*nepochs), int(nepochs)])
plt.yscale('log')
plt.legend(loc = 'upper right')
plt.savefig(basedir + 'ML_models/3Parameter/Output/' + str(nepochs) + 'epochs_' + str(nNodes) + 'nodes_' + str(nLayers) + 'layers.png', bbox_inches = 'tight')

plt.show()



