#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 09:48:25 2021

@author: MarkusRosenberger
"""

'''
In this script, the training and test data is created with the Kompot Code v2 and 
saved afterwards to a chosen directory
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

# Function that determines when iteration is stopped
def StopFunction(model, t, it):
    
    ShouldStop = False
    
    # starting at something > 1 to get no problems
    if it > 10:
        
    # calculate gradient of Texo between two time steps
        dTdt = ( model.evolve.TgasExoLog[-1] - model.evolve.TgasExoLog[-2] ) / ( model.evolve.tLog[-1] - model.evolve.tLog[-2] )
        
        if np.abs(dTdt) < 1e-5:
            ShouldStop = True
    
    return ShouldStop


# define directories
basedir = '/Users/MarkusRosenberger/Documents/Uni Wien/MasterAstro/Machine Learning/Earth/'
traindir = basedir + 'ML_models/3Parameter/Trainingdata/'
testdir = basedir + 'ML_models/3Parameter/Testdata/'

# some constants
ME = 5.972e24                       # earth mass in kg
rho = 5510                          # earth density in kg/m^3
RE = 6378137                        # radius of Earth in m 


# number of training or test profiles to calculate
n = 512

# define if profiles are 'training' or 'test'
mode = 'training'

parameters_train = np.ndarray(shape=(n, 1,3))
Temp_train = np.ndarray(shape=(n, 104))

print(n)

for i in range(n):
    start = time.time()

    # create random values between given borders
    M = np.random.uniform(low=0.4, high=10)
    fC = np.random.uniform(low=1e-4, high=1e-3)
    fxuv = np.random.uniform(low=1, high=100)
    
    # calculate radius of planet depending in mass and density
    R = (3*M*ME/(4*rho*np.pi))**(1/3) 
    R /= RE
    
    # to use in initialisation
    Rstr = str(R) + ' Rearth'
    Mstr = str(M) + ' Mearth'
    
    # parameter vector to save afterwards
    parameters = [M, fC, fxuv]
    
    # calculate Model
    myModel = kp.Model()
    myModel.Initialise(ParameterFile = basedir + 'parameters.txt', Mpl = Mstr, Rpl = Rstr, fCO2 = float(fC), FxuvIn = float(fxuv))
    myModel.EvolveState(StopFunction = StopFunction)
    
    # output vectors
    parameters_train[i] = parameters
    Temp_train[i] = np.log10(myModel.state.Tgas)

    # just to see if it's still running
    print(i+1, parameters, 'finished in', round((time.time()-start)/60, 3), 'minutes')


if mode == 'training':
    np.save(traindir + 'parameters_train', parameters_train)
    np.save(traindir + 'Temp_train', Temp_train)

if mode == 'test':
    np.save(testdir + 'parameters_test', parameters_train)
    np.save(testdir + 'Temp_test', Temp_train)