# -*- coding: utf-8 -*-
"""
Created on Wed May 27 11:23:54 2015

@author: schurterb

Functions for opening and closeing hdf5 files with training and testing data.
"""

import os
import h5py

#Opens hdf5 files containing the training (or testing) data and labels.
#Params: data_folder containing training data
#        data = name of hdf5 file contianing data
#        labels = name of hdf5 file contianing labels for data
#Returns: x = variable accessing data
#         y = variable accessing labels for data
#         tuple containing open hdf5 file accessors, closing them will close x
#       and y. They must be closed after training and testing is complete.
def open_data(data_folder,data, labels):
    
    home_folder = os.getcwd() + '/'
    os.chdir('/.')
    
    data = h5py.File(data_folder + data, 'r')
    x = data['main']

    labels = h5py.File(data_folder + labels, 'r')
    y = labels['main']
    
    #Return home after loading the data
    os.chdir(home_folder)
    return x, y, (data, labels)

#Function to close hdf5 files containing training and testing data
#Params: tuple containing file accessors
def close_data(files):
    for f in files:
        f.close()