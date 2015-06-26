# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 09:37:19 2015

@author: schurterb

Function to create, train, and test convolutional neural network.
"""

from cnn import CNN
from trainer import Trainer
from load_data import LoadData


#Load the data for training
data_folder = 'nobackup/turaga/data/fibsem_medulla_7col/trvol-250-1-h5/'
data_file = 'img_normalized.h5'
label_file = 'groundtruth_aff.h5'
training_data = LoadData(directory = data_folder, data_file_name = data_file,
                         label_file_name = label_file)
#------------------------------------------------------------------------------
                         
#Structural hyper-parameters
num_layers = 5
num_filters = 6
filter_size = 5 #This only one side of each filter, which are assumed to be cubic
batch_size = 10
activation = 'relu'
#cost_func = 'MSE'

#Create the network to train
network = CNN(num_layers = num_layers, num_filters = num_filters, 
              filter_size = filter_size, activation = activation)
#------------------------------------------------------------------------------
              
              
#Learning Methods include standard SGD, RMSprop, and ADAM
learning_method = 'RMSprop'
use_batches = True #If false, the weights are updated after each example
#Hyper-parameters for the chosen learning method
learning_rate = 0.0001
decay_rate = 0.98
damping = 1.0e-8

#Create a trainer for the network
network_trainer = Trainer(network.get_network(), batch_size = batch_size,
                          learning_method = learning_method,
                          learning_rate = learning_rate, decay_rate = decay_rate,
                          damping = damping, use_batches = use_batches)
#------------------------------------------------------------------------------


#Number of updates to train over
num_updates = 10

#Train the network
train_error = network_trainer.train(training_data.get_data(), training_data.get_labels(), duration = num_updates)
#------------------------------------------------------------------------------


#After training, close the data_files
training_data.close()
