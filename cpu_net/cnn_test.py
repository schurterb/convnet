# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 09:37:19 2015

@author: schurterb

Function to create, train, and test convolutional neural network.
"""

from cnn_old import CNN
from trainer_old import Trainer
from load_data_old import LoadData


results_folder = 'results/test_results/'

#Load the data for training
data_folder = 'nobackup/turaga/data/fibsem_medulla_7col/trvol-250-1-h5/'
data_file = 'img_normalized.h5'
label_file = 'groundtruth_aff.h5'
training_data = LoadData(directory = data_folder, data_file_name = data_file,
                         label_file_name = label_file)
#------------------------------------------------------------------------------
                         
#Structural hyper-parameters
num_layers = 3
num_filters = 6
filter_size = 5 #This only one side of each filter, which are assumed to be cubic
activation = 'relu'
#cost_func = 'MSE'

print 'creating first network'
#Create a 3-layer network to train
network = CNN(num_layers = num_layers, num_filters = num_filters, 
              filter_size = filter_size, activation = activation)
#------------------------------------------------------------------------------
              
              
#Learning Methods include standard SGD, RMSprop, and ADAM
learning_method = 'RMSprop'
use_batches = False #If false, the weights are updated after each example
#Hyper-parameters for the chosen learning method
learning_rate = 0.0001
decay_rate = 0.98
damping = 1.0e-8

#Number of updates to train over
num_updates = 5
batch_size = 10

print 'training first network'
#Create a trainer for the network
network_trainer = Trainer(network.get_network(), batch_size = batch_size,
                          learning_method = learning_method,
                          learning_rate = learning_rate, decay_rate = decay_rate,
                          damping = damping, log_folder = results_folder,
                          use_batches = use_batches, print_updates = True)
#------------------------------------------------------------------------------

#Train the network
train_error = network_trainer.train(training_data.get_data(), training_data.get_labels(), duration = num_updates)
#------------------------------------------------------------------------------

#Save the weights of the 3-layer network
network.save_weights(results_folder)
#------------------------------------------------------------------------------

#Now create a 5-layer network with the first 2 layers being from the previous network
num_layers = 5

print 'creating second network'
#Create a 3-layer network to train
network = CNN(num_layers = num_layers, num_filters = num_filters, 
              filter_size = filter_size, activation = activation,
              load_folder = results_folder)
  

print 'training second network'            
#Create a trainer for the network
network_trainer = Trainer(network.get_network(), batch_size = batch_size,
                          learning_method = learning_method,
                          learning_rate = learning_rate, decay_rate = decay_rate,
                          damping = damping, log_folder = results_folder,
                          use_batches = use_batches, print_updates = True)
#------------------------------------------------------------------------------

#Train the network
train_error = network_trainer.train(training_data.get_data(), training_data.get_labels(), duration = num_updates)
#------------------------------------------------------------------------------

#Save the weights of the 3-layer network
network.save_weights(results_folder)
#------------------------------------------------------------------------------

#After training, close the data_files
training_data.close()
