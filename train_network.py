# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 05:42:17 2015

@author: schurterb

Training a network with ADAM and no mini-batches
"""

import time
from cnn import CNN
from trainer import Trainer
from load_data import LoadData


results_folder = 'ADAMtest/'

train_data_folder = 'nobackup/turaga/data/fibsem_medulla_7col/trvol-250-1-h5/'
test_data_folder = 'nobackup/turaga/data/fibsem_medulla_7col/tstvol-520-1-h5/'
data_file = 'img_normalized.h5'
label_file = 'groundtruth_aff.h5'

#Structural hyper-parameters
num_layers = 5
num_filters = 6
filter_size = 5 #This only one side of each filter, which are assumed to be cubic
activation = 'relu'
#cost_func = 'MSE'

#Learning Methods include standard SGD, RMSprop, and ADAM
learning_method = 'RMSprop'
batch_size = 10
num_updates = 100
log_interval = 100

#Hyper-parameters for the chosen learning method
learning_rate = 0.00001
beta1 = 0.9
beta2 = 0.9
damping = 1.0e-8
early_stop = False

print_updates = True

starttime=time.clock()
#Create the network to train
print '\nInitializing Network'
network = CNN(num_layers = num_layers, num_filters = num_filters, 
              filter_size = filter_size, activation = activation)
#------------------------------------------------------------------------------
print 'Opening Data Files'
#Load the data for training
training_data = LoadData(directory = train_data_folder, data_file_name = data_file,
                         label_file_name = label_file)
#------------------------------------------------------------------------------
print 'Initializing Trainer'
#Create a trainer for the network
network_trainer = Trainer(network, training_data.get_data(), training_data.get_labels(), 
                          batch_size = batch_size, log_interval=log_interval,
                          learning_method = learning_method,
                          learning_rate = learning_rate, beta1 = beta1,
                          beta2 = beta2, damping = damping, 
                          log_folder = results_folder)
#------------------------------------------------------------------------------


init_time = time.clock() - starttime
print "Initialization = " + `init_time` + " seconds"
print 'Training...\n'
#Train the network
starttime = time.clock()
train_error= network_trainer.train(num_updates, early_stop, print_updates)
total_time = time.clock() - starttime                                    
training_data.close()
#------------------------------------------------------------------------------

#Store all the results
train_error.tofile(results_folder + 'learning_curve.csv', sep=',')
print "Total Training Time    = " + `total_time` + " seconds"
print "  Sampling Time        = " + `sampt` + " seconds"
print "  Training Time        = " + `traint` + " seconds"