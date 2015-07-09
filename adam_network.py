# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 05:42:17 2015

@author: schurterb

Training a network with ADAM and no mini-batches
"""

from cnn import CNN
from trainer2 import Trainer
from load_data import LoadData


results_folder = 'results/ADAM/'

train_data_folder = 'nobackup/turaga/data/fibsem_medulla_7col/trvol-250-1-h5/'
test_data_folder = 'nobackup/turaga/data/fibsem_medulla_7col/tstvol-520-1-h5/'
data_file = 'img_normalized.h5'
label_file = 'groundtruth_aff.h5'

#Structural hyper-parameters
num_layers = 10
num_filters = 20
filter_size = 5 #This only one side of each filter, which are assumed to be cubic
activation = 'relu'
#cost_func = 'MSE'

#Learning Methods include standard SGD, RMSprop, and ADAM
learning_method = 'ADAM'
batch_size = 100
log_interval = 100
num_updates = 10000

#Hyper-parameters for the chosen learning method
learning_rate = 0.00001
beta1 = 0.9
beta2 = 0.9
damping = 1.0e-8
early_stop = False

print_updates = True

print 'Creating Network'
#Create the network to train
network = CNN(num_layers = num_layers, num_filters = num_filters, 
              filter_size = filter_size, activation = activation)
#------------------------------------------------------------------------------

print 'Opening Data Files'
#Load the data for training
training_data = LoadData(directory = train_data_folder, data_file_name = data_file,
                         label_file_name = label_file)
#------------------------------------------------------------------------------
                         
print 'Preparing Trainer'
#Create a trainer for the network
network_trainer = Trainer(network, training_data.get_data(), training_data.get_labels(),
                          batch_size = batch_size, 
                          learning_method = learning_method,
                          learning_rate = learning_rate, beta1 = beta1,
                          beta2 = beta2, damping = damping,
                          log_folder = results_folder, log_interval = log_interval,
                          print_updates = print_updates)
#------------------------------------------------------------------------------

print 'Training...'
#Train the network
train_error = network_trainer.train(duration = num_updates, early_stop = early_stop)
training_data.close()
#------------------------------------------------------------------------------


##Load the data for testing
#testing_data = LoadData(directory = test_data_folder, data_file_name = data_file,
#                         label_file_name = label_file)
##------------------------------------------------------------------------------              
#   
#                      
##Predict the affinity of the test set
#prediction = network.predict(testing_data.get_data())
#testing_data.close()
##------------------------------------------------------------------------------

print 'Trianing Complete'
#Store all the results
#prediction.tofile(results_folder + 'test_prediction.csv', sep=',')
network.save_weights(results_folder)