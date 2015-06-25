# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 05:42:17 2015

@author: schurterb

Training a network with ADAM and no mini-batches
"""

from cnn import CNN
from trainer import Trainer
from load_data import LoadData


results_folder = 'results/RMSprop_batch/'

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
batch_size = 100
use_mini_batches = True
num_updates = 10000

#Hyper-parameters for the chosen learning method
learning_rate = 0.00001
decay_rate = 0.99
damping = 0.000001
early_stop = True

print_updates = True


#Create the network to train
network = CNN(num_layers = num_layers, num_filters = num_filters, 
              filter_size = filter_size, activation = activation)
#------------------------------------------------------------------------------


#Create a trainer for the network
network_trainer = Trainer(network.get_network(), batch_size = batch_size,
                          learning_rate = learning_rate, decay_rate = decay_rate,
                          damping = damping, use_batches = use_mini_batches, 
                          print_updates = print_updates)
#------------------------------------------------------------------------------


#Load the data for training
training_data = LoadData(directory = train_data_folder, data_file_name = data_file,
                         label_file_name = label_file)
#------------------------------------------------------------------------------


#Train the network
train_error = network_trainer.train(training_data.get_data(), training_data.get_labels(), 
                                    duration = num_updates, early_stop = early_stop)
training_data.close()
#------------------------------------------------------------------------------


#Load the data for testing
testing_data = LoadData(directory = test_data_folder, data_file_name = data_file,
                         label_file_name = label_file)
#------------------------------------------------------------------------------              
   
                      
#Predict the affinity of the test set
prediction = network.predict(testing_data.get_data())
testing_data.close()
#------------------------------------------------------------------------------


#Store all the results
train_error.tofile(results_folder + 'learning_curve.csv', sep=',')
prediction.tofile(results_folder + 'test_prediction.csv', sep=',')
network.save_weights(results_folder)