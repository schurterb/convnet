# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 05:42:17 2015

@author: schurterb

Training a network designed according to the parameters in net_shape.
"""

import time
import os
from cnn import CNN
from trainer import Trainer
from load_data import LoadData

"""Define and train a trial network"""
def trial_net(net_shape):
        
    train_data_folder = 'nobackup/turaga/data/fibsem_medulla_7col/trvol-250-1-h5/'
    test_data_folder = 'nobackup/turaga/data/fibsem_medulla_7col/tstvol-520-1-h5/'
    data_file = 'img_normalized.h5'
    label_file = 'groundtruth_aff.h5'
    
    #Structural hyper-parameters
    num_layers, num_filters, filter_size = net_shape
    activation = 'relu'
    #cost_func = 'MSE'
    
    #Learning Methods include standard SGD, RMSprop, and ADAM
    learning_method = 'ADAM'
    batch_size = 100
    use_mini_batches = False
    num_updates = 1000
    
    #Hyper-parameters for the chosen learning method
    learning_rate = 0.001
    beta1 = 0.9
    beta2 = 0.99
    decay_rate = 1 - 1.0e-08
    damping = 1.0e-8
    early_stop = True
    
    print_updates = False
    results_folder = 'results/net_opt_2/nl='+`num_layers`+'_nf='+`num_filters`+'_fs='+`filter_size`+'/'
    if not os.path.exists(results_folder): os.makedirs(results_folder)    
    
    print 'creating network'
    #Create the network to train
    network = CNN(num_layers = num_layers, num_filters = num_filters, 
                  filter_size = filter_size, activation = activation)
    #------------------------------------------------------------------------------
    
    print 'training network'
    #Create a trainer for the network
    network_trainer = Trainer(network.get_network(), batch_size = batch_size,
                              learning_method = learning_method,
                              learning_rate = learning_rate, beta1 = beta1,
                              beta2 = beta2, decay_rate = decay_rate,
                              damping = damping, use_batches = use_mini_batches,
                              log_folder = results_folder,
                              print_updates = print_updates)
    #------------------------------------------------------------------------------
    
    #Load the data for training
    training_data = LoadData(directory = train_data_folder, data_file_name = data_file,
                             label_file_name = label_file)
    #------------------------------------------------------------------------------
    start_time = time.clock()
    #Train the network
    train_error = network_trainer.train(training_data.get_data(), training_data.get_labels(), 
                                        duration = num_updates, early_stop = early_stop)
    train_time = time.clock() - start_time
    training_data.close()
    #------------------------------------------------------------------------------
    
    print 'testing network'
    #Load the data for testing
    testing_data = LoadData(directory = test_data_folder, data_file_name = data_file,
                             label_file_name = label_file)
    #------------------------------------------------------------------------------              
                       
    #Predict the affinity of the test set
    loss = network.loss(testing_data.get_data(), testing_data.get_labels())
    testing_data.close()
    #------------------------------------------------------------------------------

    network.save_weights(results_folder)
    train_error.tofile(results_folder + 'learning_curve.csv', sep=',')
    train_time.tofile(results_folder + 'training_time.csv', sep=',')
    f = open(results_folder + 'loss.txt')
    f.write('test loss = '+`loss`)
    f.close()
    
    return loss