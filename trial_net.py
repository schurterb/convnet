# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 05:42:17 2015

@author: schurterb

Training a network designed according to the parameters in net_shape.
"""

import time
import os
import numpy as np
from cnn import CNN
from trainer import Trainer
from load_data import LoadData

"""Define and train a trial network"""
def trial_net(net_shape):
        
    train_data_folder = 'nobackup/turaga/data/fibsem_medulla_7col/tstvol-250-2-h5/'
    data_file = 'img_normalized.h5'
    label_file = 'groundtruth_aff.h5'
    
    #Structural hyper-parameters
    num_layers, num_filters, filter_size = net_shape
    activation = 'relu'
    #cost_func = 'MSE'
    
    #Learning Methods include standard SGD, RMSprop, and ADAM
    learning_method = 'RMSprop'
    batch_size = 100
    num_updates = 1
    
    #Hyper-parameters for the chosen learning method
    learning_rate = 0.0001
    beta1 = 0.9
    beta2 = 0.9
    damping = 0.0001
    early_stop = False
    
    print_updates = False
    results_folder = 'results/hyperopt/nl='+`num_layers`+'_nf='+`num_filters`+'_fs='+`filter_size`+'/'
    if not os.path.exists(results_folder): os.makedirs(results_folder)    
    
    print 'creating network'
    #Create the network to train
    network = CNN(num_layers = num_layers, num_filters = num_filters, 
                  filter_size = filter_size, activation = activation)
    #------------------------------------------------------------------------------
    
    #Load the data for training
    training_data = LoadData(directory = train_data_folder, data_file_name = data_file,
                             label_file_name = label_file)
    #------------------------------------------------------------------------------    
    
    print 'training network'
    #Create a trainer for the network
    network_trainer = Trainer(network.get_network(), training_data.get_data(), training_data.get_labels(), 
                              batch_size = batch_size, learning_method = learning_method,
                              learning_rate = learning_rate, beta1 = beta1,
                              beta2 = beta2, damping = damping,
                              log_folder = results_folder,
                              print_updates = print_updates)
    #------------------------------------------------------------------------------
    
    start_time = time.clock()
    #Train the network
    train_error = network_trainer.train(
                                        duration = num_updates, early_stop = early_stop)
    tot_train_time = time.clock() - start_time
    training_data.close()
    #------------------------------------------------------------------------------
    
    loss = np.mean(train_error[-100::])

    network.save_weights(results_folder)
    #train_error.tofile(results_folder + 'learning_curve.csv', sep=',')
    #train_time.tofile(results_folder + 'training_time.csv', sep=',')
    f = open(results_folder + 'loss.txt', 'w')
    f.write("test loss = "+`loss`+"\ntotal train time = "+`tot_train_time`)
    f.close()
    
    return loss