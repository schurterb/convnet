# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 05:42:17 2015

@author: schurterb

Train a network as defined by a provided config file
"""

import time
import ConfigParser
import argparse

import theano
from cnn import CNN
from trainer import Trainer
from load_data import LoadData


def trainnetwork(config_file):
    
    #Open configuration file for this network
    config = ConfigParser.ConfigParser()
    config.read(config_file)
    
    #Set the device on which to perform these computations
    device = config.get('General', 'device')
    theano.sandbox.cuda.use(device)
    if (device != 'cpu'):
        theano.config.nvcc.flags='-use=fast=math'
        theano.config.allow_gc=False
    #------------------------------------------------------------------------------
        
    print '\nOpening Data Files'
    #Load the data for training
    training_data = LoadData(directory = config.get('Training Data', 'folders').split(','),
                             data_file_name = config.get('Training Data', 'data_file'),
                             label_file_name = config.get('Training Data', 'label_file'),
                             seg_file_name = config.get('Training Data', 'seg_file'))
    #------------------------------------------------------------------------------
                             
    starttime=time.clock()
    #Create the network and trainer    
    if config.getboolean('Network', 'load_weights'):
        print 'Initializing Network'
        network = CNN(weights_folder = config.get('General', 'directory')+config.get('Network', 'weights_folder'),
                      activation = config.get('Network', 'activation'),
                      cost_func = config.get('Network', 'cost_func'))
         
        print 'Initializing Trainer'             
        network_trainer = Trainer(network, training_data.get_data(), training_data.get_labels(), training_data.get_segments(),
                                  use_malis = config.getboolean('Training', 'use_malis'),
                                  chunk_size = config.getint('Training', 'chunk_size'),  
                                  batch_size = config.getint('Training', 'batch_size'),
                                  learning_method = config.get('Training', 'learning_method'),
                                  learning_rate = config.getfloat('Training', 'learning_rate'), 
                                  beta1 = config.getfloat('Training', 'beta1'),
                                  beta2 = config.getfloat('Training', 'beta2'), 
                                  damping = config.getfloat('Training', 'damping'), 
                                  trainer_folder = config.get('General', 'directory')+config.get('Training', 'trainer_folder'),
                                  log_interval = config.getint('Training', 'log_interval'),
                                  log_folder = config.get('General', 'directory')+config.get('Training', 'log_folder'))
    else:
        print 'Initializing Network'
        network = CNN(num_layers = config.getint('Network', 'num_layers'), 
                      num_filters = config.getint('Network', 'num_filters'), 
                      filter_size = config.getint('Network', 'filter_size'), 
                      activation = config.get('Network', 'activation'),
                      cost_func = config.get('Network', 'cost_func'))
                      
        print 'Initializing Trainer'             
        network_trainer = Trainer(network, training_data.get_data(), training_data.get_labels(), training_data.get_segments(),
                                  use_malis = config.getboolean('Training', 'use_malis'),
                                  batch_size = config.getint('Training', 'batch_size'),
                                  learning_method = config.get('Training', 'learning_method'),
                                  learning_rate = config.getfloat('Training', 'learning_rate'), 
                                  beta1 = config.getfloat('Training', 'beta1'),
                                  beta2 = config.getfloat('Training', 'beta2'), 
                                  damping = config.getfloat('Training', 'damping'), 
                                  log_interval = config.getint('Training', 'log_interval'),
                                  log_folder = config.get('General', 'directory')+config.get('Training', 'log_folder'))
                              
    init_time = time.clock() - starttime    
    #------------------------------------------------------------------------------
    print "Initialization = " + `init_time` + " seconds"
    
    starttime = time.clock()
    #Train the network
    print 'Training...\n'
    train_error = network_trainer.train(config.getint('Training', 'num_epochs'), 
                                        config.getboolean('Training', 'early_stop'), 
                                        config.getboolean('Training', 'print_updates'))
    total_time = time.clock() - starttime     
    #------------------------------------------------------------------------------   
    print "Total Time     = " + `total_time` + " seconds"                            
    
    training_data.close()
    return train_error
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", help="path to network config file. default is current directory")
    parser.add_argument("-n", help="name of network folder in networks directory. overrides -c flag")
    
    args = parser.parse_args()
    if args.c:
        config_file = args.c
    else:
        config_file = "network.cfg"
    
    if args.n:
        config_file = "networks/" + args.n + "/network.cfg"
        
    trainnetwork(config_file)
    
    
    