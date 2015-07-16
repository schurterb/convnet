# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 05:42:17 2015

@author: schurterb

Training a network with ADAM and no mini-batches
"""

import time
import ConfigParser
import argparse

import theano
from cnn import CNN
from load_data import LoadData


def test_network(config_file):
    
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

    starttime=time.clock()
    print '\nInitializing Network'
    network = CNN(weights_folder = config.get('Network', 'weights_folder'),
                  activation = config.get('Network', 'activation'),
                  cost_func = config.get('Network', 'cost_func'))
    #------------------------------------------------------------------------------
    
    print 'Opening Data Files'
    test_data = LoadData(directory = config.get('Testing Data', 'folders').split(',')[0], 
                         data_file_name = config.get('Testing Data', 'data_file'),
                         label_file_name = config.get('Testing Data', 'label_file'))
    #------------------------------------------------------------------------------
    init_time = time.clock() - starttime                        
                           
                             
    print 'Making Predictions'
    starttime = time.clock()
    network.predict(test_data.get_data(),
                    results_folder = config.get('Testing', 'prediction_folder'), 
                    name = config.get('Testing', 'prediction_file'))
    testing_time = time.clock() - starttime
    #------------------------------------------------------------------------------
    
    print "Initialization = " + `init_time` + " seconds"
    print "Testing Time   = " + `testing_time` + " seconds"
    #------------------------------------------------------------------------------
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="path to .ini file for network. default is network.ini")
    
    args = parser.parse_args()
    if args.config:
        config_file = args.config
    else:
        config_file = "network.ini"
        
    test_network(config_file)
    