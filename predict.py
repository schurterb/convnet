# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 05:42:17 2015

@author: schurterb

Training a network with ADAM and no mini-batches
"""

import os
import glob
import time
import ConfigParser
import argparse

import theano
from cnn import CNN
from load_data import LoadData


def makeprediction(config_file, data_file=None, out_path=None, out_file=None, gpu=None):
    
    #Open configuration file for this network
    config = ConfigParser.ConfigParser()
    config.read(config_file)
    
    #Set the device on which to perform these computations
    if gpu:
        theano.sandbox.cuda.use(gpu)
        theano.config.nvcc.flags='-use=fast=math'
        theano.config.allow_gc=False
    else:
        device = config.get('General', 'device')
        theano.sandbox.cuda.use(device)
        if (device != 'cpu'):
            theano.config.nvcc.flags='-use=fast=math'
            theano.config.allow_gc=False
    #------------------------------------------------------------------------------

    starttime=time.clock()
    print '\nInitializing Network'
    if os.path.exists(config.get('General', 'directory')+config.get('Network', 'weights_folder')):
        network = CNN(weights_folder = config.get('General', 'directory')+config.get('Network', 'weights_folder'),
                      activation = config.get('Network', 'activation'))
    else:
        print 'Error: Weights folder does not exist. Could not initialize network'
        return;
    #------------------------------------------------------------------------------
    
    print 'Opening Data Files'
    if data_file:
        test_data = LoadData(directory = '', data_file_name = data_file)
    else:
        test_data = LoadData(directory = config.get('Testing Data', 'folders').split(','), 
                             data_file_name = config.get('Testing Data', 'data_file'))
    #------------------------------------------------------------------------------
    init_time = time.clock() - starttime
    print "Initialization = " + `init_time` + " seconds"       
                           
            
    starttime = time.clock()                 
    print 'Making Predictions'
    if out_path and out_file:
        network.predict(test_data.get_data(),
                        results_folder = out_path,
                        name = out_file)
    elif out_path:
        network.predict(test_data.get_data(),
                        results_folder = out_path,
                        name = config.get('Testing', 'prediction_file'))
    elif out_file:
        network.predict(test_data.get_data(),
                        results_folder = config.get('General', 'directory')+config.get('Testing', 'prediction_folder'),
                        name = out_file)
    else:
        network.predict(test_data.get_data(),
                        results_folder = config.get('General', 'directory')+config.get('Testing', 'prediction_folder'),
                        name = config.get('Testing', 'prediction_file'))
    pred_time = time.clock() - starttime
    #------------------------------------------------------------------------------
    print "Prediction Time   = " + `pred_time` + " seconds"
        
    test_data.close()
    
    
def predictall(directory, data_file=None, out_path=None, out_file=None, gpu=None):
    folders = os.listdir(directory)
    networks = []
    for folder in folders:
        if os.path.isfile(directory+folder+"/network.cfg") and not glob.glob(directory+folder+"/results/*.h5"):
            networks.append(folder)
    
    print "Making predictions for the following networks:\n",networks
    for net in networks:
        makeprediction(directory+net+"/network.cfg", data_file, out_path, out_file, gpu)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", help="Path to network config file. Default is current directory.")
    parser.add_argument("-n", help="Name of network folder in networks directory - overrides -c flag.")
    parser.add_argument("-all", help="Makes predictions with all networks in the given directory that do not yet have predictions.")
    parser.add_argument("--data", help="Path to data to make predictions on - overrides [Testing Data] section in config file.")
    parser.add_argument("--outpath", help="Path to folder where prediction should be stored - overrides [Testing] section in config file.")
    parser.add_argument("--outfile", help="Name of file containing prediction - overrides [Testing] section in config file.")
    parser.add_argument("--gpu", help="Select gpu to use. Useful when makeing predictions for all networks in a directory.")
    
    args = parser.parse_args()
    if args.c:
        config_file = args.c
    else:
        config_file = "network.cfg"
    
    if args.n:
        config_file = "networks/" + args.n + "/network.cfg"
        
    if args.all:
        predictall(args.all, args.data, args.outpath, args.outfile, args.gpu)
    else:
        makeprediction(config_file, args.data, args.outpath, args.outfile, args.gpu)
    