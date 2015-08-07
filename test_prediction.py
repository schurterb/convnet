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


def test_prediction(prediction_folder):
    
    #Open configuration file for this network
    config = ConfigParser.ConfigParser()
    config.read(prediction_folder)
    
    #Set the device on which to perform these computations
    device = config.get('General', 'device')
    theano.sandbox.cuda.use(device)
    if (device != 'cpu'):
        theano.config.nvcc.flags='-use=fast=math'
        theano.config.allow_gc=False
    #------------------------------------------------------------------------------
    
    print 'Opening Data Files'
    test_data = LoadData(directory = config.get('Testing Data', 'folders').split(','), 
                         data_file_name = config.get('Testing Data', 'data_file'),
                         label_file_name = config.get('Testing Data', 'label_file'),
                         seg_file_name = config.get('Testing Data', 'seg_file'))

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", help="path to folder containing predictions. Default is current directory.")
    
    args = parser.parse_args()
    if args.f:
        prediction_folder = args.f
    else:
        prediction_foler = ""
        
    test_prediction(prediction_folder)
    