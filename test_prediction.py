# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 05:42:17 2015

@author: schurterb

Training a network with ADAM and no mini-batches
"""

import time
import ConfigParser
import argparse

import matlab.engine


def test_prediction(config_file):
    
    #Open configuration file for this network
    config = ConfigParser.ConfigParser()
    config.read(config_file)
    
    test_data_folder = config.get('Testing Data', 'folders').split(',');
    label_file = config.get('Testing Data', 'label_file');

    results_folder = config.get('Testing', 'prediction_folder');
    pred_file = config.get('Testing', 'prediction_file')+'_0.h5';
    
    print "Starting Matlab"
    eng = matlab.engine.start_matlab()
    print "Evaluating Predictions"
    eng.evaluate_predictions(test_data_folder[0]+label_file, results_folder+pred_file, results_folder, "Testing "+results_folder.split('/')[-1]+" predictions")
    print "Evaluation Complete"

    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", help="path to .ini file for network. default is network.ini")
    
    args = parser.parse_args()
    if args.f:
        config_file = args.f
    else:
        config_file = "network.ini"
        
    test_prediction(prediction_folder)
    