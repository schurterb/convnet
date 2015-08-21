# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 05:42:17 2015

@author: schurterb

Call a matlab function to test the predictions made by a convolutional network. 
"""

import os
import ConfigParser
import argparse
from analyzer import Analyzer
from load_data import LoadData

from pylab import *
ion()

def testprediction(config_file, pred_file=None, label_file=None, out_path=None):     

    config = ConfigParser.ConfigParser()
    config.read(config_file)
    
    test_data = LoadData(directory = config.get('Testing Data', 'folders').split(','), 
                         data_file_name = config.get('Testing Data', 'data_file'),
                         label_file_name = config.get('Testing Data', 'label_file'), 
                         seg_file_name = config.get('Testing Data', 'seg_file'))
    
    res = Analyzer(raw = test_data.get_data()[0],
                   target = test_data.get_labels()[0])
    res.add_results(results_folder = config.get('General','directory'),
                    name = config_file.split('/')[-3],
                    prediction_file = config.get('Testing', 'prediction_file')+'_0', 
                    learning_curve_file = 'learning_curve')           
    res.analyze(-1, pred_file=pred_file, label_file=label_file, out_path=out_path)
    
    print "Displaying Results"
    res.learning(1)
    res.learning(10)
    res.performance()
    #res.display(0)
    
    raw_input("Press enter to close results")
    
    
def testall(directory, pred_file=None, label_file=None, out_path=None):
    folders = os.listdir(directory)
    config_file = directory+folders[0]+"/network.cfg"
    config = ConfigParser.ConfigParser()
    config.read(config_file)
    
    test_data = LoadData(directory = config.get('Testing Data', 'folders').split(','), 
                         data_file_name = config.get('Testing Data', 'data_file'),
                         label_file_name = config.get('Testing Data', 'label_file'),
                         seg_file_name = config.get('Testing Data', 'seg_file'))
    
    res = Analyzer(raw = test_data.get_data()[0], 
                   target = test_data.get_labels()[0])
                   
    for folder in folders:
        config_file = directory+folder+"/network.cfg"
        config = ConfigParser.ConfigParser()
        config.read(config_file)
        
        res.add_results(results_folder = config.get('General','directory'),
                        name = folder,
                        prediction_file = config.get('Testing', 'prediction_file')+'_0', 
                        learning_curve_file = 'learning_curve')
                           
        res.analyze(-1, pred_file=pred_file, label_file=label_file, out_path=out_path)
        
    print "Displaying Results"
    res.learning(1)
    res.learning(10)
    res.performance()
    
    raw_input("Press enter to close results")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", help="Path to network config file. Default is current directory.")
    parser.add_argument("-n", help="Name of network folder in networks directory. Overrides -c flag.")
    parser.add_argument("-all", help="Test the prediction in each folder of the provided directory. Overrides -c and -n flags.")
#    parser.add_argument("--data", help="Path to prediction to test - overrides [Testing Data] section in config file.")
#    parser.add_argument("--labels", help="Path to labels for testing prediction - overrides [Testing Data] section in config file.")
#    parser.add_argument("--outpath", help="Path to folder where test results should be stored - overrides [Testing] section in config file.")
    
    args = parser.parse_args()
    if args.c:
        config_file = args.c
    else:
        config_file = "network.cfg"
    
    if args.n:
        config_file = "networks/" + args.n + "/network.cfg"
    
    if args.all:
        testall(args.all)#, args.data, args.labels, args.outpath)
    else:
        testprediction(config_file)#, args.data, args.labels, args.outpath)
    