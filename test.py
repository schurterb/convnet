# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 05:42:17 2015

@author: schurterb

Call a matlab function to test the predictions made by a convolutional network. 
"""

import os
import ConfigParser
import argparse
import subprocess


def testprediction(config_file, pred_file=None, label_file=None, out_path=None):
    
    #Open configuration file for this network
    config = ConfigParser.ConfigParser()
    config.read(config_file)
    
    if not label_file:
        test_data_folder = config.get('Testing Data', 'folders').split(',');
        label_file = test_data_folder[0] + config.get('Testing Data', 'label_file');

    if not pred_file and not out_path:
        out_path = config.get('Testing', 'prediction_folder');
        pred_file = out_path + config.get('Testing', 'prediction_file')+'_0.h5';
    elif not pred_file: 
        pred_file = config.get('Testing', 'prediction_folder') + config.get('Testing', 'prediction_file')+'_0.h5';
        
    name = "evaluating prediction by "+config_file.split('/')[-2]
        
    subprocess.call(["matlab -nosplash -nodisplay -r \"evaluate_predictions(\'"+label_file+"\',\'"+pred_file+"\',\'"+out_path+"\',\'"+name+"\'); exit\""], shell=True);#% (label_file, pred_file, out_path, name)])
    
    
def testall(directory, pred_file=None, label_file=None, out_path=None):
    folders = os.listdir(directory)
    for folder in folders:
        testprediction(folder+"/network.cfg",pred_file,label_file,out_path)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-all", help="Test the prediction in each folder of the provided directory. Overrides -c and -n flags.")
    parser.add_argument("-c", help="Path to network config file. Default is current directory.")
    parser.add_argument("-n", help="Name of network folder in networks directory. Overrides -c flag.")
    parser.add_argument("--data", help="Path to prediction to test - overrides [Testing Data] section in config file.")
    parser.add_argument("--labels", help="Path to labels for testing prediction - overrides [Testing Data] section in config file.")
    parser.add_argument("--outpath", help="Path to folder where test results should be stored - overrides [Testing] section in config file.")
    
    args = parser.parse_args()
    if args.c:
        config_file = args.c
    else:
        config_file = "network.cfg"
    
    if args.n:
        config_file = "networks/" + args.n + "/network.cfg"
    
    if args.all:
        testall(args.all, args.data, args.labels, args.outpath)
    else:
        testprediction(config_file, args.data, args.labels, args.outpath)
    