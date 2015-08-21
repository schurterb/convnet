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
from analyzer import Analyzer
from load_data import LoadData


def testprediction(config_file, pred_file=None, label_file=None, out_path=None):     
    
    #Open configuration file for this network
    config = ConfigParser.ConfigParser()
    config.read(config_file)
    
    if not label_file:
        test_data_folder = config.get('Testing Data', 'folders').split(',');
        label_file = test_data_folder[0] + config.get('Testing Data', 'label_file');

    if not (pred_file and out_path):
        out_path = config.get('General', 'directory') + config.get('Testing', 'prediction_folder');
        pred_file = out_path + config.get('Testing', 'prediction_file')+'_0.h5';
    elif not pred_file: 
        pred_file = config.get('General', 'directory') + config.get('Testing', 'prediction_folder') + config.get('Testing', 'prediction_file')+'_0.h5';
        
    name = "evaluating prediction by "+config_file.split('/')[-2]
    
    subprocess.call(["matlab -nosplash -nodisplay -r \"evaluate_predictions(\'"+label_file+"\',\'"+pred_file+"\',\'"+out_path+"\',\'"+name+"\'); exit\""], shell=True);#% (label_file, pred_file, out_path, name)])
  

def testall(directory, pred_file=None, label_file=None, out_path=None):
    folders = os.listdir(directory)
    for folder in folders:
        testprediction(folder+"/network.cfg",pred_file,label_file,out_path)
        

def testprediction2(config_file, pred_file=None, label_file=None, out_path=None):     

    print "Args:",config_file,pred_file,label_file,out_path    
    
    config = ConfigParser.ConfigParser()
    config.read(config_file)
    
    test_data = LoadData(directory = config.get('Testing Data', 'folders').split(','), 
                         data_file_name = config.get('Testing Data', 'data_file'),
                         label_file_name = config.get('Testing Data', 'label_file'), 
                         seg_file_name = config.get('Testing Data', 'seg_file'))
    
    res = Analyzer(raw = test_data.get_data()[0],
                   target = test_data.get_labels()[0])
    res.add_results(results_folder = config.get('General','directory') + config.get('Testing', 'prediction_folder'),
                    name = config_file.split('/')[-],
                    prediction_file = config.get('Testing', 'prediction_file')+'_0', 
                    learning_curve_file = 'learning_curve')           
    res.analyze(0, pred_file=pred_file, label_file=label_file, out_path=out_path)
    
    
def testall2(directory, pred_file=None, label_file=None, out_path=None):
    folders = os.listdir(directory)
    config_file = folders[0]+"/network.cfg"
    config = ConfigParser.ConfigParser()
    config.read(config_file)
    
    test_data = LoadData(directory = config.get('Testing Data', 'folders').split(','), 
                         data_file_name = config.get('Testing Data', 'data_file'),
                         label_file_name = config.get('Testing Data', 'label_file'),
                         seg_file_name = config.get('Testing Data', 'seg_file'))
    
    res = Analyzer(raw = test_data.get_data()[0], 
                   target = test_data.get_labels()[0])
                   
    for folder in folders:
        config_file = folder+"/network.cfg"
        config = ConfigParser.ConfigParser()
        config.read(config_file)
        
        res.add_results(results_folder = config.get('Testing', 'prediction_folder') ,
                        name = folder.split('/')[-2],
                        prediction_file = config.get('Testing', 'prediction_file')+'_0', 
                        learning_curve_file = 'learning_curve')
                           
        res.analyze(-1, pred_file=pred_file, label_file=label_file, out_path=out_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", help="Path to network config file. Default is current directory.")
    parser.add_argument("-n", help="Name of network folder in networks directory. Overrides -c flag.")
    parser.add_argument("-all", help="Test the prediction in each folder of the provided directory. Overrides -c and -n flags.")
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
        print "Testing all networks\n"
        testall(args.all, args.data, args.labels, args.outpath)
    else:
        print "Testing one network\n"
        testprediction2(config_file, args.data, args.labels, args.outpath)
    