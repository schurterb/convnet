# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 03:56:22 2015

@author: schurterb

Script to analyze the results of net_opt_2, which generates a list of 
folders containing the results of training different networks.
"""

import os
import ConfigParser
from analyzer import Analyzer
from load_data import LoadData


def ViewResults( **kwargs):
    directory = kwargs.get('directory', '')
    network = kwargs.get('network', None)
    prediction_file = kwargs.get('predictions_file', None)
    
    if network:
        #Assume that all networks are tested on the same set of data
        config = ConfigParser.ConfigParser()
        config.read("networks/"+network+"/network.cfg")
        data = LoadData(directory = config.get('Testing Data', 'folders').split(',')[0],
                        data_file_name = config.get('Testing Data', 'data_file'),
                        label_file_name = config.get('Testing Data', 'label_file'))
        
        if not prediction_file: prediction_file = "test_prediction_0"
        
        results = Analyzer(target = data.get_labels()[0], raw = data.get_data()[0])
        results.add_results(results_folder = "networks/"+network+'/',
                            name = network,
                            prediction_file = prediction_file)
    
    else:
        folders = os.listdir(directory)
        networks = []
        for folder in folders:
            if os.path.isfile(directory + folder+"/network.cfg"):
                networks.append(folder)
        
        #Assume that all networks are tested on the same set of data
        config = ConfigParser.ConfigParser()
        config.read(directory+networks[0]+"/network.cfg")
        data = LoadData(directory = config.get('Testing Data', 'folders').split(',')[0],
                        data_file_name = config.get('Testing Data', 'data_file'),
                        label_file_name = config.get('Testing Data', 'label_file'))
        
        if not prediction_file: prediction_file = "test_prediction_0"
        
        results = Analyzer(target = data.get_labels()[0], raw = data.get_data()[0])
        for net in networks:
            results.add_results(results_folder = directory+net+'/',
                                name = net,
                                prediction_file = prediction_file)
                                
    return results

