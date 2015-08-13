# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 03:56:22 2015

@author: schurterb

Script to analyze the results of net_opt_2, which generates a list of 
folders containing the results of training different networks.
"""

import os
import ConfigParser
import numpy as np
from analyzer import Analyzer
from load_data import LoadData



def analysis(directory, pred_file=None, lc_file=None, ri_file=None):
    folders = os.listdir(directory)
    networks = []
    for folder in folders:
        if os.path.exits(folder+"/network.cfg") and os.path.exits(folder+"/results/"):
            networks.append(folder)
    
    #Assume that all networks are tested on the same set of data
    config = ConfigParser.ConfigParser()
    config.read(networks[0]+"/network.cfg")
    data = LoadData(directory = config.get('Testing Data', 'folders').split(',')[0],
                    data_file_name = config.get('Testing Data', 'data_file'),
                    label_file_name = config.get('Testing Data', 'label_file'))
    
    if not pred_file: pred_file = "test_prediction"
    if not lc_file: lc_file = "learning_curve"
    if not ri_file: ri_file = "randIndex"
    
    results = Analyzer(target = data.get_labels()[0], raw = data.get_data()[0])
    for net in networks:
        results.add_results(results_folder = net+"/results",
                            name = net,
                            prediction_file = pred_file,
                            learning_curve_file = lc_file)
                            
    return results

