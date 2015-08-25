# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 03:56:22 2015

@author: schurterb

Script to analyze the results of net_opt_2, which generates a list of 
folders containing the results of training different networks.
"""

import os
import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
from pylab import *
ion()

from oldanalyzer import Analyzer
from load_data import LoadData


results_folder = 'networks/'
#results_folder = ''

test_data_folder = '/nobackup/turaga/data/fibsem_medulla_7col/tstvol-520-1-h5/'
data_file = 'img_normalized.h5'
label_file = 'groundtruth_aff.h5'
seg_file = 'groundtruth_seg.h5'


#res_files = ('conv-5.6.5', 'conv-5.30.5')#, 'conv-10.20.5', 'conv-8.25.5', 'conv-12.11.5')
res_files = ('conv-5.6.5', 'conv-5.30.5', 'conv-8.25.5', 'conv-8.50.5', 'conv-9.20.5', 'conv-10.20.5', 'conv-12.11.5')
#------------------------------------------------------------------------------ 

#Load the data for testing
print 'Loading Data'
test_data = LoadData(directory = test_data_folder, data_file_name = data_file,
                     label_file_name = label_file, seg_file_name = seg_file)
#------------------------------------------------------------------------------                    

#Check all possible network structures to see which were trained or not.
print 'Loading Results'
results = Analyzer(threshold_steps =100, target = test_data.get_labels()[0], raw = test_data.get_data()[0])
for res_name in res_files:
    folder = results_folder + res_name +'/'
    if os.path.exists(folder):
        results.add_results(results_folder = folder,
                            name = res_name + ' test',
                            prediction_file = 'test_prediction_0', 
                            learning_curve_file = 'MSE_learning')
#        results.add_results(results_folder = folder,
#                            name = res_name + ' train',
#                            prediction_file = 'train_prediction', analyze=True)
#------------------------------------------------------------------------------


print 'Analysis complete'

#Display the learning curves
results.learning(10)

###Display the performance metrics
#results.performance()

print "Displaying results"