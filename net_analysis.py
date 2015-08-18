# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 03:56:22 2015

@author: schurterb

Script to analyze the results of net_opt_2, which generates a list of 
folders containing the results of training different networks.
"""

import os
import numpy as np
from analyzer import Analyzer
from load_data import LoadData


results_folder = 'networks/'
#results_folder = ''

test_data_folder = '/nobackup/turaga/data/fibsem_medulla_7col/tstvol-520-2-h5/'
data_file = 'img_normalized.h5'
label_file = 'groundtruth_aff.h5'
seg_file = 'groundtruth_seg.h5'


#res_files = ('ADAM', 'ADAM_1', 'ADAM_2')#, 'ADAM_3')
#res_files = ('rmsp_1', 'rmsp_2')#, 'rmsp_3')
#res_files = ("results/deep_nets_1/ADAM", "results/learn_opt_res1/name='RMSprop'_lr=0.0001_b1=0.9_b2=0_dp=1e-08", "results/deep_nets_2/rmsp_1")
res_files = ('conv-5.6.5', 'conv-5.30.5', 'conv-10.20.5', 'conv-8.25.5', 'conv-12.11.5')
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
                            learning_curve_file = 'randIndex(10648 ex)',
                            analyze=False)
#        results.add_results(results_folder = folder,
#                            name = res_name + ' train',
#                            prediction_file = 'train_prediction', analyze=True)
#------------------------------------------------------------------------------


print 'Analysis complete'

#Display the learning curves
results.learning(10)

###Display the performance metrics
#results.performance()
#
##Close test data
##test_data.close()
#print 'thresholds: ',results.theta
#
#print 'tp:',results.tp[0]
#print 'fp:',results.fp[0]
#
#print 'npos:',results.npos[0]
#print 'nneg:',results.nneg[0]
