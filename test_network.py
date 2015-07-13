# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 05:42:17 2015

@author: schurterb

Training a network with ADAM and no mini-batches
"""

import time
from cnn import CNN
from load_data import LoadData


results_folder = 'results/ADAM_2/'
prediction_file_name = 'test_prediction'

train_data_folder = 'nobackup/turaga/data/fibsem_medulla_7col/trvol-250-1-h5/'
test_data_folder = 'nobackup/turaga/data/fibsem_medulla_7col/tstvol-520-1-h5/'
data_file = 'img_normalized.h5'
label_file = 'groundtruth_aff.h5'


starttime=time.clock()

print '\nInitializing Network'
network = CNN(weights_folder = results_folder)
#------------------------------------------------------------------------------


print 'Opening Data Files'
test_data = LoadData(directory = test_data_folder, data_file_name = data_file,
                         label_file_name = label_file)
#------------------------------------------------------------------------------
init_time = time.clock() - starttime                        
                         
                         
print 'Making Predictions'
starttime = time.clock()
network.predict(test_data.get_data(), results_folder, prediction_file_name)
testing_time = time.clock() - starttime
#------------------------------------------------------------------------------


print "Initialization = " + `init_time` + " seconds"
print "Testing Time   = " + `testing_time` + " seconds"
#------------------------------------------------------------------------------