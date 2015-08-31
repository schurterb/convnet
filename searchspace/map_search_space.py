# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 10:20:54 2015

@author: schurterb

Script to search for maximum network shapes allowed by selected gpu
"""

import theano
import time
import numpy as np

from cnn import CNN
from trainer import Trainer
from load_data import LoadData

#theano.config.warn=False

print "\nDefining Search Space..." 
#------------------------------------------------------------------------------
#Define search space
folder = 'results/searchspace/'
max_layers = 18
max_filters = 300
max_fsize = 9
max_batch = 100
layer_interval = 2
filter_interval = 50
fsize_interval = 2
batch_interval = 50


#------------------------------------------------------------------------------
#Select device to test on
device = 'gpu0'

theano.sandbox.cuda.use(device)
if (device != 'cpu'):
    theano.config.nvcc.flags='-use=fast=math'
    theano.config.allow_gc=False
   

print "\nLoading Data..." 
#------------------------------------------------------------------------------
#Load data (even though most of it won't be used)
train_data_folder = '/nobackup/turaga/data/fibsem_medulla_7col/trvol-250-1-h5/'
data_file = 'img_normalized.h5'
label_file = 'groundtruth_aff.h5'
seg_file = 'groundtruth_seg.h5'
data = LoadData(directory = train_data_folder, 
                         data_file_name = data_file,
                         label_file_name = label_file,
                         seg_file_name = seg_file)


#------------------------------------------------------------------------------
#Search
log_interval = 100
results = []
print "\nBeginning Search:"
fsize = 7
while(fsize > 2):
    nlayers = 4
    while(nlayers < max_layers):
        nfilters = 50
        while(nfilters < max_filters):
            name = "conv-"+`nlayers`+"."+`nfilters`+"."+`fsize`
            try:
                buildtime = 0
                starttime = time.clock()
                network = CNN(num_layers = nlayers, num_filters = nfilters, 
                              filter_size = fsize, activation = 'relu')
                buildtime = time.clock() - starttime
            except:
                print name+": build fail\n"
             
           
            batchsize = max_batch
            try:
                traintime = 0
                chunck_size = int(round((log_interval*batchsize)**(1.0/3.0)) + (network.net_shape.shape[0]*(network.net_shape[0,1] -1) +1))
                netTrainer = Trainer(network,
                                     data.get_data()[0][0:chunck_size,0:chunck_size,0:chunck_size],
                                     data.get_labels()[0][:,0:chunck_size,0:chunck_size,0:chunck_size],
                                     learning_method = 'ADAM',
                                     log_folder = folder,
                                     log_interval = log_interval)
                                     
                starttime = time.clock()                   
                netTrainer.train(1, False, False)
                traintime = time.clock() - starttime
                
                results.append([nlayers,nfilters,fsize,batchsize,buildtime,traintime])
                print name+" training with batchsize "+`batchsize`+": success\n"
            except:
                print name+" training with batchsize "+`batchsize`+": fail\n"
                nfilters = max_filters
            print "\n"
            nfilters += filter_interval
        nlayers += layer_interval
    fsize -= fsize_interval

results = np.asarray(results)
results.tofile(folder+'convnet_searchspace_b100.csv',sep=',')