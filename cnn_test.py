# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 09:37:19 2015

@author: schurterb

Function to create, train, and test convolutional neural network.
"""

from cnn import CNN
from trainer import Trainer
from load_data import open_data, close_data




#Load the data for training
data_folder = 'nobackup/turaga/data/fibsem_medulla_7col/trvol-250-1-h5/'
data_file = 'img_normalized.h5'
label_file = 'groundtruth_aff.h5'
train_data, train_label, files = open_data(data_folder, data_file, label_file)


#Create the network to train
network = CNN(num_layers = 3, num_filters = 3, filter_size = 3, batch_size = 4, activation = 'relu', cost_func = 'MSE')

#Create a trainer for the network
network_trainer = Trainer(network.get_network(), batch_size = 4, learning_rate = 0.001)

#Train the network
network_trainer.train(train_data, train_label, 10)

#After training, close the data_files
close_data(files)
