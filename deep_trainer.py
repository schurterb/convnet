# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 05:17:53 2015

@author: schurterb

A DeepTrainer uses the cnn and trainer classes to train deep networks
layer-by-layer, thereby reducing the training time.
"""

import os
import time
import numpy as np
from cnn import CNN
from trainer import Trainer
from load_data import LoadData


class DeepTrainer(object):
    
    def __init__(self, num_layers, num_filters, filter_size, **kwargs):
        
        self.num_layers = num_layers
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.activation = kwargs.get('activation', 'relu')
        self.cost_func = kwargs.get('cost_func', 'MSE')  
        
        self.results_folder = kwargs.get('results_folder', '')
        self.train_dir = kwargs.get('train_directory', '')
        self.train_data_file = kwargs.get('train_data_File', 'train_data.csv')
        self.train_label_file = kwargs.get('train_label_file', 'train_labels.csv')
        self.test_dir = kwargs.get('test_directory', '')
        self.test_data_file = kwargs.get('test_data_File', 'train_data.csv')
        self.test_label_file = kwargs.get('test_label_file', 'train_labels.csv')
        
        self.train_data = LoadData(directory = self.data_dir,
                                   data_file_name = self.train_data_file,
                                   label_file_name = self.train_label_file)
        self.test_data = LoadData(directory = self.test_dir,
                                   data_file_name = self.test_data_file,
                                   label_file_name = self.test_label_file)
        self.__init_report()                           
     
     
    """Initial report, creating the deeptrainer log"""
    def __init_report(self): 
        report = "### DeepTrainer Log ###\n"
        report += "network depth = "+`self.num_layers`+"\n"
        report += "filters per layer = "+`self.num_filters`+"\n"
        report += "filter size = "+`self.filter_size`+"\n"
        report += "activation function = "+`self.activation`+"\n\n"
        
        f = open(self.results_folder + 'deeptrainer_log.txt', 'w')
        f.write(report)
        f.close()
   
   
    """Append a training report to the deeptrainer_log"""
    def __train_report(self, error, train_time, depth):
        report =  "-- Training update at "+`depth`+" layers --\n"
        report += "Final training error = "+`error[-1]`+" \n"
        report += "Training time = "+`train_time/60`+" minutes \n\n"
        f = open(self.results_folder + 'deeptrainer_log.txt', 'w')
        f.append(report)
        f.close()
        
        error.tofile(self.results_folder + `depth` + 'layer_lc.csv', sep=',')
    
    
    """Final report to the deeptrainer log"""
    def __final_report(self, prediction): 
        report = "%% Final Results %%\n"
        
        f = open(self.results_folder + 'deeptrainer_log.txt', 'w')
        f.append(report)
        f.close()
        
        prediction.tofile(self.results_folder + 'test_prediction.csv', sep=',')
        
        
    """Train a single layer of the network"""
    def __train_layer(self, depth):
        try:
            assert self.num_layers >= depth

            #Generate the network to train
            network = CNN(num_layers = depth, 
                          num_filters = self.num_filters, 
                          filter_size = self.filter_size, 
                          activation = self.activation,
                          load_folder = self.results_folder)
            
            #Training parameters
            learning_method = 'ADAM'
            learning_rate = 0.001
            beta1 = 0.9
            beta2 = 0.99
            decay_rate = 1 - 1.0e-08
            damping = 1.0e-8
            early_stop = True
            
            n_batches = 200
            batch_size = 100
            use_batches = False
            
            #Create a trainer for the network
            network_trainer = Trainer(network.get_network(), batch_size = batch_size,
                                      learning_method = learning_method,
                                      learning_rate = learning_rate, beta1 = beta1,
                                      beta2 = beta2, decay_rate = decay_rate,
                                      damping = damping, use_batches = use_batches,
                                      log_folder = self.results_folder)
                                      
            #Train this version of the network
            start_time = time.clock()
            train_error = network_trainer.train(self.train_data.get_data(), self.train_data.get_labels(), 
                                                duration = n_batches, early_stop = early_stop)
            train_time = time.clock() - start_time
                                                
            #Store results from training this layer
            self.__report(train_error, train_time, depth)
            network.save_weights(self.results_folder)
        except:
            print "Error: training depth exceeds network depth"
            
            
    """
    Train the deep network one layer at a time, starting with 2 layers    
    (since that is the smallest network we can start with)
    """
    def train(self):
        
        for nl in range(2, len(self.num_layers)+1):
            self.__train_layer(nl)
        
        #Generate the network to train
        network = CNN(num_layers = self.num_layers, 
                      num_filters = self.num_filters, 
                      filter_size = self.filter_size, 
                      activation = self.activation,
                      load_folder = self.results_folder)
                      
        prediction = network.predict(self.test_data.get_data())

        self.train_data.close()
        self.test_data.close()        
        
        self.__final_report(prediction)