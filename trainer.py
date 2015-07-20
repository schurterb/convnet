# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 06:10:40 2015

@author: schurterb

Trainer class for a CNN using stochastic gradient descent.
"""

import theano
import theano.sandbox.cuda
from theano import tensor as T
import numpy as np

import os
import time
import logging
import csv


theano.config.floatX = 'float32'


class Trainer(object):
    
    
    """Define the updates to be performed each training round"""
    def __perform_updates(self):
        
        w_grads = T.grad(self.cost, self.w)       
        b_grads = T.grad(self.cost, self.b)
        
        if(self.learning_method == 'RMSprop'):
            
            #Initialize shared variable to store MS of gradient btw updates
            self.vw = ()
            self.vb = ()
            for layer in range(0, len(self.w)):
                self.vw = self.vw + (theano.shared(np.ones(self.net_shape[layer,:], dtype=theano.config.floatX), name='vw'+`layer`) ,)
                self.vb = self.vb + (theano.shared(np.ones(self.net_shape[layer,0], dtype=theano.config.floatX), name='vb'+`layer`) ,)      
            
            vw_updates = [
                (r, (self.b2*r) + (1-self.b2)*grad**2)
                for r, grad in zip(self.vw, w_grads)                   
            ]
            vb_updates = [
                (r, (self.b2*r) + (1-self.b2)*grad**2)
                for r, grad in zip(self.vb, b_grads)                   
            ]
            w_updates = [
                (param, param - self.lr*(grad/(T.sqrt(r) + self.damp)))
                for param, r, grad in zip(self.w, self.vw, w_grads)                   
            ]
            b_updates = [
                (param, param - self.lr*(grad/(T.sqrt(r) + self.damp)))
                for param, r, grad in zip(self.b, self.vb, b_grads)                   
            ]    
            self.updates = vw_updates + vb_updates + w_updates + b_updates
            
        elif(self.learning_method == 'ADAM'):
            
            #Initialize shared variable to store the momentum and the 
            # variance terms btw updates
            self.mw = ()
            self.mb = ()
            self.vw = ()
            self.vb = ()
            for layer in range(0, len(self.w)):
                self.mw = self.mw + (theano.shared(np.zeros(self.net_shape[layer,:], dtype=theano.config.floatX), name='mw'+`layer`) ,)
                self.mb = self.mb + (theano.shared(np.zeros(self.net_shape[layer,0], dtype=theano.config.floatX), name='mb'+`layer`) ,)
                self.vw = self.vw + (theano.shared(np.ones(self.net_shape[layer,:], dtype=theano.config.floatX), name='vw'+`layer`) ,)
                self.vb = self.vb + (theano.shared(np.ones(self.net_shape[layer,0], dtype=theano.config.floatX), name='vb'+`layer`) ,)
            self.t = theano.shared(np.asarray(1, dtype=theano.config.floatX))
            
            mw_updates = [
                (m, (self.b1*m) + ((1- self.b1)*grad))
                for m, grad in zip(self.mw, w_grads)                   
            ]
            mb_updates = [
                (m, (self.b1*m) + ((1- self.b1)*grad))
                for m, grad in zip(self.mb, b_grads)                   
            ]
            vw_updates = [
                (v, ((self.b2*v) + (1-self.b2)*(grad**2)) )
                for v, grad in zip(self.vw, w_grads)                   
            ]
            vb_updates = [
                (v, ((self.b2*v) + (1-self.b2)*(grad**2)) )
                for v, grad in zip(self.vb, b_grads)                   
            ]
            w_updates = [
                (param, param - self.lr * (m/(1- (self.b1**self.t) ))/(T.sqrt(v/(1- (self.b2**self.t) ))+self.damp))
                for param, m, v in zip(self.w, self.mw, self.vw)                   
            ]
            b_updates = [
                (param, param - self.lr * ( m/(1- (self.b1**self.t) ))/(T.sqrt( v/(1- (self.b2**self.t) ))+self.damp))
                for param, m, v in zip(self.b, self.mb, self.vb)                   
            ]
            t_update = [
                (self.t, self.t+1)
            ]
            self.updates = mw_updates + mb_updates + vw_updates + vb_updates + w_updates + b_updates + t_update
            
        else: #The default is standard SGD
            
            w_updates = [
                (param, param - self.lr*grad)
                for param, grad in zip(self.w, w_grads)                   
            ]
            b_updates = [
                (param, param - self.lr*grad)
                for param, grad in zip(self.b, b_grads)                   
            ]    
            self.updates = w_updates + b_updates
            
            
    """
    Record the cost for the current batch
    """
    def __set_log(self):
        self.error = theano.shared(np.zeros(self.log_interval, dtype=theano.config.floatX), name='error')
        self.update_counter = theano.shared(np.zeros(1, dtype='int32'), name='update_counter') 
                                        
        log_updates = [
            (self.error, T.set_subtensor(self.error[self.update_counter], self.cost)),
            (self.update_counter, self.update_counter +1),
        ]
        self.updates = log_updates + self.updates
    
    
    """
    Set the training examples for the current batch based on the sample values
    """
    def __load_batch(self): 
        self.batch = theano.shared(np.zeros((3, self.batch_size), 
                                     dtype = 'int32'), name='batch')
        self.batch_counter = theano.shared(np.zeros(1, dtype='int32'), name='batch_counter') 
        self.index = theano.shared(np.zeros(1, dtype='int32'), name='index')                
        
        self.Xsub = theano.shared(np.zeros(self.input_shape,
                                  dtype = theano.config.floatX), name='Xsub')
        self.Ysub = theano.shared(np.zeros(self.output_shape, 
                                  dtype = theano.config.floatX), name='Ysub')
                
        scan_vals = [
            (self.index, self.batch_counter - (self.update_counter*self.batch_size) ),
            (self.Xsub, T.set_subtensor(self.Xsub[:,:,:,self.index[0]], self.training_data[0][self.batch[0, self.batch_counter[0]]:self.batch[0, self.batch_counter[0]] +self.seg, 
                                                                                              self.batch[1, self.batch_counter[0]]:self.batch[1, self.batch_counter[0]] +self.seg, 
                                                                                              self.batch[2, self.batch_counter[0]]:self.batch[2, self.batch_counter[0]] +self.seg]) ),
            (self.Ysub, T.set_subtensor(self.Ysub[:,self.index[0]], self.training_labels[0][:, self.batch[0, self.batch_counter[0]]+self.offset, 
                                                                                               self.batch[1, self.batch_counter[0]]+self.offset, 
                                                                                               self.batch[2, self.batch_counter[0]]+self.offset]) ),
            (self.batch_counter, self.batch_counter+1)
        ]
        outputs, x_updates = theano.scan(lambda:scan_vals, n_steps=self.batch_size)
        self.updates = x_updates + self.updates 
        
        
    """
    Loop through the specified number of updates strictly using the GPU before
    reporting back to log the current status
    """
    def __update_loop(self):
        
        outputs, self.updates = theano.scan(lambda x, y: self.updates, non_sequences = [self.X, self.Y], n_steps=self.log_interval)

  
    """
    Define the function for GPU training
    """
    def __theano_training_model(self):
        
        self.__perform_updates()
        
        self.__set_log()
        
        self.__load_batch()
        
#        #Using this update loop can speed up computation; however, it also uses a significant amount of GPU memory,
#        # which prevents extremely large networks from being trained.
#        self.__update_loop()
        
        self.train_model = theano.function(inputs=[], outputs=[],
                                           updates = self.updates,
                                           givens = [(self.X, self.Xsub),
                                                     (self.Y, self.Ysub)],
                                           allow_input_downcast=True)
                                
        self.reset_counters = theano.function(inputs=[], outputs=[], updates=[(self.batch_counter, [0]), (self.update_counter, [0])])
            
            
    """
    Network must be a list constaining the key components of the network
     to be trained, namely its symbolic theano reprentation (first parameter),
     its cost function (second parameter), its shared weight and bias 
     variables (second and third parameters, rspectively)
    """
    def __init__(self, network, train_data, train_labels, **kwargs):
        #Network parameters
        self.X = network.X
        self.Y = network.Y
        self.out = network.out
        self.cost = network.cost
        self.w = network.w
        self.b = network.b
        self.net_shape = network.net_shape
        
        #Training parameters
        trainer_status = "\n### Convolutional Network Trainer Log ###\n\n"
        trainer_status += "Network Parameters\n"
        trainer_status += "num layers = "+ `network.num_layers` +"\n"
        trainer_status += "num filters = "+ `network.num_filters` +"\n"
        trainer_status += "filter size = "+ `network.filter_size` +"\n"        
        trainer_status += "activation = "+ `network.activation` +"\n\n"
                
        self.rng = kwargs.get('rng', np.random.RandomState(42))
        
        trainer_status += "Trainer Parameters\n"
        trainer_status += "cost function = "+ `network.cost_func` +"\n"
        self.learning_method = kwargs.get('learning_method', 'standardSGD')
        trainer_status += "learning method = "+self.learning_method+"\n"
        self.batch_size = kwargs.get('batch_size', 100)
        trainer_status += "batch size = "+`self.batch_size`+"\n"
        self.lr = kwargs.get('learning_rate', 0.0001)
        trainer_status += "learning rate = "+`self.lr`+"\n"
        
        self.b1 = kwargs.get('beta1', 0.9)
        if(self.learning_method=='ADAM'): trainer_status += "beta 1 = "+`self.b1`+"\n"
        
        self.b2 = kwargs.get('beta2', 0.999)
        self.damp = kwargs.get('damping', 1.0e-08)
        if(self.learning_method=='RMSprop') or (self.learning_method=='ADAM'): 
            trainer_status += "beta 2 = "+`self.b2`+"\n"
            trainer_status += "damping term = "+`self.damp`+"\n"
            
        self.log_interval = kwargs.get('log_interval', 100)
        self.log_folder = kwargs.get('log_folder', '')
        
        self.seg = int(self.net_shape.shape[0]*(self.net_shape[0,1] -1) +1)
        self.offset = (self.seg -1)/2    
        self.input_shape = (self.seg, self.seg, self.seg, self.batch_size)
        self.output_shape = (3, self.batch_size)
        
        self.log_file = self.log_folder + 'trainer.log'
        self.__clear_log()
        self.__init_lc()
        logging.basicConfig(filename=self.log_file, level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.logger.info(trainer_status+"\n")        
        
        #Load the training set into memory
        self.__load_training_data(train_data, train_labels)
        
        #Initialize the training function
        self.__theano_training_model()
                                            

    """
    Load all the training samples needed for training this network. Ensure that
    there are about equal numbers of positive and negative examples.
    """
    def __load_training_data(self, train_set, train_labels):
          
        #Load all the training data into the GPUs memore
        if(type(train_set) == tuple):
            self.training_data = ()
            self.training_labels = ()
            self.data_size = ()
            self.num_dsets = 0
            for data, labels in zip(train_set, train_labels):
                self.training_data += (theano.shared(data[:,:,:], borrow=True) ,)
                self.training_labels += (theano.shared(labels[:,:,:,:], borrow=True) ,)
                self.data_size += (self.training_data[-1].get_value(borrow=True).shape[-1] ,)
                self.num_dsets += 1
        else:
            self.training_data = (theano.shared(train_set[:,:,:], borrow=True) ,)
            self.training_labels = (theano.shared(train_labels[:,:,:,:], borrow=True) ,)
            self.data_size = (self.training_data[-1].get_value(borrow=True).shape[-1] ,)
            self.num_dsets = 1
        
        #List all the positions of negative labels
        self.negatives = ()
        self.epoch_length = 0
        for n in range(self.num_dsets):
            negative_samples = []
            for i in range(0, self.data_size[n]-self.seg):
                for j in range(0, self.data_size[n]-self.seg):
                    for k in range(0, self.data_size[n]-self.seg):
                        if(self.training_labels[n].get_value(borrow=True)[:,i+self.offset, j+self.offset, k+self.offset].sum() == 0):
                            negative_samples += [[i, j, k]]
            
            self.negatives += (np.asarray(negative_samples, dtype='int32') ,)
            self.epoch_length += (self.data_size[n]-self.seg)**3
            
        self.epoch_length = self.epoch_length/(self.log_interval*self.batch_size)
        

    """
    Set the training examples for this batch
    """
    def __get_examples(self):
        samples = np.zeros((3, self.batch_size*self.log_interval), dtype = 'int32')
        for i in range(0, self.batch_size*self.log_interval):
            idx = self.rng.randint(0, self.num_dsets)
            sel = self.rng.randn(1)
            if (sel > 0):   #Search for a positive example. They are common.
                sample = self.rng.randint(0, self.data_size[idx] - self.seg, 3)
                while np.sum(np.sum(self.negatives[idx] == sample, 1) == 3):
                    sample = self.rng.randint(0, self.data_size[idx] - self.seg, 3)
            else:           #Select a negative example from the known negatives, rather than searching for them.
                sel = self.rng.randint(0, self.negatives[idx].shape[0], 1)[0]
                sample = self.negatives[idx][sel]
            samples[:,i] = sample
        self.batch.set_value(samples, borrow=True)

        
    """
    Store the trainining and weight values at regular intervals
    """
    def __store_status(self, error):
        
        with open(self.log_folder + 'learning_curve.csv', 'ab') as lcf:
            fw = csv.writer(lcf, delimiter=',')
            fw.writerow([error])
        
        weights_folder = self.log_folder + 'weights/'
        trainer_folder = self.log_folder + 'trainer/'
        if not os.path.exists(weights_folder):
            os.mkdir(weights_folder)
        if not os.path.exists(trainer_folder):
            os.mkdir(trainer_folder)
        
        for i in range(0, self.net_shape.shape[0]):
            self.w[i].get_value().tofile(weights_folder + 'layer_'+`i`+'_weights.csv', sep=',')
            self.b[i].get_value().tofile(weights_folder + 'layer_'+`i`+'_bias.csv', sep=',')
            if(self.learning_method == 'RMSprop'):
                self.vw[i].get_value().tofile(trainer_folder + 'layer_'+`i`+'_weight_var.csv', sep=',')
                self.vb[i].get_value().tofile(trainer_folder + 'layer_'+`i`+'_bias_var.csv', sep=',')
            elif(self.learning_method == 'ADAM'):
                self.vw[i].get_value().tofile(trainer_folder + 'layer_'+`i`+'_weight_var.csv', sep=',')
                self.vb[i].get_value().tofile(trainer_folder + 'layer_'+`i`+'_bias_var.csv', sep=',')
                self.mw[i].get_value().tofile(trainer_folder + 'layer_'+`i`+'_weight_mnt.csv', sep=',')
                self.mb[i].get_value().tofile(trainer_folder + 'layer_'+`i`+'_bias_mnt.csv', sep=',')
    

    """Clear the logging file"""
    def __clear_log(self):
        with open(self.log_file, 'w'):
            pass
    
    
    """Make sure there is a file for the learning curve"""
    def __init_lc(self):
        if not os.path.isfile(self.log_folder + 'learning_curve.csv'):
            open(self.log_folder + 'learning_curve.csv', 'w').close()
    
    
    """
    Log the current status of the trainer and the network
    """
    def __log_trainer(self, epoch, error, train_time):
        start_cost = error[epoch*self.epoch_length]
        end_cost = error[(epoch+1)*self.epoch_length -1]
        trainer_status  = "\n-- Status at epoch: "+`epoch`+" --\n"
        trainer_status += "Change in average cost: "+`start_cost`+" -> "+`end_cost`+"\n"
        diff = start_cost - end_cost
        pcent = (diff/start_cost)*100
        trainer_status += "     Improvement of "+`diff`+" or "+`pcent`+"%\n"
        trainer_status += "Number of examples seen: "+`self.batch_size*self.log_interval*self.epoch_length`+"\n"
        trainer_status += "Training time: "+`train_time/60`+" minutes\n\n"
        
        self.logger.info(trainer_status)
    
    
    """   
    Train the network on a specified training set with a specified target
     (supervised training). This training uses stochastic gradient descent
     mini-batch sizes set at initialization. Training samples are selected 
     such that there is an equal number of positive and negative samples 
     each batch.        
    Returns: network cost at each update
    """
    def train(self, duration, early_stop = False, print_updates = True):
        
        train_error = np.zeros(duration * self.epoch_length)
        
        epoch_time = 0
        total_time = 0        
        
        epoch = 0
        while(epoch < duration):
            
            starttime = time.clock()
            for i in range(0, self.epoch_length):
                
                self.reset_counters()
                self.__get_examples()
    
                for j in range(0, self.log_interval):
                    self.train_model()      
                
                train_error[i] = np.mean(self.error.get_value(borrow=True))
                
                if(print_updates):
                    print 'Average cost over updates '+`i*self.log_interval`+' - '+`(i+1)*self.log_interval`+' ('+`self.batch_size*self.log_interval`+' examples): '+`train_error[i]`
            
                self.__store_status(train_error[i])
            epoch_time = time.clock() - starttime
            total_time += epoch_time            
            
            self.__log_trainer(epoch, train_error, epoch_time)
            
            if early_stop and (train_error[train_error > 0][-1] < 0.002):
                  epoch = duration
            
            epoch += 1
            
        return train_error
            
