# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 06:10:40 2015

@author: schurterb

Trainer class for a CNN using stochastic gradient descent.
"""

import time

import theano
import theano.sandbox.cuda
from theano.sandbox.cuda.basic_ops import gpu_from_host
from theano import tensor as T
from theano import Out
import numpy as np

theano.config.nvcc.flags='-use=fast=math'
theano.config.allow_gc=False
theano.config.floatX = 'float32'
theano.sandbox.cuda.use('gpu1')


class Trainer(object):
    
    
    """Define the updates to be performed each training round"""
    def __perform_updates(self):
        
        w_grads = T.grad(self.cost, self.w)       
        b_grads = T.grad(self.cost, self.b)
        
        if(self.learning_method == 'RMSprop'):
            
            #Initialize shared variable to store MS of gradient btw updates
            self.rw = ()
            self.rb = ()
            for layer in range(0, len(self.w)):
                self.rw = self.rw + (theano.shared(np.ones(self.net_shape[layer,:], dtype=theano.config.floatX), name='rw'+`layer`) ,)
                self.rb = self.rb + (theano.shared(np.ones(self.net_shape[layer,0], dtype=theano.config.floatX), name='rb'+`layer`) ,)      
            
            rw_updates = [
                (r, (self.b1*r) + (1-self.b1)*grad**2)
                for r, grad in zip(self.rw, w_grads)                   
            ]
            rb_updates = [
                (r, (self.b1*r) + (1-self.b1)*grad**2)
                for r, grad in zip(self.rb, b_grads)                   
            ]
            w_updates = [
                (param, param - self.lr*(grad/(T.sqrt(r) + self.damp)))
                for param, r, grad in zip(self.w, self.rw, w_grads)                   
            ]
            b_updates = [
                (param, param - self.lr*(grad/(T.sqrt(r) + self.damp)))
                for param, r, grad in zip(self.b, self.rb, b_grads)                   
            ]    
            self.updates = rw_updates + rb_updates + w_updates + b_updates
            
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
        self.samples = theano.shared(np.zeros((3, self.batch_size*self.log_interval), 
                                     dtype = 'int32'), name='samples')
        self.batch_counter = theano.shared(np.zeros(1, dtype='int32'), name='batch_counter') 
        self.index = theano.shared(np.zeros(1, dtype='int32'), name='index')                
        
        self.Xsub = theano.shared(np.zeros(self.input_shape,
                                  dtype = theano.config.floatX), name='Xsub')
        self.Ysub = theano.shared(np.zeros(self.output_shape, 
                                  dtype = theano.config.floatX), name='Ysub')
                
        scan_vals = [
            (self.index, self.batch_counter - (self.update_counter*self.batch_size) ),
            (self.Xsub, T.set_subtensor(self.Xsub[:,:,:,self.index[0]], self.training_data[self.samples[0, self.batch_counter[0]]:self.samples[0, self.batch_counter[0]] +self.seg, 
                                                                                           self.samples[1, self.batch_counter[0]]:self.samples[1, self.batch_counter[0]] +self.seg, 
                                                                                           self.samples[2, self.batch_counter[0]]:self.samples[2, self.batch_counter[0]] +self.seg]) ),
            (self.Ysub, T.set_subtensor(self.Ysub[:,self.index[0]], self.training_labels[:, self.samples[0, self.batch_counter[0]]+self.offset, 
                                                                                            self.samples[1, self.batch_counter[0]]+self.offset, 
                                                                                            self.samples[2, self.batch_counter[0]]+self.offset]) ),
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
        print 'setting updates'
        self.__perform_updates()
        print 'setting logger'
        self.__set_log()
        print 'setting batches'
        self.__load_batch()
        print 'defining loops'
        self.__update_loop()
        print 'preparing function'
        examples = T.imatrix('examples')
        self.train_model = theano.function(inputs=[examples], outputs=[],
                                           updates = self.updates,
                                           givens = [(self.samples, examples),
                                                     (self.X, self.Xsub),
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
        trainer_status = "### Convolutional Network Trainer Log ###\n\n"
        trainer_status += "Network Parameters\n"
        trainer_status += "num layers = "+ `network.num_layers` +"\n"
        trainer_status += "num filters = "+ `network.num_filters` +"\n"
        trainer_status += "filter size = "+ `network.filter_size` +"\n"        
        trainer_status += "activation = "+ `network.activation` +"\n"
        trainer_status += "cost function = "+ `network.cost_func` +"\n\n"
        
        trainer_status += "Trainer Parameters\n"
        self.learning_method = kwargs.get('learning_method', 'standardSGD')
        trainer_status += "learning method = "+self.learning_method+"\n"
        self.print_updates = kwargs.get('print_updates', False)
        self.rng = kwargs.get('rng', np.random.RandomState(42))
        self.batch_size = kwargs.get('batch_size', 100)
        trainer_status += "batch size = "+`self.batch_size`+"\n"
        self.lr = kwargs.get('learning_rate', 0.0001)
        trainer_status += "learning rate = "+`self.lr`+"\n"
        self.b1 = kwargs.get('beta1', 0.9)
        trainer_status += "beta 1 = "+`self.b1`+"\n"
        self.b2 = kwargs.get('beta2', 0.999)
        trainer_status += "beta 2 = "+`self.b2`+"\n"
        self.damp = kwargs.get('damping', 1.0e-08)
        trainer_status += "damping term = "+`self.damp`+"\n\n"
        
        self.log_interval = kwargs.get('log_interval', 100)
        self.log_folder = kwargs.get('log_folder', '')
        
        self.seg = int(self.net_shape.shape[0]*(self.net_shape[0,1] -1) +1)
        self.offset = (self.seg -1)/2    
        self.input_shape = (self.seg, self.seg, self.seg, self.batch_size)
        self.output_shape = (3, self.batch_size)
        
        log_file = open(self.log_folder + 'trainer_log.txt', 'w')
        log_file.write(trainer_status)
        log_file.close()
               
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
        self.training_data = theano.shared(train_set[:,:,:])#, dtype=theano.config.floatX)
        self.training_labels = theano.shared(train_labels[:,:,:,:])#, dtype=theano.config.floatX)
        self.data_size = self.training_data.get_value(borrow=True).shape[-1]
        
        #List all the positions of negative labels
        side_len = self.training_data.get_value(borrow=True).shape[0]
        negative_samples = []
        for i in range(0, side_len-self.seg):
            for j in range(0, side_len-self.seg):
                for k in range(0, side_len-self.seg):
                    if(self.training_labels.get_value(borrow=True)[:,i+self.offset, j+self.offset, k+self.offset].sum() == 0):
                        negative_samples += [[i, j, k]]
        
        self.negatives = np.asarray(negative_samples, dtype='int32')
        

    """
    Set the training examples for this batch
    """
    def __get_examples(self):
        samples = np.zeros((3, self.batch_size*self.log_interval), dtype = 'int32')
        for i in range(0, self.batch_size*self.log_interval):
            sel = self.rng.randn(1)
            if (sel > 0):   #Search for a positive example. They are common.
                sample = self.rng.randint(0, self.data_size - self.seg, 3)
                while np.sum(np.sum(self.negatives == sample, 1) == 3):
                    sample = self.rng.randint(0, self.data_size - self.seg, 3)
            else:           #Select a negative example from the known negatives, rather than searching for them.
                sel = self.rng.randint(0, self.negatives.shape[0], 1)[0]
                sample = self.negatives[sel]
            
            samples[:,i] = sample
        return samples
    
        
    """
    Log the trainining and weight values at regular intervals
    """
    def __log_status(self, error):
        error.tofile(self.log_folder + 'learning_curve.csv', sep=',')
        for i in range(0, self.net_shape.shape[0]):
            self.w[i].get_value().tofile(self.log_folder + 'layer_'+`i`+'_weights.csv', sep=',')
            self.b[i].get_value().tofile(self.log_folder + 'layer_'+`i`+'_bias.csv', sep=',')
    
    
    """   
    Train the network on a specified training set with a specified target
     (supervised training). This training uses stochastic gradient descent
     mini-batch sizes set at initialization. Training samples are selected 
     such that there is an equal number of positive and negative samples 
     each batch.        
    Returns: network cost at each update
    """
    def train(self, duration, early_stop = False):

        #Epochs to average over for early stopping
        averaging_len = self.log_interval
        if(early_stop):
            early_stop = 0.0001
        else:
            early_stop = -1
        
        train_error = np.zeros((duration/self.log_interval, self.log_interval))
        
        epoch = 0
        train_init = time.clock()
        while(epoch < duration/self.log_interval):
                       
            self.reset_counters()
            
            self.train_model(self.__get_examples())           
            
            train_error[epoch] = self.error.get_value(borrow=True)
            
            if(self.print_updates):
                print 'Cost over updates '+`epoch*self.log_interval`+' - '+`(epoch+1)*self.log_interval`+' : '+`np.mean(train_error[epoch])`
            
            if((epoch+1)%self.log_interval == 0):
                self.__log_status(train_error[train_error > 0])
                
            if (epoch%averaging_len == 0) and (epoch >= averaging_len*2):
                error_diff = np.mean(train_error[epoch - averaging_len*2:epoch-averaging_len]) - np.mean(train_error[epoch - averaging_len:epoch])
                if(np.abs(error_diff) < early_stop):
                  epoch = duration
            
            epoch += 1
        train_time = time.clock() - train_init
        
        log_file = open(self.log_folder + 'trainer_log.txt', 'a')
        log_file.write("\nTraining Time = "+`train_time`)
        log_file.close()
            
        return train_error
            
