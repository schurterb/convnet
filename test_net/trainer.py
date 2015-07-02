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

theano.config.warn.gpu_set_subtensor1=False
#theano.config.nvcc.flags='-use=fast=math'
theano.config.allow_gc=False
theano.config.floatX = 'float32'
theano.sandbox.cuda.use('gpu0')


class Trainer(object):
    
    
    """Define the updates to be performed each training round"""
    def __learning_updates(self, learning_method):
        
        w_grads = T.grad(self.cost, self.w)       
        b_grads = T.grad(self.cost, self.b)
        
        if(learning_method == 'RMSprop'):
            
            #Initialize shared variable to store MS of gradient btw updates
            self.rw = ()
            self.rb = ()
            for layer in range(0, len(self.w)):
                self.rw = self.rw + (theano.shared(np.ones(self.net_shape[layer,:], dtype=theano.config.floatX)) ,)
                self.rb = self.rb + (theano.shared(np.ones(self.net_shape[layer,0], dtype=theano.config.floatX)) ,)      
            
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
            
        elif(learning_method == 'ADAM'):
            
            #Initialize shared variable to store the momentum and the 
            # variance terms btw updates
            self.mw = ()
            self.mb = ()
            self.vw = ()
            self.vb = ()
            for layer in range(0, len(self.w)):
                self.mw = self.mw + (theano.shared(np.zeros(self.net_shape[layer,:], dtype=theano.config.floatX)) ,)
                self.mb = self.mb + (theano.shared(np.zeros(self.net_shape[layer,0], dtype=theano.config.floatX)) ,)
                self.vw = self.vw + (theano.shared(np.ones(self.net_shape[layer,:], dtype=theano.config.floatX)) ,)
                self.vb = self.vb + (theano.shared(np.ones(self.net_shape[layer,0], dtype=theano.config.floatX)) ,)
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
            self.updates =  mw_updates + mb_updates + vw_updates + vb_updates + w_updates + b_updates + t_update
            
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
    Network must be a list constaining the key components of the network
     to be trained, namely its symbolic theano reprentation (first parameter),
     its cost function (second parameter), its shared weight and bias 
     variables (second and third parameters, rspectively)
    """
    def __init__(self, network, train_set, train_labels, **kwargs):
        #Network parameters
        self.X = network[0]
        self.Y = network[1]
        self.out = network[2]
        self.cost = network[3]
        self.w = network[4]
        self.b = network[5]
        self.net_shape = network[6]
        
        #Training parameters
        trainer_status = ""
        learning_method = kwargs.get('learning_method', 'standardSGD')
        trainer_status += "learning_method = "+learning_method+"\n"
        self.print_updates = kwargs.get('print_updates', False)
        self.rng = kwargs.get('rng', np.random.RandomState(42))
        self.batch_size = kwargs.get('batch_size', 100)
        trainer_status += "batch_size = "+`self.batch_size`+"\n"
        self.use_batches = kwargs.get('use_batches', True)
        trainer_status += "use_batches = "+`self.use_batches`+"\n\n"
        self.lr = kwargs.get('learning_rate', 0.0001)
        trainer_status += "learning rate = "+`self.lr`+"\n"
        self.b1 = kwargs.get('beta1', 0.9)
        trainer_status += "beta 1 = "+`self.b1`+"\n"
        self.b2 = kwargs.get('beta2', 0.999)
        trainer_status += "beta 2 = "+`self.b2`+"\n"
        self.damp = kwargs.get('damping', 1.0e-08)
        trainer_status += "damping term = "+`self.damp`+"\n\n"
        trainer_status += "Network shape = "+`self.net_shape`+"\n"
        
        self.log_intervals = kwargs.get('log_intervals', 100)
        self.log_folder = kwargs.get('log_folder', '')
        
        self.seg = int(self.net_shape.shape[0]*(self.net_shape[0,1] -1) +1)
        self.offset = (self.seg -1)/2
        self.input_shape = (self.seg, self.seg, self.seg, self.batch_size)
        self.output_shape = (3, self.batch_size)
        
        log_file = open(self.log_folder + 'trainer_log.txt', 'w')
        log_file.write(trainer_status)
        log_file.close()        

        self.num_examples = kwargs.get('num_examples', 50000) 
        if(self.num_examples < self.batch_size):
            print "Warning: Batch size smaller than set of examples. Duplicate examples WILL be present in each batch."
        #Initialize all the training data at once
        self.__load_training_data(train_set, train_labels)        
        
        #Initialize the list of updates to be performed
        self.__learning_updates(learning_method)
        
        #Initialize the training function
        self.__theano_training_model()
        
        
    """
    Define the function for GPU training
    """
    def __theano_training_model(self):
        #Store training error in shared memory as well
        self.error = theano.shared(np.zeros(self.log_intervals, dtype=theano.config.floatX))
        self.count = theano.shared(np.zeros(1, dtype='int32'))
        self.samples = theano.shared(np.zeros((3, self.batch_size, self.log_intervals), dtype='int32'))  
        
        self.Xsub = theano.shared(np.zeros((self.seg, self.seg, self.seg, self.batch_size),
                                  dtype = theano.config.floatX))
        self.batch_counter = theano.shared(np.zeros(1, dtype='int32'))
                
        set_x_sub = [(self.Xsub, T.set_subtensor(self.Xsub[:,:,:,self.batch_counter], 
                                                 self.training_data[self.samples[0, self.batch_counter, self.count]:self.samples[0, self.batch_counter, self.count]+self.seg,
                                                                    self.samples[1, self.batch_counter, self.count]:self.samples[1, self.batch_counter, self.count]+self.seg,
                                                                    self.samples[2, self.batch_counter, self.count]:self.samples[2, self.batch_counter, self.count]+self.seg])),
                     (self.batch_counter, self.batch_counter+1)]
                     
        x_res, x_updates = theano.scan(lambda:set_x_sub,
                                       n_steps=self.batch_size)
        x_updates = x_updates + [(self.batch_counter, 0)]                                                                                 
        
        log_updates = [
            (self.error, T.set_subtensor(self.error[self.count], self.cost)),
            (self.count, self.count +1)
        ]
        self.updates = x_updates + log_updates + self.updates
        
        n_updates = T.iscalar("n_updates")
        results, updates = theano.scan(lambda:self.updates, 
                                       non_sequences = [self.X, self.Y],
                                       n_steps=n_updates)
                                       
        self.train_model = theano.function(inputs=[n_updates], outputs=[],
                                           updates = updates,
                                           givens = {self.X: self.Xsub,
                                                     self.Y: self.Ysub},
                                           allow_input_downcast=True)
        
    
    """
    Load all the training samples needed for training this network. Ensure that
    there are about equal numbers of positive and negative examples.
    """
    def __load_training_data(self, train_set, train_labels):
          
        #Load all the training data into the GPUs memore
        self.training_data = theano.shared(train_set[:,:,:])
        self.training_labels = theano.shared(train_labels[:,:,:,:])
        
        #List all the positions of negative labels
        side_len = self.training_data.get_value(borrow=True).shape[0]
        negative_samples = []
        for i in range(0, side_len-self.seg):
            for j in range(0, side_len-self.seg):
                for k in range(0, side_len-self.seg):
                    if(self.training_labels.get_value(borrow=True)[:,i+self.offset, j+self.offset, k+self.offset].sum() == 0):
                        negative_samples += [[i, j, k]]
        
        self.negatives = theano.shared(np.asarray(negative_samples, dtype='int32'))
            
        
        
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
    def train(self, num_updates, **kwargs):
                
        self.early_stop = kwargs.get('early_stop', True)

        #Epochs to average over for early stopping
        averaging_len = 100
        if(self.early_stop):
            self.early_stop = 0.0001
        else:
            self.early_stop = 0
            
               
        train_error = np.zeros(num_updates)
        
        epoch = 0
        while(epoch < num_updates/self.log_intervals):
        
            self.samples.set_value(self.rng.randint(0, self.num_examples, self.batch_size*self.log_intervals*3).reshape((3, self.batch_size, self.log_intervals)), borrow=True)            
            
            self.train_model(self.log_intervals)
                
            train_error[epoch*self.log_intervals:(epoch+1)*self.log_intervals] = self.error.get_value(borrow=True)
            self.__log_status(train_error[train_error > 0])
                    
            if(self.print_updates):
                print 'Cost at update '+`epoch`+': '+`train_error[epoch*self.log_intervals]`
                
            if (epoch%averaging_len == 0) and (epoch >= averaging_len*2):
                error_diff = np.mean(train_error[epoch - averaging_len*2:epoch-averaging_len]) - np.mean(train_error[epoch - averaging_len:epoch])
                if(np.abs(error_diff) < self.early_stop):
                  epoch = self.duration
            
            epoch += 1
            
        return train_error
            
