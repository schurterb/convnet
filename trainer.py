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
from malis import findMalisLoss

import os
import time
import logging
import csv


theano.config.floatX = 'float32'


class Trainer(object):
    
    """Define the updates for RMSprop"""
    def __rmsprop(self, w_grads, b_grads):
        #Initialize shared variable to store MS of gradient btw updates
        self.vw = ()
        self.vb = ()
        if self.load_folder and self.malis:            
            trainer_folder = self.load_folder + 'trainer/malis/'
            if os.path.exists(trainer_folder):
                for layer in range(0, len(self.w)):
                    self.vw = self.vw + (theano.shared(np.genfromtxt(self.load_folder+'layer_'+`layer`+'_weight_var.csv', delimiter=',').reshape(self.net_shape[layer,:]).astype(theano.config.floatX), name='vw'+`layer`) ,)
                    self.vb = self.vb + (theano.shared(np.genfromtxt(self.load_folder+'layer_'+`layer`+'_bias_var.csv', delimiter=',').reshape(self.net_shape[layer,0]).astype(theano.config.floatX), name='vb'+`layer`) ,)
        elif self.load_folder:            
            trainer_folder = self.load_folder + 'trainer/basic/'
            if os.path.exists(trainer_folder):
                for layer in range(0, len(self.w)):
                    self.vw = self.vw + (theano.shared(np.genfromtxt(self.load_folder+'layer_'+`layer`+'_weight_var.csv', delimiter=',').reshape(self.net_shape[layer,:]).astype(theano.config.floatX), name='vw'+`layer`) ,)
                    self.vb = self.vb + (theano.shared(np.genfromtxt(self.load_folder+'layer_'+`layer`+'_bias_var.csv', delimiter=',').reshape(self.net_shape[layer,0]).astype(theano.config.floatX), name='vb'+`layer`) ,)
        else:
            for layer in range(0, len(self.w)):
                    self.vw = self.vw + (theano.shared(np.ones(self.net_shape[layer,:], dtype=theano.config.floatX), name='vw'+`layer`) ,)
                    self.vb = self.vb + (theano.shared(np.ones(self.net_shape[layer,0], dtype=theano.config.floatX), name='vb'+`layer`) ,)
        
        if(len(self.vw) == 0):
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
        return vw_updates + vb_updates + w_updates + b_updates
        
        
    """Define the updates for ADAM"""
    def __adam(self, w_grads, b_grads):
        #Initialize shared variable to store the momentum and the 
            # variance terms btw updates
        self.mw = ()
        self.mb = ()
        self.vw = ()
        self.vb = ()
        if self.load_folder and self.malis:            
            trainer_folder = self.load_folder + 'trainer/malis/'
            if os.path.exists(trainer_folder):
                for layer in range(0, len(self.w)):
                    self.mw = self.mw + (theano.shared(np.genfromtxt(self.load_folder+'layer_'+`layer`+'_weight_mnt.csv', delimiter=',').reshape(self.net_shape[layer,:]).astype(theano.config.floatX), name='mw'+`layer`) ,)
                    self.mb = self.mb + (theano.shared(np.genfromtxt(self.load_folder+'layer_'+`layer`+'_bias_mnt.csv', delimiter=',').reshape(self.net_shape[layer,0]).astype(theano.config.floatX), name='mb'+`layer`) ,)
                    self.vw = self.vw + (theano.shared(np.genfromtxt(self.load_folder+'layer_'+`layer`+'_weight_var.csv', delimiter=',').reshape(self.net_shape[layer,:]).astype(theano.config.floatX), name='vw'+`layer`) ,)
                    self.vb = self.vb + (theano.shared(np.genfromtxt(self.load_folder+'layer_'+`layer`+'_bias_var.csv', delimiter=',').reshape(self.net_shape[layer,0]).astype(theano.config.floatX), name='vb'+`layer`) ,)
        elif self.load_folder:            
            trainer_folder = self.load_folder + 'trainer/basic/'
            if os.path.exists(trainer_folder):
                for layer in range(0, len(self.w)):
                    self.mw = self.mw + (theano.shared(np.genfromtxt(self.load_folder+'layer_'+`layer`+'_weight_mnt.csv', delimiter=',').reshape(self.net_shape[layer,:]).astype(theano.config.floatX), name='mw'+`layer`) ,)
                    self.mb = self.mb + (theano.shared(np.genfromtxt(self.load_folder+'layer_'+`layer`+'_bias_mnt.csv', delimiter=',').reshape(self.net_shape[layer,0]).astype(theano.config.floatX), name='mb'+`layer`) ,)
                    self.vw = self.vw + (theano.shared(np.genfromtxt(self.load_folder+'layer_'+`layer`+'_weight_var.csv', delimiter=',').reshape(self.net_shape[layer,:]).astype(theano.config.floatX), name='vw'+`layer`) ,)
                    self.vb = self.vb + (theano.shared(np.genfromtxt(self.load_folder+'layer_'+`layer`+'_bias_var.csv', delimiter=',').reshape(self.net_shape[layer,0]).astype(theano.config.floatX), name='vb'+`layer`) ,)
        else:
            for layer in range(0, len(self.w)):
                    self.mw = self.mw + (theano.shared(np.zeros(self.net_shape[layer,:], dtype=theano.config.floatX), name='mw'+`layer`) ,)
                    self.mb = self.mb + (theano.shared(np.zeros(self.net_shape[layer,0], dtype=theano.config.floatX), name='mb'+`layer`) ,)
                    self.vw = self.vw + (theano.shared(np.ones(self.net_shape[layer,:], dtype=theano.config.floatX), name='vw'+`layer`) ,)
                    self.vb = self.vb + (theano.shared(np.ones(self.net_shape[layer,0], dtype=theano.config.floatX), name='vb'+`layer`) ,)
                    
        if(len(self.vw) == 0):
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
        return mw_updates + mb_updates + vw_updates + vb_updates + w_updates + b_updates + t_update
        
        
    """Define the updates for standard SGD"""
    def __standard(self, w_grads, b_grads):     
        
        w_updates = [
            (param, param - self.lr*grad)
            for param, grad in zip(self.w, w_grads)                   
        ]
        b_updates = [
            (param, param - self.lr*grad)
            for param, grad in zip(self.b, b_grads)                   
        ]    
        return w_updates + b_updates
        
        
    """Define the updates for MALIS training method"""
    def __malis_grad(self):
        
        self.mgrad = T.tensor4('grad')
        self.out = self.out.dimshuffle(0,4,3,2,1)
        self.grad_counter = theano.shared(np.zeros(1, dtype='int32'), name='grad_counter')
        w_grad = ()
        b_grad = ()
        for pw, pb in zip(self.w, self.b):
            w_grad += (theano.shared(np.zeros(pw.get_value(borrow=True).shape, dtype=theano.config.floatX)) ,)
            b_grad += (theano.shared(np.zeros(pb.get_value(borrow=True).shape, dtype=theano.config.floatX)) ,)
        
        flat_mgrad = self.mgrad.flatten()
        flat_out = self.out.flatten()
        
        
        wgrad_updates = [
            (grad, flat_mgrad[self.grad_counter[0]]*T.grad(flat_out[self.grad_counter[0]], param))
            for grad, param in zip(w_grad, self.w)
        ]
        bgrad_updates = [
            (grad, flat_mgrad[self.grad_counter[0]]*T.grad(flat_out[self.grad_counter[0]], param))
            for grad, param in zip(b_grad, self.b)
        ]
        
        
        if(self.learning_method == 'RMSprop'):
            param_updates = self.__rmsprop(w_grad, b_grad)            
        elif(self.learning_method == 'ADAM'):
            param_updates = self.__adam(w_grad, b_grad)            
        else: #The default is standard SGD   
            param_updates = self.__standard(w_grad, b_grad)
            
        count_update = [
            (self.grad_counter, self.grad_counter+1)
        ]
        
        output, malis_updates = theano.scan(lambda a, b:wgrad_updates+bgrad_updates+param_updates+count_update, 
                                            non_sequences=[self.X, self.mgrad], 
                                            n_steps=3*(self.chunk_size**3))
        
        return malis_updates
    
    
    """Define the updates to be performed each training round"""
    def __perform_updates(self):  

        if self.malis:
            self.updates = self.__malis_grad()
        else:
            w_grad = T.grad(self.cost, self.w)
            b_grad = T.grad(self.cost, self.b)
        
            if(self.learning_method == 'RMSprop'):
                self.updates = self.__rmsprop(w_grad, b_grad)
                
            elif(self.learning_method == 'ADAM'):
                self.updates = self.__adam(w_grad, b_grad)
                
            else: #The default is standard SGD   
                self.updates = self.__standard(w_grad, b_grad)
            
    
    """
    Record the cost for the current batch
    """
    def __set_log(self):
        if self.malis:
            self.Y = T.tensor4('Y')
            self.pixerr = T.mean(1/2.0*((self.out - self.Y.dimshuffle('x',0,1,2,3))**2), dtype=theano.config.floatX) 
            
        else:
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
        self.batch = theano.shared(np.zeros((4, self.batch_size), 
                                     dtype = 'int32'), name='batch')
        self.batch_counter = theano.shared(np.zeros(1, dtype='int32'), name='batch_counter') 
        self.index = theano.shared(np.zeros(1, dtype='int32'), name='index')                
        
        self.Xsub = theano.shared(np.zeros(self.input_shape,
                                  dtype = theano.config.floatX), name='Xsub')
        self.Ysub = theano.shared(np.zeros(self.output_shape, 
                                  dtype = theano.config.floatX), name='Ysub')
               
        scan_vals = [
            (self.index, self.batch_counter - (self.update_counter*self.batch_size) ),
            (self.Xsub, T.set_subtensor(self.Xsub[:,:,:,self.index[0]], self.training_data[self.batch.get_value(borrow=True)[0, self.batch_counter.get_value(borrow=True)[0]]]
                                                                                          [self.batch[1, self.batch_counter[0]]:self.batch[1, self.batch_counter[0]] +self.seg, 
                                                                                           self.batch[2, self.batch_counter[0]]:self.batch[2, self.batch_counter[0]] +self.seg, 
                                                                                           self.batch[3, self.batch_counter[0]]:self.batch[3, self.batch_counter[0]] +self.seg]) ),
            (self.Ysub, T.set_subtensor(self.Ysub[:,self.index[0]], self.training_labels[self.batch.get_value(borrow=True)[0, self.batch_counter.get_value(borrow=True)[0]]]
                                                                                        [:, self.batch[1, self.batch_counter[0]]+self.offset, 
                                                                                            self.batch[2, self.batch_counter[0]]+self.offset, 
                                                                                            self.batch[3, self.batch_counter[0]]+self.offset])),
            (self.batch_counter, self.batch_counter+1)
        ]
        outputs, x_updates = theano.scan(lambda:scan_vals, n_steps=self.batch_size)
        self.updates = x_updates + self.updates 

    
    """
    Define the function(s) for GPU training
    """
    def __theano_training_model(self):
        
        self.__perform_updates()
            
        self.__set_log()
        
        if self.malis:  
            self.Xsub = theano.shared(np.zeros(self.input_shape,
                                               dtype = theano.config.floatX), name='Xsub')
            self.Ysub = theano.shared(np.zeros(self.output_shape, 
                                               dtype = theano.config.floatX), name='Ysub')  
                                               
            self.malis_forward = theano.function(inputs = [], outputs = self.out,
                                                 givens = [(self.X, self.Xsub)],
                                                 allow_input_downcast=True)
            self.malis_backward = theano.function(inputs = [self.mgrad], outputs = self.pixerr,
                                                  updates = self.updates,
                                                  givens = [(self.X, self.Xsub),
                                                            (self.Y, self.Ysub)],
                                                  allow_input_downcast=True)
            self.reset_counter = theano.function(inputs=[], outputs=[], updates=[(self.grad_counter, [0])])
        else:            
            self.__load_batch()
            
            self.train_model = theano.function(inputs=[], outputs=[],
                                               updates = self.updates,
                                               givens = [(self.X, self.Xsub),
                                                         (self.Y, self.Ysub)],
                                               allow_input_downcast=True)
                                    
            self.reset_counter = theano.function(inputs=[], outputs=[], updates=[(self.batch_counter, [0]), (self.update_counter, [0])])
            
        
    """
    Network must be a list constaining the key components of the network
     to be trained, namely its symbolic theano reprentation (first parameter),
     its cost function (second parameter), its shared weight and bias 
     variables (second and third parameters, rspectively)
    """
    def __init__(self, network, train_data, train_labels, train_seg = None, **kwargs):
        #Network parameters
        self.X = network.X
        self.Y = network.Y
        self.out = network.out
        self.cost = network.cost
        self.w = network.w
        self.b = network.b
        self.net_shape = network.net_shape
        
        trainer_status = "\n### Convolutional Network Trainer Log ###\n\n"
        trainer_status += "Network Parameters\n"
        trainer_status += "num layers = "+ `network.num_layers` +"\n"
        trainer_status += "num filters = "+ `network.num_filters` +"\n"
        trainer_status += "filter size = "+ `network.filter_size` +"\n"        
        trainer_status += "activation = "+ `network.activation` +"\n\n"
                
        self.rng = kwargs.get('rng', np.random.RandomState(42))
        
        #Training parameters
        trainer_status += "Trainer Parameters\n"
        self.malis = kwargs.get('use_malis', False)
        if self.malis:
            trainer_status += "cost function = Rand Index\n"
        else:
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
        self.load_folder = kwargs.get('trainer_folder', None)
            
        self.log_interval = kwargs.get('log_interval', 100)
        self.log_folder = kwargs.get('log_folder', '')
        
        self.seg = int(self.net_shape.shape[0]*(self.net_shape[0,1] -1) +1)
        self.offset = (self.seg -1)/2    
        if self.malis:
            self.chunk_size = int(round((self.batch_size*self.log_interval)**(1.0/3.0)))
            self.input_shape = (self.chunk_size+self.seg, self.chunk_size+self.seg, self.chunk_size+self.seg, 1)
            self.output_shape = (self.chunk_size, self.chunk_size, self.chunk_size, 3)
        else:
            self.chunk_size = self.seg
            self.input_shape = (self.seg, self.seg, self.seg, self.batch_size)
            self.output_shape = (3, self.batch_size)
        
        self.log_file = self.log_folder + 'trainer.log'
        self.__clear_log()
        self.__init_lc()
        logging.basicConfig(filename=self.log_file, level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.logger.info(trainer_status+"\n")        
        
        #Load the training set into memory
        if self.malis and not train_seg:
            print "ERROR: Training with MALIS requires groundtruth segmentation.\n"
            exit(0)
        self.__load_training_data(train_data, train_labels, train_seg)
        
        #Initialize the training function
        self.__theano_training_model()
                                            

    """
    Load all the training samples needed for training this network. Ensure that
    there are about equal numbers of positive and negative examples.
    """
    def __load_training_data(self, train_set, train_labels, train_seg=None):
          
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
            
        #If there are ground-truth segmentations to load, load them into cpu memory 
        # (they aren't needed on the gpu)
        if train_seg and (type(train_seg) == tuple):
            self.segmentations = ()
            for seg in train_seg:
                self.segmentations = (np.asarray(seg, dtype=np.intc, order='F') ,)
        elif (train_seg != None):
            self.segmentations = (np.asarray(train_seg, dtype=np.intc, order='F') ,)
        else:
            print "No groundtruth segmentations provided."
            
        
        if(self.learning_method == 'malis'):
            self.epoch_length = 0
            for dsize in self.data_size:
                self.epoch_length += (dsize - self.chunk_size)**3
        else:
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
    Set the training examples for this set of batches
    """
    def __set_batches(self):
        samples = np.zeros((4, self.batch_size*self.log_interval), dtype = 'int32')
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
            samples[:,i] = np.append(idx, sample)
        self.batch.set_value(samples, borrow=True)
        
        
    """
    Set an image chunk for malis training
    """
    def __set_chunk(self):
        idx = self.rng.randint(0, self.num_dsets)
        sample = self.rng.randint(0, self.data_size[idx] - (self.chunk_size+(self.seg-1)), 3)
        self.Xsub.set_value(self.training_data[idx].get_value(borrow=True)[sample[0]:sample[0]+self.chunk_size+(self.seg-1),
                                                                           sample[1]:sample[1]+self.chunk_size+(self.seg-1),
                                                                           sample[2]:sample[2]+self.chunk_size+(self.seg-1)].reshape((self.chunk_size+(self.seg-1),self.chunk_size+(self.seg-1),self.chunk_size+(self.seg-1),1)), borrow=True)
        self.Ysub.set_value(np.transpose(self.training_labels[idx].get_value(borrow=True)[:, self.offset+sample[0]:self.offset+sample[0]+self.chunk_size,
                                                                                             self.offset+sample[1]:self.offset+sample[1]+self.chunk_size,
                                                                                             self.offset+sample[2]:self.offset+sample[2]+self.chunk_size]), borrow=True)
        self.compTrue = np.transpose(self.segmentations[idx], (2,1,0))[self.offset+sample[0]:self.offset+sample[0]+self.chunk_size,
                                                                       self.offset+sample[1]:self.offset+sample[1]+self.chunk_size,
                                                                       self.offset+sample[2]:self.offset+sample[2]+self.chunk_size]
    
    
    """
    Store the trainining and weight values at regular intervals
    """
    def __store_status(self, error, error_type='MSE'):
        
                   
        if(error_type == 'rand'):
            error_file = 'randIndex.csv'
        else:
            error_file = 'learning_curve.csv'
        with open(self.log_folder + error_file, 'ab') as ef:
            fw = csv.writer(ef, delimiter=',')
            fw.writerow([error])
        
        weights_folder = self.log_folder + 'weights/'
        if self.malis:
            trainer_folder = self.log_folder + 'trainer/malis/'
        else:
            trainer_folder = self.log_folder + 'trainer/basic/'
        
        if not os.path.exists(weights_folder):
            os.mkdir(weights_folder)
        if not os.path.exists(trainer_folder):
            os.makedirs(trainer_folder)
        
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
        if self.malis and not os.path.isfile(self.log_folder + 'randIndex.csv'):
            open(self.log_folder + 'randIndex.csv', 'w').close()
    
    
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
        
        #Testing
        self.epoch_length = 20
        pred_time = 0
        malis_time = 0
        train_time = 0        
        res = []
        
        epoch = 0
        while(epoch < duration):
            
            starttime = time.clock()
            for i in range(0, self.epoch_length):
                   
                self.reset_counter()
                if self.malis:
                    self.__set_chunk()
                    
                    checktime = time.clock()
                    prediction = self.malis_forward()
                    pred_time += time.clock() - checktime
                    
                    checktime = time.clock()
                    dloss, randIndex, totLoss = findMalisLoss(self.compTrue, 
                                                              self.Ysub.get_value(borrow=True).astype(dtype='f', order='F'),
                                                              prediction.reshape((self.chunk_size, self.chunk_size, self.chunk_size, 3)).astype(dtype='f', order='F'))
                    malis_time += time.clock() - checktime
                                                              
                    checktime = time.clock()
                    train_error[i] = self.malis_backward(dloss)
                    train_time += time.clock() - checktime
                    
                    res.append((self.Xsub.get_value(borrow=True), self.Ysub.get_value(borrow=True),  prediction.reshape((self.chunk_size, self.chunk_size, self.chunk_size, 3)), dloss))
                    if(print_updates):
                        print 'Rand Index for MALIS update',i,'(',(self.chunk_size**3),'examples):',randIndex
                        
                    self.__store_status(randIndex, )
                else:
                    self.__set_batches()
                    
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
            
        return train_error, pred_time, malis_time, train_time, res
            
