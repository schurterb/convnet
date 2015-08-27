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
from malis import malisloss

import os
import time
import logging
import csv


theano.config.floatX = 'float32'


class Trainer(object):        
        
    """Define the cost function used to evaluate this network"""
    def __set_cost(self):
        if(self.cost_func == 'rand'):
            self.cost = malisloss()(self.out.dimshuffle(4,3,2,1,0), self.Y.dimshuffle(4,3,2,1,0), self.C.dimshuffle(3,2,1,0))
        if(self.cost_func == 'class'):
            self.cost = T.mean(T.nnet.binary_crossentropy(self.out, self.Y), dtype=theano.config.floatX)
        else:
            self.cost = T.mean(1/2.0*((self.out - self.Y)**2), dtype=theano.config.floatX)        
    
    
    """Define the updates for RMSprop"""
    def __rmsprop(self, w_grads, b_grads):
        self.vw = ()
        self.vb = ()
        if self.load_folder and (self.cost_func == 'rand'):            
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
        self.mw = ()
        self.mb = ()
        self.vw = ()
        self.vb = ()
        if self.load_folder and (self.cost_func == 'rand'):            
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
    
    
    """Define the updates to be performed each training round"""
    def __perform_updates(self):  

        w_grad = T.grad(self.cost, self.w)
        b_grad = T.grad(self.cost, self.b)
    
        if(self.learning_method == 'RMSprop'):
            self.updates = self.__rmsprop(w_grad, b_grad) + self.updates 
            
        elif(self.learning_method == 'ADAM'):
            self.updates = self.__adam(w_grad, b_grad) + self.updates 
            
        else: #The default is standard SGD   
            self.updates = self.__standard(w_grad, b_grad) + self.updates 
            
    
    """
    Record the cost for the current batch
    """
    def __set_log(self):
        self.error = theano.shared(np.zeros(self.log_interval, dtype=theano.config.floatX), name='error')
        log_updates = [
            (self.error, T.set_subtensor(self.error[self.update_counter], self.cost)),
            (self.update_counter, self.update_counter +1),
        ]
        self.updates = log_updates + self.updates
    
    
    """
    Set the training examples for the current batch based on the sample values
    """
    def __load_batch(self): 
        self.update_counter = theano.shared(np.zeros(1, dtype='int32'), name='update_counter')
        self.batch = theano.shared(np.zeros((4, self.batch_size), 
                                     dtype = 'int32'), name='batch')
        self.batch_counter = theano.shared(np.zeros(1, dtype='int32'), name='batch_counter') 
        self.index = theano.shared(np.zeros(1, dtype='int32'), name='index')                
        
        self.Xsub = theano.shared(np.zeros(self.input_shape,
                                  dtype = theano.config.floatX), name='Xsub')
        self.Ysub = theano.shared(np.zeros(self.output_shape, 
                                  dtype = theano.config.floatX), name='Ysub')
        self.Csub = theano.shared(np.zeros(self.output_shape[1:5], 
                                  dtype = theano.config.floatX), name='Csub')
               
        scan_vals = [
            (self.index, self.batch_counter - (self.update_counter*self.batch_size) ),
            (self.Xsub, T.set_subtensor(self.Xsub[:,:,:,self.index[0]], self.training_data[self.batch[0, self.batch_counter[0]],
                                                                                           self.batch[1, self.batch_counter[0]]:self.batch[1, self.batch_counter[0]] +self.seg, 
                                                                                           self.batch[2, self.batch_counter[0]]:self.batch[2, self.batch_counter[0]] +self.seg, 
                                                                                           self.batch[3, self.batch_counter[0]]:self.batch[3, self.batch_counter[0]] +self.seg
                                                                                           ]) ),
            (self.Ysub, T.set_subtensor(self.Ysub[:,:,:,:,self.index[0]], self.training_labels[self.batch[0, self.batch_counter[0]],
                                                                                               :, self.batch[1, self.batch_counter[0]]+self.offset:self.batch[1, self.batch_counter[0]]+self.offset+self.chunk_size, 
                                                                                                  self.batch[2, self.batch_counter[0]]+self.offset:self.batch[2, self.batch_counter[0]]+self.offset+self.chunk_size, 
                                                                                                  self.batch[3, self.batch_counter[0]]+self.offset:self.batch[3, self.batch_counter[0]]+self.offset+self.chunk_size
                                                                                                  ]) ),
            (self.Csub, T.set_subtensor(self.Csub[:,:,:,self.index[0]], self.segmentations[self.batch[0, self.batch_counter[0]],
                                                                                           self.batch[1, self.batch_counter[0]]+self.offset:self.batch[1, self.batch_counter[0]]+self.offset+self.chunk_size, 
                                                                                           self.batch[2, self.batch_counter[0]]+self.offset:self.batch[2, self.batch_counter[0]]+self.offset+self.chunk_size, 
                                                                                           self.batch[3, self.batch_counter[0]]+self.offset:self.batch[3, self.batch_counter[0]]+self.offset+self.chunk_size
                                                                                           ]) ),
            (self.batch_counter, self.batch_counter+1)
        ]
        
        if self.batch_size > 1:
            outputs, x_updates = theano.scan(lambda:scan_vals, n_steps=self.batch_size)
            self.updates = x_updates + self.updates 
        else:
            self.updates = scan_vals + self.updates

    
    """
    Define the function(s) for GPU training
    """
    def __theano_training_model(self):
        
        self.updates = []
         
        self.__load_batch()
    
        self.__set_cost()
    
        self.__perform_updates()
    
        self.__set_log()
    
        self.train_network = theano.function(inputs=[], outputs=self.out,
                                             updates = self.updates,
                                             givens = [(self.X, self.Xsub),
                                                       (self.Y, self.Ysub),
                                                       (self.C, self.Csub)],
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
        self.out = network.out
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
        self.cost_func = kwargs.get('cost_func', 'MSE')
        trainer_status += "cost function = "+ `self.cost_func` +"\n"
            
        self.learning_method = kwargs.get('learning_method', 'standardSGD')
        trainer_status += "learning method = "+self.learning_method+"\n"
        self.batch_size = kwargs.get('batch_size', 100)
        self.chunk_size = kwargs.get('chunk_size', 1)
        if (self.cost_func == 'rand') and (self.chunk_size <= 1):
            self.chunk_size = 10
        trainer_status += "batch size = "+`self.batch_size`+"\n"
        trainer_status += "chunk size = "+`self.chunk_size`+"\n"
        
        self.lr = kwargs.get('learning_rate', 0.0001)
        trainer_status += "learning rate = "+`self.lr`+"\n"
        
        self.b1 = kwargs.get('beta1', 0.9)
        if(self.learning_method=='ADAM'): trainer_status += "beta 1 = "+`self.b1`+"\n"
        
        self.b2 = kwargs.get('beta2', 0.9)
        self.damp = kwargs.get('damping', 1.0e-08)
        if(self.learning_method=='RMSprop') or (self.learning_method=='ADAM'): 
            trainer_status += "beta 2 = "+`self.b2`+"\n"
            trainer_status += "damping term = "+`self.damp`+"\n"
        self.load_folder = kwargs.get('trainer_folder', None)
            
        self.log_interval = kwargs.get('log_interval', 100)
        self.log_folder = kwargs.get('log_folder', '')
        
        self.seg = int(self.net_shape.shape[0]*(self.net_shape[0,1] -1))+self.chunk_size
        self.offset = (self.seg-self.chunk_size)/2    
        self.input_shape = (self.seg, self.seg, self.seg, self.batch_size)
        self.output_shape = (3, self.chunk_size, self.chunk_size, self.chunk_size, self.batch_size)

         
        self.log_file = self.log_folder + 'trainer.log'
        self.__clear_log()
        self.__init_lc()
        logging.basicConfig(filename=self.log_file, level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.logger.info(trainer_status+"\n")    
        
        ftensor5 = T.TensorType(theano.config.floatX, (False,)*5)   
        self.Y = ftensor5('Y')
        self.C = T.tensor4('C')
        
        if (self.cost_func == 'rand') and not train_seg:
            print "ERROR: Training with MALIS requires groundtruth segmentation.\n"
            exit(0)
        self.__load_training_data(train_data, train_labels, train_seg)
        
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
            self.segmentations = ()
            self.data_size = ()
            self.num_dsets = 0
            for data, labels, seg in zip(train_set, train_labels, train_seg):
                self.training_data += (data[()] ,)
                self.training_labels += (labels[()] ,)
                self.segmentations += (seg[()].astype(np.intc, order='F') ,)
                self.data_size += (self.training_data[-1].shape[-1] ,)
                self.num_dsets += 1
        else:
                self.training_data = (train_set[()] ,)
                self.training_labels = (train_labels[()] ,)
                self.segmentations = (train_seg[()].astype(np.intc, order='F') ,)
                self.data_size = (self.training_data[-1].shape[-1] ,)
                self.num_dsets = 1
        
        self.training_data = theano.shared(np.asarray(self.training_data), name='data')
        self.training_labels = theano.shared(np.asarray(self.training_labels), name='labels')  
        self.segmentations = theano.shared(np.asarray(self.segmentations), name='seg')
                
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
                            if(self.training_labels.get_value(borrow=True)[n][:,i+self.offset, j+self.offset, k+self.offset].sum() == 0):
                                negative_samples += [[i, j, k]]
                
                self.negatives += (np.asarray(negative_samples, dtype='int32') ,)
                self.epoch_length += (self.data_size[n]-self.seg)**3
                
            self.epoch_length = self.epoch_length/(self.log_interval*self.batch_size*(self.chunk_size**3))
    
    
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
    Store the trainining and weight values at regular intervals
    """
    def __store_status(self, error):
         
        weights_folder = self.log_folder + 'weights/' 
        error_file = self.cost_func + '_learning.csv'
        trainer_folder = self.log_folder + 'trainer/'+self.cost_func+'/'
            
        with open(self.log_folder + error_file, 'ab') as ef:
            fw = csv.writer(ef, delimiter=',')
            fw.writerow([error])
        
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
        if not os.path.isfile(self.log_folder + self.cost_func + '_learning.csv'):
            open(self.log_folder + self.cost_func + '_learning.csv', 'w').close()

    
    
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
        trainer_status += "Number of examples seen: "+`self.batch_size*self.log_interval*self.epoch_length*(self.chunk_size**3)`+"\n"
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
        epoch = 0
        while(epoch < duration):
            if(print_updates): print 'Epoch:',epoch
            epochstart = time.clock()
            for i in range(self.epoch_length):
                   
                self.reset_counter()
                self.__set_batches()
                
                for j in range(0, self.log_interval):
                    self.train_network()
                
                train_error[i] = np.mean(self.error.get_value(borrow=True))
                
                if print_updates:
                    print self.cost_func,'error for updates',i*self.log_interval,' - ',(i+1)*self.log_interval,'('+`self.batch_size*self.log_interval*(self.chunk_size**3)`+' examples):',train_error[i]
                self.__store_status(train_error[i])
            epoch_time = time.clock() - epochstart
            
            self.__log_trainer(epoch, train_error, epoch_time)
            
            if early_stop and (train_error[train_error > 0][-1] < 0.002):
                  epoch = duration
            
            epoch += 1
            
        return train_error
            
