# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 06:10:40 2015

@author: schurterb

Trainer function for a CNN.
"""


import theano
import theano.tensor as T
import numpy as np

theano.config.floatX = 'float32'


class Trainer(object):
    
    """Define the cost function for calculating the updates"""
    def __set_cost(self):
        if (self.cost_func == 'class'):
            self.cost = T.mean(T.nnet.binary_crossentropy(self.out, self.Y.dimshuffle('x',0,'x','x','x')))
        else:
            self.cost = T.mean(1/2.0*((self.out - self.Y.dimshuffle('x',0,'x','x','x'))**2)) 
            
    
    """Define the updates to be performed each training round"""
    def __set_updates(self, learning_method):
        
        w_grads = T.grad(self.cost, self.w)       
        b_grads = T.grad(self.cost, self.b)
        
        if(learning_method == 'RMSprop'):
            
            #Initialize shared variable to store MS of gradient btw updates
            self.rw = ()
            self.rb = ()
            for layer in range(0, len(self.w)):
                self.rw = self.rw + (theano.shared(np.ones(self.w[layer].shape, dtype=theano.config.floatX)) ,)
                self.rb = self.rb + (theano.shared(np.ones(self.w[layer].shape[0], dtype=theano.config.floatX)) ,)      
            
            rw_updates = [
                (r, (self.dr*r) + (1-self.dr)*grad**2)
                for r, grad in zip(self.rw, w_grads)                   
            ]
            rb_updates = [
                (r, (self.dr*r) + (1-self.dr)*grad**2)
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
                self.mw = self.mw + (theano.shared(np.zeros(self.w[layer].shape, dtype=theano.config.floatX)) ,)
                self.mb = self.mb + (theano.shared(np.zeros(self.w[layer].shape[0], dtype=theano.config.floatX)) ,)
                self.vw = self.vw + (theano.shared(np.ones(self.w[layer].shape, dtype=theano.config.floatX)) ,)
                self.vb = self.vb + (theano.shared(np.ones(self.w[layer].shape[0], dtype=theano.config.floatX)) ,)
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
    Network must be a list constaining the key components of the network
     to be trained, namely its symbolic theano reprentation (first parameter),
     its cost function (second parameter), its shared weight and bias 
     variables (second and third parameters, rspectively)
    """
    def __init__(self, network, **kwargs):
        #Network parameters
        self.X = network[0]
        self.Y = network[1]
        self.out = network[2]
        self.w = network[3]
        self.b = network[4]
        self.net_shape = network[5]
        
        #Trainint parameters
        learning_method = kwargs.get('learning_method', 'standardSGD')
        self.rng = kwargs.get('rng', np.random.RandomState(42))
        self.batch_size = kwargs.get('batch_size', 1)
        self.use_batches = kwargs.get('use_batches', True)
        self.cost_func = kwargs.get('cost_func', 'MSE')
        self.lr = kwargs.get('learning_rate', 0.0001)
        self.dr = kwargs.get('decay_rate', 0.99)
        self.damp = kwargs.get('damping', 1.0e-08)
        self.b1 = kwargs.get('beta1', 0.9)
        self.b2 = kwargs.get('beta2', 0.999)
        
        self.seg = int(self.net_shape.shape[0]*(self.net_shape[0,1] -1) +1)
        self.offset = (self.seg -1)/2    
        self.input_shape = (self.seg, self.seg, self.seg, self.batch_size)
        self.output_shape = (3, self.batch_size)
     
        
        #Initialize the cost function and the list of updates to be performed
        self.__set_cost()
        self.__set_updates(learning_method)
        
        #Initialize the training function
        self.train_model = theano.function(inputs=[self.X, self.Y], outputs=self.cost, updates=self.updates, allow_input_downcast=True)
    
    """
    Randomly select a batch_size array of samples to train for a given update.
    """
    def __get_samples(self, train_set, train_labels):
        #Keep count of the number of positive and negative examples to keep their
        # ratios even per 
        num_pos = 0
        num_neg = 0    
        Xsub = np.zeros(self.input_shape)
        Ysub = np.zeros(self.output_shape)
        for i in range(0, self.batch_size):
            #For this update, randomly select whether this will be a positive or
            # a negative example
            sel = self.rng.randn(1)
            if (sel > 0) and (num_pos/self.batch_size < 0.5):
                sel = 1
                num_pos += 1
            else:
                sel = 0
                num_neg += 1
            #Draw a random sample to train on
            new_sample = self.rng.randint(0, train_set.shape[-1]-self.seg, 3)
            xpos = new_sample[0]
            ypos = new_sample[1]
            zpos = new_sample[2]
            Ysub[:, i] = train_labels[:, xpos+self.offset, ypos+self.offset, zpos+self.offset]
            while ((Ysub[:, i].sum() > 0) and (sel == 0)) or ((Ysub[:, i].sum() == 0) and (sel == 1)):
                new_sample = self.rng.randint(0, train_set.shape[-1]-self.seg, 3)
                xpos = new_sample[0]
                ypos = new_sample[1]
                zpos = new_sample[2]
                Ysub[:, i] = train_labels[:, xpos+self.offset, ypos+self.offset, zpos+self.offset]
                
            Xsub[:,:,:,i] = train_set[xpos:xpos+self.seg, ypos:ypos+self.seg, zpos:zpos+self.seg]
                
        return Xsub, Ysub
    
    
    """   
    Train the network on a specified training set with a specified target
     (supervised training). This training uses stochastic gradient descent
     mini-batch sizes set at initialization. Training samples are selected 
     such that there is an equal number of positive and negative samples 
     each batch.        
    Returns: network cost at each update
    """
    def train(self, train_set, train_labels, duration, early_stop = None):

        #Epochs to average over for early stopping
        averaging_len = 100
        if(early_stop == None):
            early_stop = -1
        
        train_error = np.zeros(duration)
        
        epoch = 0
        while(epoch < duration):
            
            Xsub, Ysub = self.__get_samples(train_set, train_labels)
                
            if(self.use_batches):
                train_error[epoch] = self.train_model(Xsub, Ysub)
            else:
                epoch_error = np.zeros(self.batch_size)
                for i in range(0, self.batch_size):
                    epoch_error[i] = self.train_model(Xsub[:,:,:,i], Ysub[:,i])
                train_error[epoch] = np.mean(epoch_error)
                
            #If the current error is sufficiently better than the original error, than
            # let us stop
            if (epoch%averaging_len == 0) and (epoch >= averaging_len*2):
                error_diff = np.mean(train_error[epoch - averaging_len*2:epoch-averaging_len]) - np.mean(train_error[epoch - averaging_len:epoch])
                if(np.abs(error_diff) < early_stop):
                  epoch = duration
            
            epoch += 1
            
        return train_error
            
