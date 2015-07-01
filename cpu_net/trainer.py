# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 06:10:40 2015

@author: schurterb

Trainer class for a CNN using stochastic gradient descent.
"""


import theano
import theano.tensor as T
import numpy as np

theano.config.floatX = 'float32'


class Trainer(object):
    
    """Define the updates to be performed each training round"""
    def __set_updates(self, learning_method):
        
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
    """def __init__(self, network, **kwargs):
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
        self.dr = kwargs.get('decay_rate', 0.99)
        trainer_status += "decay rate = "+`self.dr`+"\n"
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
        
        #Initialize the list of updates to be performed
        self.__set_updates(learning_method)
        
        #Initialize the training function
        self.train_model = theano.function(inputs=[self.X, self.Y], outputs=self.cost, updates=self.updates, allow_input_downcast=True)
    
    
    """
    Randomly select a batch_size array of samples to train for a given update.
    """
    def __get_samples(self, train_set, train_labels):
          
        Ytemp = np.zeros(3)
        samples = np.zeros(self.batch_size, dtype = 'int32')
        for i in range(0, self.batch_size):
            #For this update, randomly select whether this will be a positive or
            # a negative example
            sel = self.rng.randn(1)
            if (sel > 0):
                sel = 1
            else:
                sel = 0
            #Draw a random sample to train on
            new_sample = self.rng.randint(0, train_set.shape[-1]-self.seg, 3)
            xpos = new_sample[0]
            ypos = new_sample[1]
            zpos = new_sample[2]
            Ytemp = train_labels[:, xpos+self.offset, ypos+self.offset, zpos+self.offset]
            while ((Ytemp.sum() > 0) and (sel == 0)) or ((Ytemp.sum() == 0) and (sel == 1)):    #Set the first half to be negative examples
                new_sample = self.rng.randint(0, train_set.shape[-1]-self.seg, 3)
                xpos = new_sample[0]
                ypos = new_sample[1]
                zpos = new_sample[2]
                Ytemp = train_labels[:, xpos+self.offset, ypos+self.offset, zpos+self.offset]
            
            check = np.sum(self.prev_pos == [xpos, ypos, zpos], 1)
            if np.sum(check == 3):
                samples[i] = np.where(check == 3)[0][0]
            else:
                samples[i] = self.num_examples
                self.Xsub[:,:,:, samples[i]] = train_set[xpos:xpos+self.seg, ypos:ypos+self.seg, zpos:zpos+self.seg]
                self.Ysub[:, samples[i]] = Ytemp
                self.num_examples += 1
                
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
    def train(self, train_set, train_labels, duration, early_stop = False):

        #Variables to keep track of previously seen training examples
        self.prev_pos = np.zeros((self.batch_size*duration, 3))
        self.Xsub = np.zeros((self.seg, self.seg, self.seg, self.batch_size*duration))
        self.Ysub = np.zeros((3, self.batch_size*duration))
        self.num_examples = 0

        #Epochs to average over for early stopping
        averaging_len = 100
        if(early_stop):
            early_stop = 0.0001
        else:
            early_stop = -1
        
        train_error = np.zeros(duration)
        
        epoch = 0
        while(epoch < duration):
            indeces = self.__get_samples(train_set, train_labels)
            
            if(self.use_batches):
                train_error[epoch] = self.train_model(self.Xsub[:,:,:,indeces], self.Ysub[:,indeces])
            else:
                epoch_error = np.zeros(self.batch_size)
                for i in indeces:
                    epoch_error[i] = self.train_model(self.Xsub[:,:,:,i:i+1], self.Ysub[:,i:i+1])
                train_error[epoch] = np.mean(epoch_error)
            
            if(self.print_updates):
                print 'Cost at update '+`epoch`+': '+`train_error[epoch]`
            
            if((epoch+1)%self.log_intervals == 0):
                self.__log_status(train_error[train_error > 0])
                
            if (epoch%averaging_len == 0) and (epoch >= averaging_len*2):
                error_diff = np.mean(train_error[epoch - averaging_len*2:epoch-averaging_len]) - np.mean(train_error[epoch - averaging_len:epoch])
                if(np.abs(error_diff) < early_stop):
                  epoch = duration
            
            epoch += 1
            
        return train_error
            
