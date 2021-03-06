# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 06:35:06 2015

@author: schurterb

Definition for creation of convolutional neural network
"""
import sys
import os

import theano
import theano.sandbox.cuda
from theano.sandbox.cuda.basic_ops import gpu_from_host
from theano import tensor as T
from theano import Out
import numpy as np
from theano.tensor.nnet.conv3d2d import conv3d
import h5py


theano.config.floatX = 'float32'


class CNN(object): 
    
    """Define the shape of the network as a 2D array"""
    def __define_network(self):
        self.sample_size = self.num_layers*(self.filter_size-1) + 1
        #Define the network shape
        self.net_shape = np.ndarray([self.num_layers, 5])
        #Define the first layer
        self.net_shape[0,:] = [self.num_filters, self.filter_size, 1, self.filter_size, self.filter_size]
        #Define all internal layers
        for i in range(1, self.num_layers-1):    #Network cannot be smaller than 2 layers
            self.net_shape[i,:] = [self.num_filters, self.filter_size, self.num_filters, self.filter_size, self.filter_size]
        #Define the output layer            
        self.net_shape[-1,:] = [3, self.filter_size, self.num_filters, self.filter_size, self.filter_size]
        
          
    """Randomly initialize the weights"""
    def __init_weights(self):

        #Initialize the filters and biases (with random values)
        im_size =  self.num_layers*(self.filter_size - 1) + 1
        self.w = ()  #Weights
        self.b = ()  #Biases
        for layer in range(0,  self.num_layers):
            #The weights are initialized within the optimum range for a tanh
            # activation function
            fan_in = self.net_shape[layer, 2] * (im_size**3)
            fan_out = self.net_shape[layer, 0] * (self.net_shape[layer, 1]**3)
            bound =np.sqrt(6. / (fan_in + fan_out))
            self.w  += (theano.shared(np.asarray(self.rng.uniform(low= -bound, high= bound, size= self.net_shape[layer, :]), dtype=theano.config.floatX), name='w'+`layer`) ,)
            self.b += (theano.shared(np.asarray(np.ones(self.net_shape[layer, 0]), dtype=theano.config.floatX), name='b'+`layer`) ,)
            im_size = im_size - self.filter_size + 1
        
        
    """
    Initialize the weights based on an input file
    If there are not enough weight files for the layers of the network, randomly
    initialize the unaccounted for layers.    
    """
    def __load_weights(self):
        
        #Initialize the filters and biases with stored values
        weights = ()
        biases = ()
        found_layer = True
        self.num_layers = 0
        while found_layer:
            try:
                weights += (np.genfromtxt(self.load_folder + 'layer_'+`self.num_layers`+'_weights.csv', delimiter=',') ,)
                biases += (np.genfromtxt(self.load_folder + 'layer_'+`self.num_layers`+'_bias.csv', delimiter=',') ,)
                self.num_layers += 1
            except:
                found_layer = False
                
        assert len(weights) == len(biases)
        
        self.num_filters = biases[0].size
        
        self.filter_size = int(np.round((weights[0].size/self.num_filters) ** (1/3.0)))
        
        self.__define_network()
                
        self.w = ()  
        self.b = () 
        for layer in range(0, self.num_layers):
            self.w += (theano.shared(np.asarray(weights[layer].reshape(self.net_shape[layer,:]), dtype=theano.config.floatX), name='w'+`layer`) ,)
            self.b += (theano.shared(np.asarray(biases[layer].reshape(self.net_shape[layer,0]), dtype=theano.config.floatX), name='b'+`layer`) ,)
    
    
    """Save the weights to an output file"""
    def save_weights(self, folder):
        for i in range(0, self.net_shape.shape[0]):
            self.w[i].get_value().tofile(folder + 'layer_'+`i`+'_weights.csv', sep=',')
            self.b[i].get_value().tofile(folder + 'layer_'+`i`+'_bias.csv', sep=',')
            
            
    """Define the network model"""
    def __model(self):

        #Customizable non-linearity, added 6/8/15
        #Default is the hyperbolic tangent

        #Prepare input tensor
        Xin = self.X.dimshuffle(3, 0, 'x', 1, 2)

        if((self.activation == 'sig') or (self.activation == 'Sig')): #Also include the option of only sigmoidal non-linear units
            #Layer 1: input layer
            out = T.nnet.sigmoid(conv3d(Xin, self.w[0], border_mode='valid') + self.b[0].dimshuffle('x','x',0,'x','x'))
                
            #Every other layer in the network, as definied by filter_shapes
            for layer in range(1, self.net_shape.shape[0]-1):
                out = T.nnet.sigmoid(conv3d(out, self.w[layer], border_mode='valid') + self.b[layer].dimshuffle('x','x',0,'x','x'))
           
        elif((self.activation == 'relu') or (self.activation == 'ReLU')): 
            #Layer 1: input layer
            out = T.maximum(conv3d(Xin, self.w[0], border_mode='valid') + self.b[0].dimshuffle('x','x',0,'x','x'), 0)
                
            #Every other layer in the network, as definied by filter_shapes
            for layer in range(1, self.net_shape.shape[0]-1):
                #An attempt to eliminate the nan errors by normalizing the relu outputs before sending them to the sigmoid function (added 6/15/15)
                out = T.maximum(conv3d(out, self.w[layer], border_mode='valid') + self.b[layer].dimshuffle('x','x',0,'x','x'), 0)
          
        else: #nonlin == 'tanh'
            #Layer 1: input layer
            out = T.tanh(conv3d(Xin, self.w[0], border_mode='valid') + self.b[0].dimshuffle('x','x',0,'x','x'))
                
            #Every other layer in the network, as definied by filter_shapes
            for layer in range(1, self.net_shape.shape[0]-1):
                out = T.tanh(conv3d(out, self.w[layer], border_mode='valid') + self.b[layer].dimshuffle('x','x',0,'x','x'))
        
        
        out = T.nnet.sigmoid(conv3d(out, self.w[-1], border_mode='valid') + self.b[-1].dimshuffle('x','x',0,'x','x'))          
        #Reshuffle the dimensions so that the last three are the xyz dimensions
        # and the second one is the number of affinity graph types (for each dimension)
        self.out = out.dimshuffle(2, 1, 3, 4, 0)


    """Initialize the network"""
    def __init__(self, **kwargs):
        
        self.num_layers = kwargs.get('num_layers', None)
        self.num_filters = kwargs.get('num_filters', None)
        self.filter_size = kwargs.get('filter_size', None)
        
        self.rng = kwargs.get('rng', np.random.RandomState(42))
        self.load_folder = kwargs.get('weights_folder', None)
        self.activation = kwargs.get('activation', 'relu')
        self.cost_func = kwargs.get('cost_func', 'MSE')  
        
        #Initialize (or load) the weights for the network
        if(self.load_folder == None):
            try:
                assert (self.num_layers != None) and (self.num_filters != None) and (self.filter_size != None)
                self.__define_network()
                self.__init_weights()
            except:
                print "ERROR: Insufficient parameters for generating new network"
                sys.exit(0)
        else:
            self.__load_weights()

        #Input and Target variables for symbolic representation of network
        self.X = T.tensor4('X')            
        
        #Create the network model
        self.__model()
        
        if(theano.config.device == 'cpu'):
            #Create a predicter based on this network model
            self.forward = theano.function(inputs=[self.X], outputs=self.out, allow_input_downcast=True)
        else:
            #Create a predicter based on this network model
            self.forward = theano.function(inputs=[self.X], outputs=Out(gpu_from_host(self.out), borrow=True), allow_input_downcast=True)
            #self.predictor = theano.function(inputs=[self.X], outputs=self.out, allow_input_downcast=True)
            #Create a function to calculate the loss of this network

            
           
       
    """
    Make a prediction on a set of inputs
    TODO: Test this...
    """
    def predict(self, x, results_folder = '', name = 'prediction', chunk_size = 75):
        
        if not os.path.exists(results_folder):
            os.mkdir(results_folder)
        
        if not type(x)==tuple or type(x)==list:
            x = (x,)
        
        for i in range(len(x)):
            xsub = x[i] 
            out_size = xsub.shape[0] - self.sample_size + 1
            rem_size = out_size%chunk_size     
            
            f = h5py.File(results_folder + name + '_'+`i`+'.h5', 'w')
            dset = f.create_dataset('main', (3, out_size, out_size, out_size), dtype='float32')

            subsets = []
            for i in range(0, out_size/chunk_size):  
               subsets.append(chunk_size)
            subsets.append(rem_size)
            
            for i in range(len(subsets)):
                for j in range(len(subsets)):
                    for k in range(len(subsets)):
                        dset[:,sum(subsets[:i]):sum(subsets[:(i+1)]),
                               sum(subsets[:j]):sum(subsets[:(j+1)]),
                               sum(subsets[:k]):sum(subsets[:(k+1)])] \
                               = np.asarray(self.forward(xsub[sum(subsets[:i]):sum(subsets[:(i+1)]) +self.sample_size -1,
                                                              sum(subsets[:j]):sum(subsets[:(j+1)]) +self.sample_size -1, 
                                                              sum(subsets[:k]):sum(subsets[:(k+1)]) +self.sample_size -1] \
                                                              .reshape((subsets[i] +self.sample_size -1, 
                                                                        subsets[j] +self.sample_size -1, 
                                                                        subsets[k] +self.sample_size -1, 1))))[:,:,:,:,0]
                          
            f.close()

 



        