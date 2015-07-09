# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 06:35:06 2015

@author: schurterb

Definition for creation of convolutional neural network
"""
import sys

import theano
import theano.sandbox.cuda
from theano.sandbox.cuda.basic_ops import gpu_from_host
from theano import tensor as T
from theano import Out
import numpy as np
from theano.tensor.nnet.conv3d2d import conv3d


theano.config.nvcc.flags='-use=fast=math'
theano.config.allow_gc=False
theano.config.floatX = 'float32'
theano.sandbox.cuda.use('gpu1')


class CNN(object):  
        
    """Randomly initialize the weights"""
    def __init_weights(self):

        #Initialize the filters and biases (with random values)
        im_size =  self.num_layers*(self.filter_size - 1) + 1
        w = ()  #Weights
        b = ()  #Biases
        weight_size = 0
        for layer in range(0,  self.num_layers):
            #The weights are initialized within the optimum range for a tanh
            # activation function
            fan_in = self.net_shape[layer, 2] * (im_size**3)
            fan_out = self.net_shape[layer, 0] * (self.net_shape[layer, 1]**3)
            bound =np.sqrt(6. / (fan_in + fan_out))
            w  += (theano.shared(np.asarray(self.rng.uniform(low= -bound, high= bound, size= self.net_shape[layer, :]), dtype=theano.config.floatX), name='w'+`layer`) ,)
            b += (theano.shared(np.asarray(np.ones(self.net_shape[layer, 0]), dtype=theano.config.floatX), name='b'+`layer`) ,)
            im_size = im_size - self.filter_size + 1
            weight_size += sys.getsizeof(w[-1].get_value(borrow=True)[0,0,0,0,0])*w[-1].get_value(borrow=True).size/(1024.0*1024.0)
            weight_size += sys.getsizeof(b[-1].get_value(borrow=True)[0])*w[-1].get_value(borrow=True).size/(1024.0*1024.0)
            
        #print 'Size of all weights = '+`weight_size`+' MB'
        self.w = w
        self.b = b
        
        
    """
    Initialize the weights based on an input file
    If there are not enough weight files for the layers of the network, randomly
    initialize the unaccounted for layers.    
    """
    def __load_weights(self):
        
        #Initialize the filters and biases with stored values
        w = ()  #Weights
        b = ()  #Biases
        for layer in range(0, self.num_layers):
            try:
                weights = np.genfromtxt(self.load_folder + 'layer_'+`layer`+'_weights.csv', delimiter=',')
                w += (theano.shared(np.asarray(weights.reshape(self.net_shape[layer,:]), dtype=theano.config.floatX), name='w'+`layer`) ,)
                bias = np.genfromtxt(self.load_folder + 'layer_'+`layer`+'_bias.csv', delimiter=',')
                b += (theano.shared(np.asarray(bias.reshape(self.net_shape[layer,0]), dtype=theano.config.floatX), name='b'+`layer`) ,)
            
            except: #If the specific weights file does not exist
                im_size = (self.num_layers - layer)*(self.filter_size -1) + 1
                fan_in = self.net_shape[layer, 2] * (im_size**3)
                fan_out = self.net_shape[layer, 0] * (self.net_shape[layer, 1]**3)
                bound =np.sqrt(6. / (fan_in + fan_out))
                w  += (theano.shared(np.asarray(self.rng.uniform(low= -bound, high= bound, size= self.net_shape[layer, :]), dtype=theano.config.floatX), name='w'+`layer`) ,)
                b += (theano.shared(np.asarray(np.ones(self.net_shape[layer, 0]), dtype=theano.config.floatX), name='b'+`layer`) ,)
            
        self.w = w
        self.b = b
        
        
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
        self.out = out.dimshuffle(0, 2, 1, 3, 4)     
        
        
    """Define the cost function used to evaluate this network"""
    def __set_cost(self):
        if (self.cost_func == 'class'):
            self.cost = T.mean(T.nnet.binary_crossentropy(self.out, self.Y.dimshuffle(1,0,'x','x','x')))
        else:
            self.cost = T.mean(1/2.0*((self.out - self.Y.dimshuffle(1,0,'x','x','x'))**2))        
        
        
    """Initialize the network"""
    def __init__(self, **kwargs):
        
        self.num_layers = kwargs.get('num_layers', 3)
        self.num_filters = kwargs.get('num_filters', 6)
        self.filter_size = kwargs.get('filter_size', 3)
        
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
        
        self.rng = kwargs.get('rng', np.random.RandomState(42))
        self.load_folder = kwargs.get('weights_folder', None)
        self.activation = kwargs.get('activation', 'relu')
        self.cost_func = kwargs.get('cost_func', 'MSE')  
        
        #Initialize (or load) the weights for the network
        if(self.load_folder == None):
            self.__init_weights()
        else:
            self.__load_weights()

        #Input and Target variables for symbolic representation of network
        self.X = T.tensor4('X')
        self.Y = T.matrix('Y')            
            
        #Create the network model
        self.__model()
        
        #Create the cost funciton
        self.__set_cost()
        
        try:
            #Create a predicter based on this network model
            self.predictor = theano.function(inputs=[self.X], outputs=Out(gpu_from_host(self.out), borrow=True), allow_input_downcast=True)
            #self.predictor = theano.function(inputs=[self.X], outputs=self.out, allow_input_downcast=True)
            #Create a function to calculate the loss of this network
            self.eval_cost = theano.function(inputs=[self.X, self.Y], outputs=Out(gpu_from_host(self.cost), borrow=True), allow_input_downcast=True)
            #self.eval_cost = theano.function(inputs=[self.X, self.Y], outputs=self.cost, allow_input_downcast=True)
        except:
            #Create a predicter based on this network model
            self.predictor = theano.function(inputs=[self.X], outputs=self.out, allow_input_downcast=True)
            #Create a function to calculate the loss of this network
            self.eval_cost = theano.function(inputs=[self.X, self.Y], outputs=self.cost, allow_input_downcast=True)
           
       
    """
    Make a prediction on a set of inputs
    Params: x must be a cubic 3D matrix
    Returns: a 4D array where the first dimension is the affinity in each 
           dimension and the last 3 dimensions correspond to the output image
    """
    def predict(self, x):
        out_size = x.shape[0] - self.sample_size + 1
        return np.asarray(self.predictor(x[:,:,:].reshape((x.shape) + (1 ,)))).reshape((3, out_size, out_size, out_size))
        
        
    """
    Calculate the loss of the networks prediction based on a target output
    Params: x must be a cubic 3D matrix consisting of per-pixel input values
            y must be a cubic 3D matrix consisting of per-pixel affinity labels
    Returns: scalar loss value
    """
    def loss(self, x, y, test_batch_size):
        offset = (self.sample_size -1)/2
        
        Ysub = np.zeros((3, test_batch_size))
        Xsub = np.zeros((self.sample_size, self.sample_size, self.sample_size, test_batch_size))
        for i in range(0, test_batch_size):
            new_sample = self.rng.randint(0, x.shape[-1]-self.sample_size, 3)
            xpos = new_sample[0]
            ypos = new_sample[1]
            zpos = new_sample[2]
            Ysub[:, i] = y[:, xpos+offset, ypos+offset, zpos+offset]
            Xsub[:,:,:, i] = x[xpos:xpos+self.sample_size, 
                               ypos:ypos+self.sample_size,
                               zpos:zpos+self.sample_size]
        
        return np.asarray(self.eval_cost(Xsub, Ysub))


    """
    Return the network for training
    Returns a list containing the symbolic definition of the network as well
     as the shared weights and the shared biases
    """
    def get_network(self):
        return [self.X, self.Y, self.out, self.cost, self.w, self.b, self.net_shape]
        



        