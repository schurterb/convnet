# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 10:39:19 2015

@author: schurterb

Analyzer class to perform a basic analysis of the results from training and 
 testing a convolutional network
"""

from analysis import showStats
from scipy.io import loadmat

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import h5py

class Analyzer(object):

    """Load the learning curve from a folder"""
    def __load_results(self, results_folder, lc_name='learning_curve', pred_name='prediction'):
        try:
            try:
                f = open(results_folder + lc_name + '.csv', 'r')
                lc = f.read()
                f.close()
                
                lc = lc.split('\r\n')
                tmp = lc[0].split(',')
                lc = np.append(tmp, lc[1::])
                lr_curve = np.zeros(lc.size)
                for i in range(lc.size):
                    if(lc[i] != ''):
                        lr_curve[i] = float(lc[i])
                
                self.learning_curve += (np.asarray(lr_curve) ,)
            except:
                self.learning_curve += (np.genfromtxt(results_folder + lc_name + '.csv', delimiter=',') ,)
        except:
            self.learning_curve += (None ,)
            print 'Error: Unable to load learning curve.'

        try:
            self.pred_file += (h5py.File(results_folder + pred_name + '.h5', 'r') ,)
            pred = self.pred_file[-1]['main'][...].astype('d', order='F')
            self.prediction += (np.transpose(pred).astype(dtype='d', order='F') ,)
#            pred_shape = self.pred_file[-1]['main'].shape
#            self.prediction += (self.pred_file[-1]['main'][...].reshape((pred_shape[1],pred_shape[2],pred_shape[3],3)).astype(dtype='d', order='F') ,)
        except:
            self.prediction += (None ,)
            print 'Error: Unable to load test prediction.'
            
    
    def __init__(self, **kwargs):
        
        self.nsteps = kwargs.get('threshold_steps', 10) 
        self.crop = kwargs.get('image_crop', 0)
        
        self.tp = []
        self.fp = []
        self.npos = []
        self.nneg = []
        self.fscore = []
        self.randIndex = []
        self.theta = []
        
        self.tpr = []
        self.fpr = []
        
        folder = kwargs.get('results_folder', None)
        self.results_folder = ()
        self.pred_file = ()
        self.learning_curve = ()
        self.prediction = ()
                
        self.target = kwargs.get('target', None)
        targ = self.target[...].astype(dtype='d', order='F')
        self.target = np.transpose(targ).astype(dtype='d', order='F')
#        targ_shape = self.target.shape
#        self.target = self.target[...].reshape((targ_shape[1],targ_shape[2],targ_shape[3],3)).astype(dtype='d', order='F')
        self.raw = kwargs.get('raw', None)
        
        if (folder != None) and (self.target != None):
            self.results_folder += (folder ,)
            self.name = (kwargs.get('name', '') ,)
            self.__load_results(self.results_folder[-1])
            self.__threshold_scan(self.results_Folder[-1])
        else:
            self.name = ()            
            
        
    """Add more than one set of results to analyze at a time""" 
    def add_results(self, **kwargs):
                
        self.results_folder += (kwargs.get('results_folder', None) ,)
        prediction_name = kwargs.get('prediction_file', None)
        learning_name = kwargs.get('learning_curve_file', None)
        if (prediction_name == None) and (learning_name == None):
            self.__load_results(self.results_folder[-1])
        elif (learning_name == None):
            self.__load_results(self.results_folder[-1], pred_name = prediction_name)
        elif (prediction_name == None):
            self.__load_results(self.results_folder[-1], lc_name = learning_name)
        else:
            self.__load_results(self.results_folder[-1], learning_name, prediction_name)
        
        self.name += (kwargs.get('name', '') ,)
        
        analyze_prediction = kwargs.get('analyze', True)
        if analyze_prediction and (self.target != None) and (self.prediction[-1] != None):
            self.__threshold_scan(self.results_folder[-1])
            

    """Plots the learning curves"""
    def learning(self, averaging_segment=1):  
        num_curves = len(self.learning_curve)
        n_figs = num_curves/7   #There can be seven unique auto-generated line colors
        rem_curves = num_curves%7
        
        assert len(self.name) == num_curves
        for f in range(0, n_figs):
            plt.figure()
            for i in range(0, 7):
                if (self.learning_curve[f*7 + i] != None):
                    idx = self.learning_curve[f*7 + i].size/averaging_segment
                    lc = np.mean(self.learning_curve[f*7 + i][0:idx*averaging_segment].reshape(idx, averaging_segment), 1)
                    plt.plot(lc, label=self.name[f*7 + i])
            plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=2, borderaxespad=0., prop={'size':20})
            plt.grid()
        if(rem_curves > 0):
            plt.figure()
            for i in range(0, len(self.learning_curve[-rem_curves::])):
                if (self.learning_curve[n_figs*7 + i] != None):
                    idx = self.learning_curve[n_figs*7 + i].size/averaging_segment
                    lc = np.mean(self.learning_curve[n_figs*7 + i][0:idx*averaging_segment].reshape(idx, averaging_segment), 1)
                    plt.plot(lc, label=self.name[n_figs*7 + i])
            plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=2, borderaxespad=0., prop={'size':20})
            plt.grid()

    
    
    """Plots the data from threshold_scan"""
    def performance(self):
        
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)        
        #fig.set_facecolor('white')
        
        for i in range(len(self.results_folder)):
            mData = loadmat(self.results_folder[i] + 'errors_new.mat')
        
            #Show rand Index results
            rTheta = mData.get('r_thresholds')[0]
            rErr = mData.get('r_fscore')[0]
            rFPR = mData.get('r_fp')[0]/mData.get('r_neg')[0]
            rTPR = mData.get('r_tp')[0]/mData.get('r_pos')[0]
        
            ax1.plot(rTheta,rErr,label=self.name[i])
            ax1.set_title('Rand F-Score', fontsize=20)
            ax1.set_ylabel('f-score', fontsize=20)
            ax1.set_xlabel('threshold', fontsize=20)
            
            ax2.plot(rFPR, rTPR,label=self.name[i])
            ax2.set_ylim([0,1])
            ax2.set_title('Rand ROC', fontsize=20)
            ax2.set_ylabel('true-positive rate', fontsize=20)
            ax2.set_xlabel('false-positive rate', fontsize=20)
            
            #Show pixel error results
            pTheta = mData.get('p_thresholds')[0]
            pErr = mData.get('p_err')[0]
            pFPR = mData.get('p_fp')[0]/mData.get('p_neg')[0]
            pTPR = mData.get('p_tp')[0]/mData.get('p_pos')[0]
        
            ax3.plot(pTheta,pErr,label=self.name[i])
            ax3.set_title('Pixel Error', fontsize=20)
            ax3.set_ylabel('pixel error', fontsize=20)
            ax3.set_xlabel('threshold', fontsize=20)
        
            ax4.plot(pFPR, pTPR,label=self.name[i])
            ax4.set_ylim([0,1])
            ax4.set_title('Pixel ROC', fontsize=20)
            ax4.set_ylabel('true-positive rate', fontsize=20)
            ax4.set_xlabel('false-positive rate', fontsize=20)
            
        ax1.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=2, borderaxespad=0., prop={'size':20})
        ax1.grid(); ax2.grid(); ax3.grid(); ax4.grid();
        
        return;
  
    
    """Displays three images: the raw data, the corresponding labels, and the predictions"""
    def display(self, result_name, equal_indexing=False):
        
        max_crop = 0
        if equal_indexing:
            for res in range(len(self.name)):
                crop = (self.target.shape[-1] - self.prediction[res].shape[-1])/2
                if(crop > max_crop):
                    max_crop = crop        

            if (type(result_name) == str):
                res = self.name.index(result_name)
            elif (type(result_name) == int) and (result_name < len(self.prediction)):
                res = result_name
            else:
                print 'Invalid Entry'
                return;
                
            fig = plt.figure(figsize=(20,10))
            fig.set_facecolor('white')
            ax1 = fig.add_subplot(1,3,1)
            ax2 = fig.add_subplot(1,3,2)
            ax3 = fig.add_subplot(1,3,3)
            fig.subplots_adjust(left=0.25, bottom=0.25)
            depth0 = 0
            
            crop = (self.target.shape[-1] - self.prediction[res].shape[-1])/2            
            
            #All these images are in gray-scale
            plt.gray()
            im1 = ax1.imshow(self.raw[0,max_crop:-max_crop,max_crop:-max_crop])
            ax1.set_title('raw image', fontsize = 20)
            
            im2 = ax2.imshow(self.target[0,0,max_crop:-max_crop,max_crop:-max_crop])
            ax2.set_title('groundtruth', fontsize = 20)
            
            im3 = ax3.imshow(self.prediction[res][0,max_crop-crop,:,:])
            ax3.set_title('prediction', fontsize = 20)     
            
            axcolor = 'lightgoldenrodyellow'
            axdepth = fig.add_axes([0.25, 0.3, 0.65, 0.03], axisbg=axcolor)
            depth = Slider(axdepth, 'Min', 0, self.prediction[res].shape[-1]-max_crop, valinit=depth0, valfmt='%0.0f')
            
            def update(val):
                zlayer = int(depth.val)
                im1.set_data(self.raw[max_crop+zlayer,max_crop:-max_crop,max_crop:-max_crop])
                im2.set_data(self.target[max_crop+zlayer,max_crop:-max_crop,max_crop:-max_crop,:])
                im3.set_data(self.prediction[res][max_crop-crop+zlayer,:,:,:])
                fig.canvas.draw()
                
            fig.suptitle(self.name[res], fontsize = 20)
            depth.on_changed(update)        
            plt.show()
        else:
            if (type(result_name) == str):
                res = self.name.index(result_name)
            elif (type(result_name) == int) and (result_name < len(self.prediction)):
                res = result_name
            else:
                print 'Invalid Entry'
                return;
                
            fig = plt.figure(figsize=(20,10))
            fig.set_facecolor('white')
            ax1 = fig.add_subplot(1,3,1)
            ax2 = fig.add_subplot(1,3,2)
            ax3 = fig.add_subplot(1,3,3)
            fig.subplots_adjust(left=0.25, bottom=0.25)
            depth0 = 0
            
            crop = (self.target.shape[0] - self.prediction[res].shape[0])/2        
            
            #All these images are in gray-scale
            plt.gray()
            im1 = ax1.imshow(self.raw[depth0 ,crop:-crop,crop:-crop])
            ax1.set_title('raw image', fontsize = 20)
            
            im2 = ax2.imshow(self.target[depth0 ,crop:-crop,crop:-crop,:])
            ax2.set_title('groundtruth', fontsize = 20)
            
            im3 = ax3.imshow(self.prediction[res][depth0 ,:,:,:])
            ax3.set_title('prediction', fontsize = 20)     
            
            axcolor = 'lightgoldenrodyellow'
            axdepth = fig.add_axes([0.25, 0.3, 0.65, 0.03], axisbg=axcolor)
            depth = Slider(axdepth, 'Min', 0, self.prediction[res].shape[-1]-crop, valinit=depth0, valfmt='%0.0f')
            
            def update(val):
                zlayer = int(depth.val)
                im1.set_data(self.raw[crop:-crop,crop:-crop,crop+zlayer])
                im2.set_data(self.target[crop+zlayer,crop:-crop,crop:-crop,:])
                im3.set_data(self.prediction[res][zlayer,:,:,:])
                fig.canvas.draw()
                
            fig.suptitle(self.name[res], fontsize = 20)
            depth.on_changed(update)        
            plt.show()

            
