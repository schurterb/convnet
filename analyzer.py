# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 10:39:19 2015

@author: schurterb

Analyzer class to perform a basic analysis of the results from training and 
 testing a convolutional network
"""
import sys
sys.path.append('randIndexOld')
#from visualizeStats import show_stats
#sys.path.append('randIndex')
from evaluateFiles import evaluateFileAtThresholds as evalAffs

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import h5py

class Analyzer(object):

    def __threshold_scan(self, results_folder):
        
        crop = (self.target.shape[0] - self.prediction[-1].shape[0])/2
        
        theta_min = self.prediction[-1].min()
        theta_max = self.prediction[-1].max()
        
        self.theta.append(np.arange(theta_min, theta_max, (theta_max-theta_min)/self.nsteps))
        
        print theta_min
        print theta_max 
        print self.theta[-1]        
        
        argsArr = [self.target[crop:crop+100, crop:crop+100, crop:crop+100, :],
               self.prediction[-1][0:100, 0:100, 0:100, :], self.theta[-1]]
               
        pres = evalAffs(argsArr + ["pixel"])
        rres = evalAffs(argsArr + ["rand"])
        
        self.fscore.append((pres[0],rres[0]))
        self.tp.append((pres[1],rres[1]))
        self.fp.append((pres[2],rres[2]))
        self.npos.append((pres[3],rres[3]))
        self.nneg.append((pres[4],rres[4]))
        
        self.tpr.append((self.tp[-1][0]/self.npos[-1][0], self.tp[-1][1]/self.npos[-1][1]))
        self.fpr.append((self.fp[-1][0]/self.nneg[-1][0], self.fp[-1][1]/self.nneg[-1][1]))
        
        return;
        
        
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
            
            self.prediction += (np.transpose(self.pred_file[-1]['main'][...], (1, 2, 3, 0)).astype(dtype='d', order='F') ,)
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
        self.target = np.transpose(self.target[...], (1, 2, 3, 0)).astype(dtype='d', order='F')
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
        
        #show_stats(self.results_folder, self.name)
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
            depth = Slider(axdepth, 'Min', 0, self.prediction[res].shape[-1]-max_crop, valinit=depth0)
            
            def update(val):
                zlayer = int(depth.val)
                im1.set_data(self.raw[max_crop+zlayer,max_crop:-max_crop,max_crop:-max_crop])
                im2.set_data(self.target[0,max_crop+zlayer,max_crop:-max_crop,max_crop:-max_crop])
                im3.set_data(self.prediction[res][0, max_crop-crop+zlayer,:,:])
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
            im1 = ax1.imshow(self.raw[0,crop:-crop,crop:-crop])
            ax1.set_title('raw image', fontsize = 20)
            
            im2 = ax2.imshow(self.target[0,crop:-crop,crop:-crop,:])
            ax2.set_title('groundtruth', fontsize = 20)
            
            im3 = ax3.imshow(self.prediction[res][0,:,:,:])
            ax3.set_title('prediction', fontsize = 20)     
            
            axcolor = 'lightgoldenrodyellow'
            axdepth = fig.add_axes([0.25, 0.3, 0.65, 0.03], axisbg=axcolor)
            depth = Slider(axdepth, 'Min', 0, self.prediction[res].shape[-1]-crop, valinit=depth0)
            
            def update(val):
                zlayer = int(depth.val)
                im1.set_data(self.raw[crop+zlayer,crop:-crop,crop:-crop])
                im2.set_data(self.target[crop+zlayer,crop:-crop,crop:-crop,:])
                im3.set_data(self.prediction[res][zlayer,:,:,:])
                fig.canvas.draw()
                
            fig.suptitle(self.name[res], fontsize = 20)
            depth.on_changed(update)        
            plt.show()

            
            
#   """Plots the data from threshold_scan"""
#    def performance(self):
#        #Prepare the figure
#        fig = plt.figure()
#        ax1 = fig.add_subplot(2,2,1)
#        ax2 = fig.add_subplot(2,2,2)
#        ax3 = fig.add_subplot(2,2,3)
#        ax4 = fig.add_subplot(2,2,4)
#    
#        #If the inputs are tuples containing multiple curves to plot, cycle through
#        # the different curves, plotting them on the same graph
#        num_curves = len(self.fscore)
#        if self.name[0] != None:
#            assert len(self.name) == num_curves
#            for i in range(0, num_curves):
#                #Rand ROC curve, first
#                ax1.plot(self.fpr[i][1][0], self.tpr[i][1][0], label=self.name[i])
#                ax1.axis([0, 1, 0, 1])
#                #ax1.annotate('theta='+`theta[th_mid]`, (fp_rate[th_mid], tp_rate[th_mid]))
#                ax1.set_title('Rand ROC')
#                ax1.set_ylabel('true-positive rate')
#                ax1.set_xlabel('false-positive rate')
#                #display the x=y line for comparison
#                x = np.asarray(np.arange(0, 1, 0.1))
#                ax1.plot(x, x, 'k--')
#                
#                #Next plot the Rand Error
#                ax2.plot(self.theta[i], self.fscore[i][1][0], label=self.name[i])
#                ax2.axis([0, 1, 0, 1])
#                ax2.set_title('randIndex')
#                ax2.set_ylabel('Error')
#                ax2.set_xlabel('threshold')
#                
#                #Pixel-wise ROC curve, first
#                ax3.plot(self.fpr[i][0][0], self.tpr[i][0][0], label=self.name[i])
#                ax3.axis([0, 1, 0, 1])
#                #ax1.annotate('theta='+`theta[th_mid]`, (fp_rate[th_mid], tp_rate[th_mid]))
#                ax3.set_title('Pixel ROC')
#                ax3.set_ylabel('true-positive rate')
#                ax3.set_xlabel('false-positive rate')
#                #display the x=y line for comparison
#                x = np.asarray(np.arange(0, 1, 0.1))
#                ax3.plot(x, x, 'k--')
#                
#                #Finally, plot the Pixel Error
#                ax4.plot(self.theta[i], self.fscore[i][0][0], label=self.name[i])
#                ax4.axis([0, 1, 0, 1])
#                ax4.set_title('Pixel Error')
#                ax4.set_ylabel('Error')
#                ax4.set_xlabel('threshold')
#                
#            ax1.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=2, borderaxespad=0., prop={'size':20})
#        else:
#            for i in range(0, num_curves):
#                #Rand ROC curve, first
#                ax1.plot(self.fpr[i][1][0], self.tpr[i][1][0], label='curve '+`i`)
#                ax1.axis([0, 1, 0, 1])
#                #ax1.annotate('theta='+`theta[th_mid]`, (fp_rate[th_mid], tp_rate[th_mid]))
#                ax1.set_title('Rand ROC')
#                ax1.set_ylabel('true-positive rate')
#                ax1.set_xlabel('false-positive rate')
#                #display the x=y line for comparison
#                x = np.asarray(np.arange(0, 1, 0.1))
#                ax1.plot(x, x, 'k--')
#                
#                #Next plot the Rand Error
#                ax2.plot(self.theta[i], self.fscore[i][1][0], label='curve '+`i`)
#                ax2.axis([0, 1, 0, 1])
#                ax2.set_title('randIndex')
#                ax2.set_ylabel('Error')
#                ax2.set_xlabel('threshold')
#                
#                #Pixel-wise ROC curve, first
#                ax3.plot(self.fpr[i][0][0], self.tpr[i][0][0], label='curve '+`i`)
#                ax3.axis([0, 1, 0, 1])
#                #ax1.annotate('theta='+`theta[th_mid]`, (fp_rate[th_mid], tp_rate[th_mid]))
#                ax3.set_title('Pixel ROC')
#                ax3.set_ylabel('true-positive rate')
#                ax3.set_xlabel('false-positive rate')
#                #display the x=y line for comparison
#                x = np.asarray(np.arange(0, 1, 0.1))
#                ax3.plot(x, x, 'k--')
#                
#                #Finally, plot the Pixel Error
#                ax4.plot(self.theta[i], self.fscore[i][0][0], label='curve '+`i`)
#                ax4.axis([0, 1, 0, 1])
#                ax4.set_title('Pixel Error')
#                ax4.set_ylabel('Error')
#                ax4.set_xlabel('threshold')
#                
#            ax1.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=2, borderaxespad=0., prop={'size':20})