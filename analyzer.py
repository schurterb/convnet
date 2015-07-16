# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 10:39:19 2015

@author: schurterb

Analyzer class to perform a basic analysis of the results from training and 
 testing a convolutional network
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import h5py


class Analyzer(object):
    
    """Calculate the number of true positives, false positives, true negatives, etc."""
    def __conf_matrix(self, prediction, target, threshold):
        
        diag1 = ((prediction[...] >= threshold) + 0.0) - ((target > threshold) + 0.0)
        fp = float((diag1 == 1).sum())  #True - False = 1
        fn = float((diag1 == -1).sum()) #False - True = -1
        
        diag2 = ((prediction[...] >= threshold) + 0.0) +  ((target > threshold) + 0.0)
        tp = float((diag2 == 2).sum())  #True + True = 2
        tn = float((diag2 == 0).sum())  #False + False = 0
        
        return tp, tn, fp, fn
        
        
    """Calculate various statistics from the results of the confusion matrix"""
    def __statistics(self, confusion_matrix):
        
        tp = confusion_matrix[0]
        tn = confusion_matrix[1]
        fp = confusion_matrix[2]
        fn = confusion_matrix[3]
        
        accuracy = (tp + tn)/(tp+tn+fp+fn)
        #check for div/0 errors
        if(tp == 0):
            precision = 0
        else:
            precision = tp/(tp+fp)
        
        #check for div/0 errors
        if(tp+fn != 0):
            recall = tp/(tp+fn)
        else:
            recall = 0
        
        #check for div/0 errors
        if(precision+recall != 0):
            Fscore = 2*((precision*recall)/(precision+recall))
        else:
            Fscore = 0
        
        tp_rate = recall
        
        #Check for div/0 errors
        if(fp+tn != 0):
            fp_rate = fp/(fp+tn)
        else:
            fp_rate = 0
        
        return accuracy, precision, recall, Fscore, tp_rate, fp_rate
        
    
    """
    Function to sweep through different threshold values.
    Only to be called after a new set of results has been loaded.    
    """
    #TODO: keep track of which networks we have scanned (as not all may have predictions)
    def __threshold_scan(self):
        crop = (self.target.shape[-1] - self.prediction[-1].shape[-1])/2

        theta_max = self.prediction[-1][...].max()
        theta_min = self.prediction[-1][...].min()
        
        theta_step = (theta_max - theta_min)/(self.nsteps)
        self.theta += (np.asarray(np.arange(theta_min + theta_step, theta_max, theta_step)) ,)
        
        #Variables to store results of analysis:
        self.accuracy += (np.zeros(len(self.theta[-1])) ,)
        self.precision += (np.zeros(len(self.theta[-1])) ,)
        self.recall += (np.zeros(len(self.theta[-1])) ,)
        self.fscore += (np.zeros(len(self.theta[-1])) ,)
        self.tpr += (np.zeros(len(self.theta[-1])) ,)
        self.fpr += (np.zeros(len(self.theta[-1])) ,)
        for i in range(0, len(self.theta[-1])):
            results = self.__statistics(self.__conf_matrix(self.prediction[-1], self.target[:, crop:-crop, crop:-crop, crop:-crop], self.theta[-1][i]))
            self.accuracy[-1][i] = results[0]
            self.precision[-1][i] = results[1]
            self.recall[-1][i] = results[2]
            self.fscore[-1][i] = results[3]
            self.tpr[-1][i] = results[4]
            self.fpr[-1][i] = results[5]
                
            
    
    """Load the learning curve from a folder"""
    def __load_results(self, results_folder, lc_name='learning_curve', pred_name='prediction'):
        try:
            lr_curve = np.genfromtxt(results_folder + lc_name + '.csv', delimiter=',')
            #self.learning_curve += (np.mean(lr_curve[lr_curve > 0].reshape((1,-1)), 0) ,)
            self.learning_curve += (lr_curve ,)
        except:
            self.learning_curve += (None ,)
            print 'Error: Unable to load learning curve.'

        try:
            self.pred_file += (h5py.File(results_folder + pred_name + '.h5', 'r') ,)
            self.prediction += (self.pred_file[-1]['main'] ,)
        except:
            self.prediction += (None ,)
            print 'Error: Unable to load test prediction.'
            
    
    def __init__(self, **kwargs):
        
        self.nsteps = kwargs.get('threshold_steps', 10) 
        self.crop = kwargs.get('image_crop', 0)
        
        self.accuracy = ()
        self.precision = ()
        self.recall = ()
        self.fscore = ()
        self.tpr = ()
        self.fpr = ()
        self.theta = ()
        
        results_folder = kwargs.get('results_folder', None)
        self.pred_file = ()
        self.learning_curve = ()
        self.prediction = ()
                
        self.target = kwargs.get('target', None)
        self.raw = kwargs.get('raw', None)
        
        if (results_folder != None) and (self.target != None):
            self.name = (kwargs.get('name', '') ,)
            self.__load_results(results_folder)
            self.__threshold_scan()
        else:
            self.name = ()

            
            
        
    """Add more than one set of results to analyze at a time""" 
    def add_results(self, **kwargs):
                
        results_folder = kwargs.get('results_folder', None)
        prediction_name = kwargs.get('prediction_file', None)
        learning_name = kwargs.get('learning_curve_file', None)
        if (prediction_name == None) and (learning_name == None):
            self.__load_results(results_folder)
        elif (learning_name == None):
            self.__load_results(results_folder, pred_name = prediction_name)
        elif (prediction_name == None):
            self.__load_results(results_folder, lc_name = learning_name)
        else:
            self.__load_results(results_folder, learning_name, prediction_name)
        
        self.name += (kwargs.get('name', '') ,)
        
        analyze_prediction = kwargs.get('analyze', True)
        if analyze_prediction and (self.target != None) and (self.prediction[-1] != None):
            self.__threshold_scan()
            
        
    """Store the results of an anlysis"""
    def store_analysis(self, results_folder, res_name=None):
        
        if res_name == None:
            index = -1
        else:
            index = np.where(self.name == res_name)[0]
            
        self.accuracy[index].tofile(results_folder + 'accuracy.csv', sep=',')
        self.precision[index].tofile(results_folder + 'precision.csv', sep=',')
        self.recall[index].tofile(results_folder + 'recall.csv', sep=',')
        self.fscore[index].tofile(results_folder + 'fscore.csv', sep=',')
        self.tpr[index].tofile(results_folder + 'true_pos_rate.csv', sep=',')
        self.fpr[index].tofile(results_folder + 'false_pos_rate.csv', sep=',')
        
        
    """Load the results of a previous analysis"""
    def load_analysis(self, results_folder, res_name=None):
        
        if res_name != None:
            self.name += (res_name ,)
        else:
            self.name += ('' ,)
            
        self.accuracy += (np.genfromtxt(results_folder + 'accuracy.csv', delimiter=',') ,)
        self.precision += (np.genfromtxt(results_folder + 'precision.csv', delimiter=',') ,)
        self.recall += (np.genfromtxt(results_folder + 'recall.csv', delimiter=',') ,)
        self.fscore += (np.genfromtxt(results_folder + 'fscore.csv', delimiter=',') ,)
        self.tpr += (np.genfromtxt(results_folder + 'true_pos_rate.csv', delimiter=',') ,)
        self.fpr += (np.genfromtxt(results_folder + 'false_pos_rate.csv', delimiter=',') ,)
        
        
    """Calculate the unthresholded classification error"""
    def error(self):
        try:
            error = np.zeros(len(self.prediction))
            for i in range(0, len(self.prediction)):
                error[i] = np.mean(np.abs(self.prediction[i][:, :, :] - self.target[:, :, :]))
            
            return zip(self.name, error)
        except:
            print 'Error: Cannot calculate prediction error.'
            print 'Either there is no prediction to analyze or no target to compare it to.\n'
        
        
    """Plots the learning curves"""
    def learning(self, averaging_segment):  
        num_curves = len(self.learning_curve)
        n_figs = num_curves/7   #There can be seven unique auto-generated line colors
        rem_curves = num_curves%7
        
        assert len(self.name) == num_curves
        for f in range(0, n_figs):
            plt.figure()
            for i in range(0, 7):
                if (self.learning_curve[i] != None):
                    lc = np.mean(self.learning_curve[f*7 + i].reshape(-1, averaging_segment), 1)
                    plt.plot(lc, label=self.name[f*7 + i])
            plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=2, borderaxespad=0., prop={'size':20})
            plt.grid()
        if(rem_curves > 0):
            plt.figure()
            for i in range(0, len(self.learning_curve[-rem_curves::])):
                if (self.learning_curve[i] != None):
                    lc = np.mean(self.learning_curve[n_figs*7 + i].reshape(-1, averaging_segment), 1)
                    plt.plot(lc, label=self.name[n_figs*7 + i])
            plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=2, borderaxespad=0., prop={'size':20})
            plt.grid()

    
    
    """Plots the data from threshold_scan"""
    def performance(self):
        #Prepare the figure
        fig = plt.figure()
        ax1 = fig.add_subplot(2,2,1)
        ax2 = fig.add_subplot(2,2,2)
        ax3 = fig.add_subplot(2,2,3)
        ax4 = fig.add_subplot(2,2,4)
    
        #If the inputs are tuples containing multiple curves to plot, cycle through
        # the different curves, plotting them on the same graph
        num_curves = len(self.fscore)
        if self.name[0] != None:
            assert len(self.name) == num_curves
            for i in range(0, num_curves):
                #ROC curve, first
                ax1.plot(self.fpr[i], self.tpr[i], label=self.name[i])
                ax1.axis([0, 1, 0, 1])
                #ax1.annotate('theta='+`theta[th_mid]`, (fp_rate[th_mid], tp_rate[th_mid]))
                ax1.set_title('ROC curve')
                ax1.set_ylabel('true-positive rate')
                ax1.set_xlabel('false-positive rate')
                #display the x=y line for comparison
                x = np.asarray(np.arange(0, 1, 0.1))
                ax1.plot(x, x, 'k--')
                
                #Next plot the Fscore
                ax2.plot(self.theta[i], self.fscore[i], label=self.name[i])
                ax3.axis([0, 1, 0, 1])
                ax2.set_title('Fscore')
                ax2.set_ylabel('Fscore')
                ax2.set_xlabel('threshold')
                
                #Plot the precrec under the ROC
                ax3.plot(self.recall[i], self.precision[i], label=self.name[i])
                ax3.axis([0, 1, 0, 1])
                ax3.set_title('precision vs recall')
                ax3.set_ylabel('precision')
                ax3.set_xlabel('recall')
                
                #Finally, plot the accuracy under the Fscore
                ax4.plot(self.theta[i], self.accuracy[i], label=self.name[i])
                ax3.axis([0, 1, 0, 1])
                ax4.set_title('accuracy')
                ax4.set_ylabel('accuracy')
                ax4.set_xlabel('threshold')
                
            ax1.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=2, borderaxespad=0., prop={'size':20})
        else:
            for i in range(0, num_curves):
                #ROC curve, first
                ax1.plot(self.fpr[i], self.tpr[i], label='curve '+`i`)
                ax1.axis([0, 1, 0, 1])
                #ax1.annotate('theta='+`theta[th_mid]`, (fp_rate[th_mid], tp_rate[th_mid]))
                ax1.set_title('ROC curve')
                ax1.set_ylabel('true-positive rate')
                ax1.set_xlabel('false-positive rate')
                #display the x=y line for comparison
                x = np.asarray(np.arange(0, 1, 0.1))
                ax1.plot(x, x, 'k--')
                
                #Next plot the Fscore
                ax2.plot(self.theta[i], self.fscore[i], label='curve '+`i`)
                ax3.axis([0, 1, 0, 1])
                ax2.set_title('Fscore')
                ax2.set_ylabel('Fscore')
                ax2.set_xlabel('threshold')
                
                #Plot the precrec under the ROC
                ax3.plot(self.recall[i], self.precision[i], label='curve '+`i`)
                ax3.axis([0, 1, 0, 1])
                ax3.set_title('precision vs recall')
                ax3.set_ylabel('precision')
                ax3.set_xlabel('recall')
                
                #Finally, plot the accuracy under the Fscore
                ax4.plot(self.theta[i], self.accuracy[i], label='curve '+`i`)
                ax3.axis([0, 1, 0, 1])
                ax4.set_title('accuracy')
                ax4.set_ylabel('accuracy')
                ax4.set_xlabel('threshold')
                
            ax1.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=2, borderaxespad=0., prop={'size':20})
    
    
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
                
            fig = plt.figure()
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
                
            fig = plt.figure()
            ax1 = fig.add_subplot(1,3,1)
            ax2 = fig.add_subplot(1,3,2)
            ax3 = fig.add_subplot(1,3,3)
            fig.subplots_adjust(left=0.25, bottom=0.25)
            depth0 = 0
            
            crop = (self.target.shape[-1] - self.prediction[res].shape[-1])/2        
            
            #All these images are in gray-scale
            plt.gray()
            im1 = ax1.imshow(self.raw[0,crop:-crop,crop:-crop])
            ax1.set_title('raw image', fontsize = 20)
            
            im2 = ax2.imshow(self.target[0,0,crop:-crop,crop:-crop])
            ax2.set_title('groundtruth', fontsize = 20)
            
            im3 = ax3.imshow(self.prediction[res][0,crop,:,:])
            ax3.set_title('prediction', fontsize = 20)     
            
            axcolor = 'lightgoldenrodyellow'
            axdepth = fig.add_axes([0.25, 0.3, 0.65, 0.03], axisbg=axcolor)
            depth = Slider(axdepth, 'Min', 0, self.prediction[res].shape[-1]-crop, valinit=depth0)
            
            def update(val):
                zlayer = int(depth.val)
                im1.set_data(self.raw[crop+zlayer,crop:-crop,crop:-crop])
                im2.set_data(self.target[0,crop+zlayer,crop:-crop,crop:-crop])
                im3.set_data(self.prediction[res][0, zlayer,:,:])
                fig.canvas.draw()
                
            fig.suptitle(self.name[res], fontsize = 20)
            depth.on_changed(update)        
            plt.show()

            