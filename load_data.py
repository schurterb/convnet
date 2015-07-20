# -*- coding: utf-8 -*-
"""
Created on Wed May 27 11:23:54 2015

@author: schurterb

Class for loading training or testing data. Currently only set up to support
 hdf5 files.
"""

import os
import h5py

class LoadData(object):
    
    """Open and read the data and label files"""
    def __read_files(self):
        
        if(self.file_type == 'hdf5'):
            if (type(self.folder) == tuple) or (type(self.folder) == list):
                self.data_file = ()
                self.label_file = ()
                self.x = ()
                self.y = ()
                for folder in self.folder:
                    self.data_file += (h5py.File(folder + self.data_file_name, 'r') ,)
                    self.x += (self.data_file[-1]['main'] ,)
                    
                    self.label_file += (h5py.File(folder + self.label_file_name, 'r') ,)
                    self.y += (self.label_file[-1]['main'] ,)
            else:
                self.data_file = (h5py.File(self.folder + self.data_file_name, 'r') ,)
                self.x = (self.data_file[-1]['main'] ,)
                
                self.label_file = (h5py.File(self.folder + self.label_file_name, 'r') ,)
                self.y = (self.label_file[-1]['main'] ,)
            
        else:   
            raise TypeError("Unsupported file type")
    
    """
    Must be initialized with a string specifying the directory containing
     the data and labels. 
    Directory must be the path from the root directory to data
    """
    def __init__(self, **kwargs):
         
         self.folder = kwargs.get('directory', '')
         self.data_file_name = kwargs.get('data_file_name', None)
         self.label_file_name = kwargs.get('label_file_name', None)
         self.file_type = kwargs.get('file_type', 'hdf5')

         self.home_folder = os.getcwd() + '/'
         os.chdir('/.')
         
         self.__read_files()
         
         os.chdir(self.home_folder)
         
    """Return the data set"""
    def get_data(self):
        return self.x
        
    """Return the label set"""
    def get_labels(self):
        return self.y
         
    """
    Close the data and label files
    In the case of hdf5 files, the data will no longer be accessible after this.
    """
    def close(self):
        for dfile, lfile in zip(self.data_file, self.label_file):
            dfile.close()
            lfile.close()
        
    