# -*- coding: utf-8 -*-
"""
Created on Wed May 27 11:23:54 2015

@author: schurterb

Class for loading training or testing data. Currently only set up to support
 hdf5 files.
"""

import h5py

class LoadData(object):
    
    """Open and read the data and label files"""
    def __read_files(self):
        
        if(self.file_type == 'hdf5'):
            if (type(self.folder) == tuple) or (type(self.folder) == list):
                self.data_file = ()
                self.x = ()
                if self.data_file_name:
                    for folder in self.folder:
                        self.data_file += (h5py.File(folder + self.data_file_name, 'r') ,)
                        self.x += (self.data_file[-1]['main'] ,)
                
                self.label_file = ()
                self.y = ()
                if self.label_file_name:
                    for folder in self.folder:  
                        self.label_file += (h5py.File(folder + self.label_file_name, 'r') ,)
                        self.y += (self.label_file[-1]['main'] ,)
                        
                self.seg_file = ()
                self.z = ()
                if self.seg_file_name:
                    for folder in self.folder:
                        self.seg_file += (h5py.File(folder + self.seg_file_name, 'r') ,)
                        self.z += (self.seg_file[-1]['main'] ,)
            else:
                self.data_file = ()
                self.x = ()
                if self.data_file_name:
                    self.data_file += (h5py.File(self.folder + self.data_file_name, 'r') ,)
                    self.x += (self.data_file[-1]['main'] ,)
                
                self.label_file = ()
                self.y = ()
                if self.label_file_name:
                    self.label_file += (h5py.File(self.folder + self.label_file_name, 'r') ,)
                    self.y += (self.label_file[-1]['main'] ,)
                        
                self.seg_file = ()
                self.z = ()
                if self.seg_file_name:
                    self.seg_file += (h5py.File(self.folder + self.seg_file_name, 'r') ,)
                    self.z += (self.seg_file[-1]['main'] ,)
            
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
         self.seg_file_name = kwargs.get('seg_file_name', None)
         self.file_type = kwargs.get('file_type', 'hdf5')

         self.__read_files()
         
         
    """Return the data set"""
    def get_data(self):
        return self.x
        
    """Return the label set"""
    def get_labels(self):
        return self.y
        
    """Return the label set"""
    def get_segments(self):
        return self.z
         
    """
    Close the data and label files
    In the case of hdf5 files, the data will no longer be accessible after this.
    """
    def close(self):        
        for dfile in self.data_file:
            dfile.close()
        for lfile in self.label_file:
            lfile.close()
        for sfile in self.seg_file:
            sfile.close()
        
    