# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 15:27:35 2015

@author: schurterb
"""

from libcpp cimport bool
import cython as cy
import numpy as np
cimport numpy as np


cdef extern from "malisLoss.cpp":
    void malisLoss(int* dims, float* conn, double* nhood, int* seg, 
                   double margin, bool pos, float* losses, 
                   double* lossReturn, double* randIndexReturn)
    

def findMalisLoss(np.ndarray[int, ndim=3] compTrue,
                  np.ndarray[float, ndim=4] affTrue,
                  np.ndarray[float, ndim=4] affEst,
                  np.ndarray[np.double_t, ndim=2] nhood = -np.eye(3)):
    
    predShape = (affEst.shape[0], affEst.shape[1], affEst.shape[2], affEst.shape[3])
    cdef np.ndarray[int, ndim=1] dims = np.asarray(predShape, dtype=np.intc)
    
    cdef np.ndarray[float, ndim=4] affpos = np.minimum(affTrue, affEst)
    cdef np.ndarray[float, ndim=4] dloss_p = np.zeros(predShape, dtype=np.float32)
    cdef np.ndarray[double, ndim=1] lossAvg_p = np.zeros(1)
    cdef np.ndarray[double, ndim=1] randIndex_p = np.zeros(1)
    
    malisLoss(&dims[0], &affpos[0,0,0,0], &nhood[0,0], &compTrue[0,0,0], 0.3, True, &dloss_p[0,0,0,0], &lossAvg_p[0], &randIndex_p[0])
    
    cdef np.ndarray[float, ndim=4] affneg = np.maximum(affTrue, affEst)
    cdef np.ndarray[float, ndim=4] dloss_n = np.zeros(predShape, dtype=np.float32)
    cdef np.ndarray[double, ndim=1] lossAvg_n = np.zeros(1)
    cdef np.ndarray[double, ndim=1] randIndex_n = np.zeros(1)
    
    malisLoss(&dims[0], &affneg[0,0,0,0], &nhood[0,0], &compTrue[0,0,0], 0.3, True, &dloss_n[0,0,0,0], &lossAvg_n[0], &randIndex_n[0])
    
    dloss = dloss_p - dloss_n
    lossAvg = (lossAvg_p + lossAvg_n)/2.0
    randIndex = (randIndex_p + randIndex_n)/2.0
        
    dloss = np.transpose(dloss.reshape((affEst.shape[3], affEst.shape[2], affEst.shape[1], affEst.shape[0])), (3,2,1,0))    
    
    return dloss, 1-randIndex, lossAvg