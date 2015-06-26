# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 09:16:24 2015

@author: schurterb

Uses hyperopt library to optimize network structure.
"""

from trial_net import trial_net

import pickle
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.mongoexp import MongoTrials


    
#Objective function to optimize
def objective(x):
    return {
            'loss': trial_net(x),
            'status': STATUS_OK,
            # --probably want more stuff to be returned--
            }


#Search space
space = [hp.choice('nlayers', range(5, 15)),
         hp.choice('nfilters', range(10, 30)),
         hp.choice('fsizes', range(5, 13, 2))]
         
         
#Set the optimizer to use a mongo database
#trials = MongoTrials('mongo://localhost:27017/hyper_opt/jobs', exp_key='exp1')
trials = Trials()

#Run the hyper optimization
best_params = fmin(objective, space,
                   algo=tpe.suggest,
                   max_evals=50,
                   trials=trials)
                   
                   
#Save the results
f = open('results/hyperopt/best_params.txt', 'w')
f.write(`best_params`)
f.close()

print best_params