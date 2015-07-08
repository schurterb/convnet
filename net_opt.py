# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 09:16:24 2015

@author: schurterb

Uses hyperopt library to optimize network structure.
"""
print 'loading modules'
from trial_net import trial_net

from hyperopt import fmin, tpe, hp, STATUS_OK, Trials

print 'Initializing...'

#Objective function to optimize
def objective(x):
    return {
            'loss': trial_net(x),
            #'loss': (x[0]**2 + x[1]**3 + x[2]**4),
            'status': STATUS_OK,
            # --probably want more stuff to be returned--
            }


#Search space
space = [hp.choice('nlayers', range(8, 15)),
         hp.choice('nfilters', range(10, 50, 2)),
         hp.choice('fsizes', range(5, 11, 2))]
         
        
#Set the optimizer to use a mongo database
#trials = MongoTrials('mongo://localhost:12345/test/jobs', exp_key='exp1')
trials = Trials()

print 'Initialization complete'
print 'Begin optimization'
#Run the hyper optimization
best_params = fmin(objective, space,
                   algo=tpe.suggest,
                   max_evals=150,
                   trials=trials)
                   
print 'Optimization complete'
                   
#Save the results
f = open('results/hyperopt/best_params.txt', 'w')
f.write(`best_params`)
f.close()

print best_params