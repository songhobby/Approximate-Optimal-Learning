import numpy as np
import matplotlib.pyplot as plt
import time
from mdp import MDP
from rl import RL
from maze import maze_generator

import bayesDQL
from bayesDQL import BayesDeepQL,BayesWorld,device

import logging
import pathlib
import datetime
import torch
import os
import glob
torch.cuda.empty_cache()
#filename=str(datetime.datetime.now().\
#             strftime("%Y%m%d%H%M%S"))
#pathlib.Path('./.logging/').mkdir(parents=True,
#            exist_ok=True)
#logging.basicConfig(filename='./.logging/'+filename,filemode='w',
#                    level=logging.INFO)


seed=0
np.random.seed(seed)
# Discount factor: scalar in [0,1)
discount = 0.95
        
# MDP object
[T,R,E] = maze_generator()
mdp = MDP(T,R,E,discount)

# RL problem
rlProblem = RL(mdp,np.random.normal)

bdql = BayesDeepQL(rlProblem,rlProblem.mdp.nStates)
bdql.build_bayes('LSTM',64,1,128)
list_of_files=glob.glob('.logging/model*')
latest_file=max(list_of_files,key=os.path.getctime)
print("Loading ",latest_file)
bdql.bayesWorld.load_state_dict(torch.load(latest_file))
bdql.bayesWorld.to(device)
bdql.train_bayes(0,4000,1,10)


#plt.figure(figsize=(15,10))
#plt.grid()
#plt.title("Maze Learning")
#plt.xlabel("Episodes")
#plt.ylabel("Rewards")
#
#Trials = 100
#nEpisodes=1000
#
#
## Test Q-learning
#cumu_ma = []
#start_time = time.time()
#for iterTr in range(Trials):
#    print(iterTr)
#    [Q,policy,cumu_reward] = rlProblem.qLearning(s0=0,initialQ=\
#    np.zeros([mdp.nActions,mdp.nStates]),
#    nEpisodes=nEpisodes,nSteps=100,epsilon=0.05,temperature=1)
#    cumu_ma.append(cumu_reward)
#print ("\nQ-learning results")
#print ("Q: ", Q)
#print ("Policy: ", policy)
#print ("Time taken for each trial: ", (time.time() - start_time)/Trials)
#plt.plot(list(range(nEpisodes)), np.mean(cumu_ma,axis=0), 
#         label="Q-learning")
#plt.legend()
#plt.show()