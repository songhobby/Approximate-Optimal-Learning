import numpy as np
import matplotlib.pyplot as plt
import time
from mdp import MDP
from rl import RL
import glob
import os
from maze import maze_generator

import bayesDQL
from bayesDQL import BayesDeepQL,BayesWorld,device

import logging
import pathlib
import datetime
import torch
torch.cuda.empty_cache()
filename=str(datetime.datetime.now().\
             strftime("%Y%m%d%H%M%S"))
pathlib.Path('./.logging/').mkdir(parents=True,
            exist_ok=True)
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
bdql.build_bayes('LSTM',64,1,64)
list_of_files=glob.glob('.logging/model*')
latest_file=max(list_of_files,key=os.path.getctime)
bdql.bayesWorld.load_state_dict(
        torch.load(latest_file))
bdql.bayesWorld.to(device)
#loss,accuracy,naive_guess,hidden=bdql.eval_bayes(0,1000,TestMazes)
#print(loss,accuracy,naive_guess)
#s0,nSteps,series,Bayes,batch_size
history=bdql.train_dql(0,200,2,bdql.bayesWorld,128,16)
hist_series = [np.array(x).mean() for x in history]
def process(his):
    his2=[]
    for serie in his:
        min_l = np.min(list(map(len,serie)))
        H = [x[:min_l] for x in serie]
        H=np.mean(H,0)
        his2.append(H)
    min_l = np.min(list(map(len,his2)))
    H = [x[:min_l] for x in his2]
    return np.mean(H,1),H[-1],H[0]
history,last_h,first_h=process(history)
plt.grid()
plt.plot(range(len(history)),history,label="Average SLearning")
#plt.plot(range(len(first_h)),first_h,label="First SLearning")
#plt.plot(range(len(last_h)),last_h,label="Last SLearning")
plt.legend()
plt.savefig("BQLearning"+filename+".png")
plt.show()