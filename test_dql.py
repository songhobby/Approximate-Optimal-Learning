import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
from mdp import MDP
from rl import RL
from maze import maze_generator

from bayesDQL import BayesDeepQL,BayesWorld,device

import logging
import pathlib
import datetime
import torch

torch.cuda.empty_cache()

torch.set_printoptions(threshold=100000000)


seed=1234567891
np.random.seed(seed)
# Discount factor: scalar in [0,1)
discount = 0.95
H=[]

ax=plt.figure(figsize=(15,10))
plt.grid()
plt.xlabel("Episodes")
plt.ylabel("Rewards (Undiscounted)")

for i in range(100):
            
    # MDP object
    [T,R,E] = maze_generator()
    mdp = MDP(T,R,E,discount)
    
    # RL problem
    rlProblem = RL(mdp,np.random.normal)
    q,p,h=rlProblem.modelBasedRL(0,defaultT=np.ones([mdp.nActions,mdp.nStates,mdp.nStates])/mdp.nStates,
                                 initialR=np.zeros([mdp.nActions,mdp.nStates]),
                                 nEpisodes=1500,nSteps=100,epsilon=0.1)
    H.append(h)
    print(i,np.mean(h[-100:]))
H=np.mean(H,0)
print(q)
#H1=np.load('QLearning100.npy')
plt.plot(range(len(H)),H,label="MBLearning")
np.save('MBLearning',np.array(H))
##
#H2=np.load('DQLearning100.npy')
#plt.plot(range(len(H2)),H2,label="DeepQL")

#H3=np.load('BDQLearning100.npy')
#plt.plot(range(len(H3)),H3,label="BayesDeepQL without training)

#H=[]
#for i in range(100):
#    [T,R,E] = maze_generator()
#    mdp = MDP(T,R,E,discount)
#    
#    # RL problem
#    rlProblem = RL(mdp,np.random.normal)
#    bdql = BayesDeepQL(rlProblem,rlProblem.mdp.nStates)
#    bdql.build_bayes('LSTM',64,1,128)
#    bdql.bayesWorld.load_state_dict(
#            torch.load('.logging/model20190321184957'))
#    bdql.bayesWorld.to(device)
#    h=bdql.dql(s0=0,nEpisodes=1500,nSteps=100000,maze=rlProblem,batch_size=256,Bayes=None,load=True)
#    print(i,h[-1])
#    H.append(h)
#min_l = np.min(list(map(len,H)))
#H = [x[:min_l] for x in H]
#H=np.mean(H,0)
#np.save('dq_T',np.array(H))
#plt.plot(range(len(H)),H,label="DQL_T")
#plt.savefig("BDQLearning.png")
#plt.legend()
#plt.show()

#H=[]
#for i in range(100):
#    [T,R,E] = maze_generator()
#    mdp = MDP(T,R,E,discount)
#    
#    # RL problem
#    rlProblem = RL(mdp,np.random.normal)
#    bdql = BayesDeepQL(rlProblem,rlProblem.mdp.nStates)
#    bdql.build_bayes('LSTM',64,1,128)
#    bdql.bayesWorld.load_state_dict(
#            torch.load('.logging/model20190321184957'))
#    bdql.bayesWorld.to(device)
#    h=bdql.dql(s0=0,nEpisodes=1500,nSteps=100000,maze=rlProblem,batch_size=256,Bayes=bdql.bayesWorld,load=True)
#    print(i,h[-1])
#    H.append(h)
#min_l = np.min(list(map(len,H)))
#H = [x[:min_l] for x in H]
#H=np.mean(H,0)
#np.save('bdq_T',np.array(H))
#plt.plot(range(len(H)),H,label="BDQL_T")
#plt.savefig("BQ_T.png")
#plt.legend()
#plt.show()
#


#bdql.build_bayes('LSTM',100,2,128)
#bdql.bayesWorld.load_state_dict(torch.load('.logging/model20190301154854'))