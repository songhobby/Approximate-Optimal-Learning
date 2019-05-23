import numpy as np
import matplotlib.pyplot as plt
import time
from mdp import MDP
from rl import RL
from maze import maze_generator

from bayesDQL import BayesDeepQL,BayesWorld

import logging
import pathlib
import datetime
import torch


torch.set_printoptions(threshold=100000000)


seed=1
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
bdql.bayesWorld.load_state_dict(torch.load('.logging/model20190313050418'))
TestMazes = bdql.Mazes_generator(128)
loss,accuracy,naive_guess,hidden=bdql.eval_bayes(0,1000,TestMazes)
print(loss,accuracy,naive_guess)
print(hidden)
