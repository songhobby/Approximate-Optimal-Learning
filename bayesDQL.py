import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random


from collections import namedtuple
import logging
import gc
import datetime

from rl import RL
from mdp import MDP
from maze import maze_generator,naive_pred

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state','action','belief',
                         'next_state','reward','done'))

TransitionC = namedtuple('TransitionC',
                        ('state','action','next_state','reward','done'))

BATCH_SIZE = 32
UPDATE = 1
TARGET_UPDATE = 1000
PRINT = 100
EPS_START = 0.5
EPS_DECAY = 1000
EPS_END = 0.1
ACTI = F.hardtanh

class ReplayBuffer(object):
    
    def __init__(self,capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        
    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity
        
    def memory(self):
        return self.memory
        
    def sample(self,batch_size):
        return random.sample(self.memory, batch_size)
    
    def one_sample(self):
        return np.random.choice(self.memory)
    
    def __len__(self):
        return len(self.memory)
    
class ReplayBufferC(object):
    
    def __init__(self,capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        
    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = TransitionC(*args)
        self.position = (self.position + 1) % self.capacity
        
    def memory(self):
        return self.memory
        
    def sample(self,batch_size):
        return random.sample(self.memory, batch_size)
    
    def one_sample(self):
        return np.random.choice(self.memory)
    
    def __len__(self):
        return len(self.memory)
    
def one_hot(ts,dim):
    res=torch.zeros([ts.size()[0],dim],dtype=torch.float,device=device)
    for i in range(ts.size()[0]):
        res[i,ts[i].long()]=1.
    return res

def reward_lc(y_pred,y_targ):
    diff = torch.abs((y_pred - y_targ) / 
                     torch.clamp(torch.abs(y_targ),min=1))
    return torch.mean(diff,-1)
    
class BayesWorld(nn.Module):
    
    def __init__(self,rnn,fsi,fai,fh,num_layers):
        super(BayesWorld,self).__init__()
        self.num_layers = num_layers
        self.fh = fh
        if(rnn == 'LSTM'):
            self.rnn = nn.LSTM(input_size=fsi+fai,
                               hidden_size=fh,
                               num_layers=num_layers)
        elif(rnn == 'GRU'):
            self.rnn = nn.GRU(input_size=fsi+fai,
                              hidden_size=fh,
                              num_layers=num_layers)
        else:
            raise Exception("unknown rnn {}".format(str(rnn)))
        
        self.rnnType = rnn
        
        self.ns1 = nn.Linear(fh,fh)
        self.ns2 = nn.Linear(fh,fh)
        self.ns3 = nn.Linear(fh,fh)
        self.ns4 = nn.Linear(fh,fsi)
        
        self.nr1 = nn.Linear(fh,fh)
        self.nr2 = nn.Linear(fh,1)
        
        self.ndone1 = nn.Linear(fh,fh)
        self.ndone2 = nn.Linear(fh,1)
        
    def init_h(self,bs):
        if(self.rnnType == 'LSTM'):
            return (torch.zeros(self.num_layers,bs,self.fh,
                                       device=device),
                           torch.zeros(self.num_layers,bs,self.fh,
                                       device=device))
        elif(self.rnnType == 'GRU'):
            return torch.zeros(self.num_layers,bs,self.fh,
                                      device=device)
        else:
            raise Exception("unknow rnn {}".format(str(self.rnnType)))
            
    def forward(self,state,action,h,training):
#        h = (h[0].detach(),h[1].detach())
        x,hn = self.rnn(torch.cat((state,action),1)\
                        .unsqueeze(0))
#        x = ACTI(self.predict1(x))
#        x = ACTI(self.predict2(x))
#        x = ACTI(self.predict_interm(x))
            
        if training:
            xs = ACTI(self.ns1(x))
            xs = ACTI(self.ns2(xs))
            xs = ACTI(self.ns3(xs))
            xs = ACTI(self.ns4(xs))
            
            xr = ACTI(self.nr1(x))
            xr = self.nr2(xr)
            xdone = ACTI(self.ndone1(x))
            xdone = ACTI(self.ndone2(xdone))
            
            return xs,xr,xdone,hn
        else:
            if(self.rnnType == 'LSTM'):
                return hn,hn[1]
            else:
                return hn,hn
            
            
    
class DeepQN(nn.Module):
    
    def __init__(self,fbi,fsi,fso):
        super(DeepQN,self).__init__()
        self.belief1 = nn.Linear(fbi,fbi)
        self.belief2 = nn.Linear(fbi,fbi)
        self.belief_interm = nn.Linear(fbi,fbi)
        
        self.bs1 = nn.Linear(fbi+fsi,fbi+fsi)
        self.bs2 = nn.Linear(fbi+fsi,fbi+fsi)
        self.bs3 = nn.Linear(fbi+fsi,fbi+fsi)
        self.bs4 = nn.Linear(fbi+fsi,fso)
        self.bs5 = nn.Linear(fso,fso)
        self.bs6 = nn.Linear(fso,fso)
    
    def forward(self,belief,state):
        x = ACTI(self.belief1(belief))
        x = ACTI(self.belief2(x))
        x = ACTI(self.belief_interm(x))
        x = torch.cat((x,state),1)
        
        x = ACTI(self.bs1(x))
        x = ACTI(self.bs2(x))
        x = ACTI(self.bs3(x))
        x = ACTI(self.bs4(x))
        x = self.bs6(self.bs5(x))
        return x
    
    def BayesTrainable(self,trigger):
        for param in self.belief1.parameters():
            param.require_grad = trigger
        for param in self.belief2.parameters():
            param.require_grad = trigger
        for param in self.belief_interm.parameters():
            param.require_grad = trigger
        
        
class BayesDeepQL(object):
    
    def __init__(self,rl,dimBelief):
        self.rl = rl
        self.dimBelief=dimBelief
       
    def Mazes_generator(self,batch_size):
        Mazes = []
        for MzIter in range(batch_size):
            [T,R,E] = maze_generator()
            mdp = MDP(T,R,E,self.rl.mdp.discount)
            rlSample = RL(mdp,np.random.normal)
            Mazes.append(rlSample)
        return Mazes
    
    def build_bayes(self,rnn,fh,num_layers,batch_size):
        self.batch_size=batch_size
        self.bayesWorld = BayesWorld(rnn,self.rl.mdp.nStates,
                                     self.rl.mdp.nActions,
                                     fh,num_layers).to(device)
        
    def train_bayes(self,s0,nSteps,Epochs,series):
        #evaluate
        TestMazes = self.Mazes_generator(self.batch_size)
        val_loss,accuracy,naive_guess,belief0=self.eval_bayes(s0,nSteps,TestMazes)
        print("Validation loss: {:6.2f} accuracy: {:4.4f} naive_accuracy: {:4.4f}"\
              .format(val_loss,accuracy,naive_guess))
        optimizer=optim.Adam(self.bayesWorld.parameters())
        for SIter in range(series):
            batch = []
            for maze in self.Mazes_generator(self.batch_size):
                s=s0
                memory=[]
                for StIter in range(nSteps):
                    action = np.random.randint(self.rl.mdp.nActions)
                    next_s,reward,done = maze.\
                    sampleRewardAndNextState(s,action)
                    memory.append([s,action,next_s,reward,done])
                    s=next_s
                    if(done):
                        s=s0
                batch.append(memory)
            batch = zip(*[list(zip(*ep)) for ep in batch])
            state_batch,action_batch,\
            next_state_batch,reward_batch,done_batch\
            =[torch.tensor(x,dtype=torch.float,
                           device=device) for x in batch]
            for EpIter in range(Epochs):
                loss=0
                hidden=self.bayesWorld.init_h(self.batch_size)
                for StIter in range(nSteps):
                    self.bayesWorld.train()
                    next_state_pred,reward_pred,done_pred,hidden\
                    =self.bayesWorld(one_hot(state_batch[:,StIter],
                                             self.rl.mdp.nStates),
                                     one_hot(action_batch[:,StIter],
                                             self.rl.mdp.nActions),
                                     hidden,True)
                    state_l = nn.CrossEntropyLoss()(next_state_pred.squeeze(),
                                                 next_state_batch[:,StIter]\
                                                 .long())
#                    reward_l = reward_lc(reward_pred.squeeze(),
#                                         reward_batch[:,StIter])
#                    done_l = nn.BCEWithLogitsLoss()(done_pred.squeeze(),
#                                                 done_batch[:,StIter])
                    loss+=state_l #(state_l+reward_l+done_l)/3
#                    loss+=state_l
                optimizer.zero_grad()
                loss=loss / nSteps
#                print("series: {:5d}/{} epochs {:5d}/{} loss: {:6.2f} state: {:6.2f} reward: {:6.2f} done: {:6.2f}"\
#                      .format(SIter,series,EpIter,Epochs,
#                              loss.item(),
#                              state_l.item(),
#                              reward_l.item(),
#                              done_l.item()))
                print("series: {:5d}/{} epochs {:5d}/{} loss: {:6.6f}"\
                      .format(SIter,series,EpIter,Epochs,loss.item()))
                loss.backward(retain_graph=True)
#                for param in self.bayesWorld.parameters():
#                    param.grad.data.clamp(-1,1)
                optimizer.step()
            val_loss,accuracy,naive_guess,belief0=self.eval_bayes(s0,nSteps,TestMazes)
            print("Validation loss: {:6.2f} accuracy: {:4.4f} naive_accuracy: {:4.4f}"\
                  .format(val_loss,accuracy,naive_guess))
        filename='model'+str(datetime.datetime.now().\
                             strftime("%Y%m%d%H%M%S"))
        print('Saving ',filename)
        torch.save(self.bayesWorld.state_dict(),'.logging/'+filename)
    def eval_bayes(self,s0,nSteps,Mazes):
        self.bayesWorld.eval()
        batch = []
        for maze in Mazes:
            s=s0
            memory=[]
            for StIter in range(nSteps):
                action = np.random.randint(self.rl.mdp.nActions)
                next_s,reward,done = maze.\
                sampleRewardAndNextState(s,action)
                memory.append([s,action,next_s,reward,done])
                s=next_s
                if(done):
                    s=s0
            batch.append(memory)
        batch = zip(*[list(zip(*ep)) for ep in batch])
        state_batch,action_batch,\
        next_state_batch,reward_batch,done_batch\
        =[torch.tensor(x,dtype=torch.float,
                       device=device) for x in batch]
        count=0
        naive_pred_count = 0
        loss=0 #loss_state=loss_reward=loss_done=0
        hidden=self.bayesWorld.init_h(self.batch_size)
        with torch.no_grad():
            for StIter in range(nSteps):
                next_state_pred,reward_pred,done_pred,hidden\
                =self.bayesWorld(one_hot(state_batch[:,StIter],
                                         self.rl.mdp.nStates),
                                 one_hot(action_batch[:,StIter],
                                         self.rl.mdp.nActions),
                                 hidden,True)
                count+=(next_state_pred.squeeze().argmax(1).long() ==\
                        next_state_batch[:,StIter].long()).sum().item()
                naive_ns = torch.tensor([naive_pred(x,y) for x,y in 
                                         zip(state_batch[:,StIter],
                                             action_batch[:,StIter])])
                naive_pred_count+=(naive_ns.long().to(device) ==\
                        next_state_batch[:,StIter].long()).sum().item()
                state_l = nn.CrossEntropyLoss()(next_state_pred.squeeze(),
                                             next_state_batch[:,StIter]\
                                             .long())
#                reward_l = reward_lc(reward_pred.squeeze(),
#                                     reward_batch[:,StIter])
#                done_l = nn.BCEWithLogitsLoss()(done_pred.squeeze(),
#                                             done_batch[:,StIter])
#                loss_state += state_l
#                loss_reward += reward_l
#                loss_done += done_l
                loss+=state_l #(state_l+reward_l+done_l)/3
                #loss+=state_l
        return loss.item()/nSteps,count / (nSteps*self.batch_size),\
               naive_pred_count / (nSteps*self.batch_size),hidden
    def train_dql(self,s0,nSteps,series,Bayes,batch_size,sample_size):
        self.policy_net = DeepQN(self.dimBelief,
                                 self.rl.mdp.nStates,
                                 self.rl.mdp.nActions).to(device)
        import os, glob
        list_of_files=glob.glob('.logging/bmodel*')
        latest_file=max(list_of_files,key=os.path.getctime)
        self.policy_net.load_state_dict(torch.load(latest_file))
        self.target_net = DeepQN(self.dimBelief,
                                 self.rl.mdp.nStates,
                                 self.rl.mdp.nActions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(),
                                    lr=1e-3)
        def action_select(s,belief):
            eps_threshold = EPS_END + (EPS_START-EPS_END)*\
            np.exp(-1.*steps/EPS_DECAY)
            if np.random.random() > eps_threshold:
                with torch.no_grad():
                    _action = self\
                    .target_net(belief,
                                one_hot(torch.tensor(s)
                                .view(batch_size,-1),
                                self.rl.mdp.nStates))
                    _action = torch.multinomial(F.softmax(_action,dim=-1),1)
                    return _action
            else:
                return torch.randint(self.rl.mdp.nActions,(
                        batch_size,1),device=device)
        def optimize_model():
            if len(memory) < sample_size:
                return 0
            transitions = memory.sample(sample_size)
            batch = Transition(*zip(*transitions))
            state_batch,action_batch,belief_batch,\
            next_s_batch,reward_batch,done_batch=\
            [torch.cat(x).to(device) for x in batch]
            Q_policy=self.policy_net(belief_batch,
                                     one_hot(state_batch,
                                             self.rl.mdp.nStates))\
            .gather(1,action_batch)
            with torch.no_grad():
                Q_target=self.target_net(belief_batch,
                                         one_hot(next_s_batch,
                                                 self.rl.mdp.nStates))\
                .max(1)[0]
            Q_expected=(Q_target*self.rl.mdp.discount)+reward_batch
            term_loss = nn.MSELoss()(self.policy_net(\
                                  belief_batch[0::sample_size],
                                  one_hot(terminal_states[\
                        np.random.randint(len(Mazes[0].mdp.E))],
                                     self.rl.mdp.nStates)),
                                     torch.zeros(batch_size,
                                                 self.rl.mdp.nActions)\
                                                 .to(device))
            loss = term_loss+nn.MSELoss()(Q_policy,Q_expected.unsqueeze(1))
            self.optimizer.zero_grad()
            loss.backward()
#            for param in self.policy_net.parameters():
#                param.grad.data.clamp_(-1,1)
            self.optimizer.step()
            return loss.item()
        self.bayesWorld.eval()
        History=[]
        for sIter in range(series):
            rewardHistory=[[0]]*batch_size
            steps=0
            Mazes=self.Mazes_generator(batch_size)
            terminal_states = torch.tensor(
                    list(zip(*[maze.mdp.E for maze in Mazes]))).to(device)
            memory=ReplayBuffer(5000)
            import gc
            print(gc.collect())
            hidden = self.bayesWorld.init_h(batch_size)
            belief=hidden
            if (Bayes.rnnType == 'LSTM'):
                belief = hidden[1]
            s=[s0]*batch_size
            action=None
            next_s=[0]*batch_size
            reward=[0.]*batch_size
            done=[0]*batch_size
            discount = [1.]*batch_size
            for StIter in range(nSteps):
                steps+=1
                for idx,Maze in enumerate(Mazes):
                    action=action_select(s,belief[-1])
                    next_s[idx],reward[idx],done[idx] =\
                    Maze.sampleRewardAndNextState(s[idx],action[idx][0])
                    rewardHistory[idx][-1]+=discount[idx]*reward[idx]
                    discount[idx]*=self.rl.mdp.discount
                    if(done[idx]):
                        rewardHistory[idx].append(0)
                        discount[idx] = 1
                memory.push(torch.tensor(s,device=device),
                            action.detach(),
                            belief[-1].detach(),
                            torch.tensor(next_s,device=device),
                            torch.tensor(reward,device=device),
                            torch.tensor(done,device=device))
                with torch.no_grad():
                    hidden,belief =\
                    Bayes(one_hot(torch.tensor(s),
                                  self.rl.mdp.nStates),
                          one_hot(action,
                                  self.rl.mdp.nActions),
                          hidden,False)
                s=next_s.copy()
                for ix,d in enumerate(done):
                    if(d):
                        s[ix] = s0
                loss=optimize_model()
                if(steps % PRINT == 0 and StIter > 99):
                    print("series: {:5d}/{} step: {:6d}/{} loss: {:6.2f} mean_reward: {:6.2f}"\
                          .format(sIter,series,StIter,nSteps,
                                  loss,np.mean([x[-2] for x in rewardHistory])))
                if(StIter % TARGET_UPDATE == 0):
                    self.target_net.load_state_dict(self.policy_net.state_dict())
            History.append(rewardHistory)
        filename='bmodel'+str(datetime.datetime.now().\
                              strftime("%Y%m%d%H%M%S"))
        print('Saving ',filename)
        torch.save(self.policy_net.state_dict(),'.logging/'+filename)
        return History
                    
    def dql(self,s0,nEpisodes,nSteps,maze,batch_size,Bayes=None,load=True):
        if(load):
            self.policy_net = DeepQN(self.dimBelief,
                                     self.rl.mdp.nStates,
                                     self.rl.mdp.nActions).to(device)
            self.policy_net.load_state_dict(torch.load('.logging/bmodel20190411171149'))
            self.target_net = DeepQN(self.dimBelief,
                                     self.rl.mdp.nStates,
                                     self.rl.mdp.nActions).to(device)
            self.target_net.load_state_dict(self.policy_net.state_dict())
        else:
            self.policy_net = DeepQN(self.dimBelief,
                                     self.rl.mdp.nStates,
                                     self.rl.mdp.nActions).to(device)
            self.target_net = DeepQN(self.dimBelief,
                                     self.rl.mdp.nStates,
                                     self.rl.mdp.nActions).to(device)
            self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(),
                                        lr=1e-3)
        rewardHistory=[0]
        s=s0
        hidden=self.bayesWorld.init_h(1)
        belief=hidden
        if(self.bayesWorld.rnnType == 'LSTM'):
            belief=hidden[1]
    
        memory=ReplayBuffer(5000)
        
        terminal_states = torch.tensor(maze.mdp.E).view(-1,1).to(device)
        def action_select(s,belief):
            eps_threshold = EPS_END + (EPS_START-EPS_END)*\
            np.exp(-1.*steps/EPS_DECAY)
            if np.random.random() > eps_threshold:
                with torch.no_grad():
                    _action = self\
                    .policy_net(belief.view(1,-1),
                                        one_hot(torch.tensor(s).view(1,-1),
                                                self.rl.mdp.nStates))
#                    _action = self\
#                    .policy_net(belief,
#                                one_hot(torch.tensor(s).view(1,-1),
#                                        self.rl.mdp.nStates)).argmax(-1)
                    _action=F.softmax(_action,dim=-1)
                    return torch.multinomial(_action,1)
            else:
                return torch.tensor([[random.randrange(self.rl.mdp.nActions)]],
                                      device=device,dtype=torch.long)
        def optimize_model():
            if len(memory) < batch_size:
                return 0
            transitions = memory.sample(batch_size)
            batch = Transition(*zip(*transitions))
            state_batch,action_batch,belief_batch,\
            next_s_batch,reward_batch,done_batch=\
            [torch.cat(x).to(device) for x in batch]
            Q_policy=self.policy_net(belief_batch,
                                     one_hot(state_batch,
                                             self.rl.mdp.nStates))\
            .gather(1,action_batch)
            with torch.no_grad():
                Q_target=self.target_net(belief_batch,
                                         one_hot(next_s_batch,
                                                 self.rl.mdp.nStates))\
                .max(-1)[0]
                
            Q_expected=Q_target*self.rl.mdp.discount+reward_batch
            term_loss = nn.MSELoss()(self.policy_net(belief[-1],
                                     one_hot(terminal_states[\
                                     np.random.randint(self.rl.mdp.nActions)],
                                             self.rl.mdp.nStates)),
                                     torch.zeros(1,4).to(device))
            loss = term_loss+nn.MSELoss()(Q_policy,Q_expected.unsqueeze(1))
            if(np.random.rand(1) < 0.0001):
                print(Q_policy.squeeze().detach())
                print(Q_expected)
                print(self.target_net(belief_batch,
                                      one_hot(next_s_batch,
                                              self.rl.mdp.nStates)))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            return loss.item()
        
        EpIter=0
        discount_factor = 1
        for steps in range(nSteps):
            action = action_select(s,belief[-1])
            next_s,reward,done=maze.sampleRewardAndNextState(s,action)
            memory.push(torch.tensor([s],device=device),
                        action,
                        belief[-1],
                        torch.tensor([next_s],device=device),
                        torch.tensor([reward],device=device),
                        torch.tensor([done],device=device))
            rewardHistory[-1]+=discount_factor * reward
            discount_factor *= self.rl.mdp.discount
            if(Bayes is not None):
                with torch.no_grad():
                    hidden,belief=\
                    Bayes(one_hot(torch.tensor([s]),
                                  self.rl.mdp.nStates),
                          one_hot(action,
                                  self.rl.mdp.nActions),
                          hidden,False)
            s=next_s
            loss = -1
            for i in range(UPDATE):
                loss=optimize_model()
            if(steps % TARGET_UPDATE == 0):
                print("Update Target")
                self.target_net.load_state_dict(
                        self.policy_net.state_dict())
                self.target_net.eval()
            if(done):
                s=s0
                discount_factor = 1
                EpIter+=1
                print("Episode: {:5d} step: {:6d}/{} loss: {:6.2f} reward: {:6.2f}"\
                      .format(len(rewardHistory),steps,nSteps,
                              loss,rewardHistory[-1]))
                if(EpIter == nEpisodes):
                    return rewardHistory
                rewardHistory.append(0)
        return rewardHistory
            
            
            
            
                    
                
                
                
    