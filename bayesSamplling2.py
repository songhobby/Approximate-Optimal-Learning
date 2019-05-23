import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import sys
from collections import namedtuple,OrderedDict
import itertools
import logging
import datetime
from tqdm import trange


from rl import RL
from mdp import MDP
from maze import maze_generator,naive_pred
from bayesDQL import one_hot, TransitionC, ReplayBufferC

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ACTIV = [F.hardtanh,F.relu]

def reward_loss(y_pred,y_targ):
#    diff = torch.abs((y_pred - y_targ) / 
#                     torch.clamp(torch.abs(y_targ),min=1))
#    return F.smooth_l1_loss(torch.mean(diff,-1),torch.tensor(1.).to(device))
    return F.smooth_l1_loss(y_pred,y_targ)

class BayesHypo(nn.Module):
    
    def __init__(self,rnnType,num_layers,fsi,fai,fh,
                 optimizer=optim.Adam,
                 activ=[0]*10):
        super(BayesHypo,self).__init__()
        self.rnnType = rnnType
        self.num_layers = num_layers
        self.fsi = fsi
        self.fai = fai
        self.fh = fh
        self.optimizer=optimizer
        self.activ = activ
        self.rnn = None
        self.h = None
        if(rnnType == 'LSTM'):
            self.rnn = nn.LSTM(input_size=fsi+fai,
                               hidden_size=fh,
                               num_layers=num_layers)
        elif(rnnType == 'GRU'):
            self.rnn = nn.GRU(input_size=fsi+fai,
                              hidden_size=fh,
                              num_layers=num_layers)
        
        self.nh1 = nn.Linear(fh,fh)
        
        self.ns1 = nn.Linear(fh,fh)
        self.ns2 = nn.Linear(fh,fh)
        self.ns3 = nn.Linear(fh,fsi)
        
        self.nr1 = nn.Linear(fh,fh)
        self.nr2 = nn.Linear(fh,fh)
        self.nr3 = nn.Linear(fh,1)
        
        self.ndone1 = nn.Linear(fh,fh)
        self.ndone2 = nn.Linear(fh,fh)
        self.ndone3 = nn.Linear(fh,1)
    
    def init_h(self,bs):
        if(self.rnnType == 'LSTM'):
            return (torch.zeros(self.num_layers,bs,self.fh,
                                       device=device),
                           torch.zeros(self.num_layers,bs,self.fh,
                                       device=device))
        elif(self.rnnType == 'GRU'):
            return torch.zeros(self.num_layers,bs,self.fh,
                                      device=device)
            
    def forward(self,state,action):
#        if(self.h is None):
#            self.h = self.init_h(state.size()[0])
#        if(EVAL):
#            if(self.rnnType == 'LSTM'):
#                idx = np.random.randint(self.h[1].size()[1])
#                h = (self.h[0][:,idx].unsqueeze(1),
#                     self.h[1][:,idx].unsqueeze(1))
#            elif(self.rnnType == 'GRU'):
#                idx = np.random.randint(self.h.size()[1])
#                h = self.h[:,idx].unsqueeze(1)
#            x,_ = self.rnn(torch.cat((state,action),1)\
#                           .unsqueeze(0),h)
#        else:
#            x,h = self.rnn(torch.cat((state,action),1)\
#                           .unsqueeze(0),self.h)
#            if(self.rnnType == 'LSTM'):
#                self.h = (h[0].detach(),h[1].detach())
#            elif(self.rnnType == 'GRU'):
#                self.h = h.detach()
        x = torch.cat((state,action),1).unsqueeze(0)
        x = ACTIV[self.activ[0]](self.nh1(x.squeeze(0)))
        
        xs = ACTIV[self.activ[1]](self.ns1(x))
        xs = ACTIV[self.activ[2]](self.ns2(xs))
        xs = ACTIV[self.activ[3]](self.ns3(xs))
        
        xr = ACTIV[self.activ[4]](self.nr1(x))
        xr = ACTIV[self.activ[5]](self.nr2(xr))
        xr = ACTIV[self.activ[6]](self.nr3(xr))
        
        xd = ACTIV[self.activ[7]](self.ndone1(x))
        xd = ACTIV[self.activ[8]](self.ndone2(xd))
        xd = ACTIV[self.activ[9]](self.ndone3(xd))
        
        return xs,xr,xd
        
        
class Universe(object):
    # World = animal,environment,theories
    
    def __init__(self,parallel,life,nStates,nActions,
                 forbidden_fruit=10,adult=100,blank_paper_ad=2,
                 Temperature1=2,Temperature2=1,granularity=4):
        self.parallel = parallel
        self.life = life
        self.nStates = nStates
        self.nActions = nActions
        self.time = forbidden_fruit
        self.adult = adult
        self.blank_paper_ad = blank_paper_ad
        self.Temperature1=Temperature1
        self.Temperature2=Temperature2
        self.granularity = granularity
        
    def bigBang(self,rnnType,num_layers,fsi,fai,fh):
        self.worlds={}
        self.children={}
        self.prototype=BayesHypo(rnnType,num_layers,fsi,fai,fh).to(device)
        for i in range(self.parallel):
            model = BayesHypo(rnnType,num_layers,fsi,fai,fh).to(device)
            model.optimizer=model.optimizer(model.parameters())
            self.worlds[model]=-1
        for i in range(self.life):
            model = BayesHypo(rnnType,num_layers,fsi,fai,fh).to(device)
            model.optimizer=model.optimizer(model.parameters())
            self.children[model]=-1
    
    def world(self,Temp=1):
        lookup = self.worlds.keys()
        losses = -np.array(list(self.worlds.values()))
        prob = np.exp((losses - max(losses))/Temp)
        prob = prob / np.sum(prob)
        ret = np.random.choice(list(lookup),p=prob)
#        if(np.random.rand(1) < 0.0001):
#            print("world {} loss {}".format(id(self.worlds[ret]),self.worlds[ret]))
        return ret
            
    def develop(self,memory,batch_size,stablizer=0.1):
        
        def drama_life(model,memory):
            transitions = memory.sample(batch_size)
            batch = TransitionC(*zip(*transitions))
            state_batch,action_batch,\
            next_s_batch,reward_batch,done_batch=\
            [torch.cat(x).to(device) for x in batch]
            state_p,reward_p,done_p=model(one_hot(state_batch,self.nStates),
                                          one_hot(action_batch,self.nActions))
            loss = nn.CrossEntropyLoss()(state_p,next_s_batch)
#            +\
#                   nn.MSELoss()(reward_p,reward_batch)+\
#                   nn.BCEWithLogitsLoss()(done_p.squeeze(-1),done_batch.float())
            model.optimizer.zero_grad()
            loss.backward()
            model.optimizer.step()
            return loss.item()
        
        self.time-=1
        weed_out =[None,-np.inf]
        join_in =[None,np.inf]
        for model,loss in self.worlds.items():
            actual_loss = drama_life(model,memory)
            if(self.worlds[model] == -1):
                self.worlds[model]=actual_loss
            else:
                self.worlds[model]=loss+stablizer*(actual_loss-loss)
            if(self.worlds[model] > weed_out[1]):
                weed_out = [model,loss]
        ret_loss = weed_out[1]
        for i in range(self.blank_paper_ad):
            for model,loss in self.children.items():
                actual_loss = drama_life(model,memory)
                if(self.children[model] == -1):
                    self.children[model]=actual_loss
                else:
                    self.children[model]=loss+stablizer*(actual_loss-loss)
                if(i == self.blank_paper_ad-1 and
                   self.children[model] < join_in[1]):
                    join_in = [model,loss]
        if(self.time < 0 and weed_out[1] >= join_in[1]):
            self.time = self.adult
            del self.worlds[weed_out[0]]
            self.worlds[join_in[0]]=join_in[1]
            del self.children[join_in[0]]
            print("\tweed_out {} {}".format(id(weed_out[0]),weed_out[1]))
            print("\tjoin_in {} {}".format(id(join_in[0]),join_in[1]))
            self.reproduce(self.Temperature1)
        elif(self.time < 0):
            print("\tweed_out {} {}".format(id(weed_out[0]),weed_out[1]))
            print("\tjoin_in {} {}".format(id(join_in[0]),join_in[1]))
            self.nirvana(self.Temperature2)
            self.time = self.adult
        return ret_loss
    
    def nirvana(self,Temperature):
        print("\tExtinction")
        for model in list(self.children.keys()):
            del self.children[model]
            del model
        for i in range(self.life):
            self.reproduce(self.Temperature2)
        print("\tReborn")
        
    
    def reproduce(self,Temperature):
        print("\tReproduce")
        lookup=list(self.worlds.keys())
        prob=np.array(list(self.worlds.values()))/float(Temperature)
        prob=np.exp(prob - max(prob))
        prob=prob / np.sum(prob)
        parents=np.random.choice(lookup,2,replace=False,p=prob)
        print("\tParents {} {}".format(id(parents[0]),id(parents[1])))
        new_born = BayesHypo(parents[0].rnnType,
                             parents[0].num_layers,
                             parents[0].fsi,
                             parents[0].fai,
                             parents[0].fh).to(device)
        new_born.optimizer=new_born.optimizer(new_born.parameters())
        cut = np.random.choice(len(list(self.prototype.parameters())),
                               size=random.randint(1,self.granularity),
                               replace=False)
        print("\tDNA dissection {}".format(cut))
        father,mother = np.random.choice(parents,2,replace=False)
        DNA = OrderedDict()
        dominant = father
        mutation = np.random.randint(len(list(self.prototype.parameters())))
        print("\tMutation {}".format(mutation))
        for i,key in enumerate(father.state_dict().keys()):
            if(i == mutation):
                DNA[key] = self.prototype.state_dict()[key].clone()
            else:
                DNA[key] = dominant.state_dict()[key].clone()
            if(i in cut):
                if(dominant is father):
                    dominant = mother
                else:
                    dominant = father
                
        new_born.load_state_dict(DNA)
        self.children[new_born] = -1
        print("\tEnd of Reproduction")
        
#class dyna(object):
#    
        

class BayesSampling(object):
    
    def __init__(self,rl,universe):
        self.rl = rl
        self.universe = universe
        
        
    def bayesSampling(self,memory_cap,batch_size,
                      s0=0,nEpisodes=1000,epsilon=0.1,temperature=1,
                      computation_cost_train=10,
                      computation_cost_plan=10,
                      quantum_limit_time=10):
        memory = ReplayBufferC(memory_cap)
        self.universe.bigBang("LSTM",1,
                              self.rl.mdp.nStates,
                              self.rl.mdp.nActions,
                              self.rl.mdp.nStates+self.rl.mdp.nActions)
        initialQ = torch.zeros(self.rl.mdp.nActions,
                               self.rl.mdp.nStates,
                               dtype=torch.float,
                               device=device)
        Q = initialQ
        n_table = torch.ones(initialQ.size(),device=device)
        r_table = torch.zeros(initialQ.size(),
                              dtype=torch.float,
                              device=device)
        learning_rate = None
        epId = 0
        reward_ep = []
        while(epId < nEpisodes):
            epId += 1
            s = s0
            reward_cum = 0
            discount_factor = 1
            if(len(memory) > batch_size and epId % 10 == 0):
#                times = min(int(computation_cost_train * len(memory) / batch_size),1000)
                for i in trange(1000):
                    self.universe.develop(memory,batch_size,stablizer=0.1)
#            if(len(memory) > batch_size and epId >= 10):
#                for i in range(computation_cost_plan):
#                    # can use more than 1 step
#                    V = torch.sum(F.softmax(Q,0) * Q,dim=0)
#                    state_tensor = one_hot(torch.arange(self.rl.mdp.nStates)\
#                                           .unsqueeze(1).expand(-1,self.rl.mdp.nActions)\
#                                           .flatten(),
#                                           self.rl.mdp.nStates)
#                    action_tensor = one_hot(torch.arange(self.rl.mdp.nActions)\
#                                            .repeat(self.rl.mdp.nStates),
#                                            self.rl.mdp.nActions)
#                    V_next_expected=torch.zeros(Q.size(),dtype=torch.float,
#                                                device=device)
#                    for j in range(quantum_limit_time):
#                        world = self.universe.world()
#                        with torch.no_grad():
#                            [sample_s_next,sample_reward,sample_done] = \
#                                world(state_tensor,action_tensor)
##                                print(sample_s_next[0])
#                            sample_s_next.scatter_(-1,sample_s_next.topk(59,-1,largest=False)[1],-1000.)
##                                print(F.softmax(sample_s_next,-1)[0])
#                            V_next = torch.sum(F.softmax(sample_s_next,dim=-1)*V,-1)
#                            print(F.softmax(sample_s_next,dim=-1))
#                            print(V_next)
#                            V_next=(r_table.flatten()+self.rl.mdp.discount*\
#                                V_next).view(self.rl.mdp.nStates,
#                                      self.rl.mdp.nActions).transpose(0,1)
#                            V_next_expected+=V_next/(float(quantum_limit_time))
#                    Q = Q + (V_next_expected - Q)/ n_table
#                    Q[:,self.rl.mdp.E] = 0
            for i in range(100):
#                action=None
                action = torch.multinomial(F.softmax(Q[:,s],dim=-1),1)[0].item()
#                if(np.random.rand(1) < epsilon):
#                    action = np.random.randint(self.rl.mdp.nActions)
#                elif(temperature != 0):
#                    action = torch.multinomial(F.softmax(Q[:,s],dim=-1),1)[0].item()
##                    boltz_state = Q[:,s].flatten()
##                    boltz_state = np.exp((boltz_state-np.max(boltz_state)) / temperature)
##                    boltz_state = np.cumsum(boltz_state / boltz_state.sum())
##                    action = np.where(boltz_state >= np.random.rand(1))[0][0]
#                else:
#                action = Q[:,s].argmax()
                [s_next,reward,done] = self.rl.sampleRewardAndNextState(s,action)
                reward_cum += discount_factor * reward
                discount_factor *= self.rl.mdp.discount
                print("\rStep: {:3d} state {:2d} action {} next_state {:2d} reward {:08.3f} done {:5} cumu_reward {:8.3f}\t\t"\
                      .format(i,s,action,s_next,reward,done,reward_cum),end=' ')
                memory.push(torch.tensor([s],device=device),
                            torch.tensor([action],device=device),
                            torch.tensor([s_next],device=device),
                            torch.tensor([reward],device=device),
                            torch.tensor([done],device=device))
                learning_rate = 1 / n_table[action,s].item()
                Q[action,s] = Q[action,s] + learning_rate*(reward +\
                 self.rl.mdp.discount*torch.max(Q[:,s_next])-Q[action,s])
                if(len(memory) > batch_size and epId >= 10):
                    for i in range(computation_cost_plan):
                        # can use more than 1 step
                        sim_s = s0
                        sim_action=None
                        if(np.random.rand(1) < 0.3):
                            sim_action = np.random.randint(self.rl.mdp.nActions)
                        else:
                            sim_action = torch.multinomial(F.softmax(Q[:,s],dim=-1),1)[0].item()
                        V = torch.sum(F.softmax(Q,0) * Q,dim=0)
                        for j in range(quantum_limit_time):
                            world = self.universe.world()
                            with torch.no_grad():
                                [sample_s_next,_,_]=\
                                    world(one_hot(torch.tensor([sim_s],device=device),
                                                  self.rl.mdp.nStates),
                                          one_hot(torch.tensor([sim_action],device=device),
                                                  self.rl.mdp.nActions))
                                    V_next = torch.sum(F.softmax(torch.clamp(sample_s_next,min=0.,max=200.),dim=-1)*V,-1)
                

                    

#                        for it_state in state_range:
#                            # can record the frequency and add a ucb bound or add reward
#                            if(it_state not in self.rl.mdp.E):
#                                action_range=list(range(self.rl.mdp.nActions))
#                                random.shuffle(action_range)
#                                for it_action in action_range:
#                                    temporal_error = 0
#                                    for j in range(quantum_limit_time):
#                                        with torch.no_grad():
#                                            world = self.universe.world(Temp=1)
#                                            [sample_s_next, sample_reward, sample_done] =\
#                                            world(one_hot(torch.tensor([[it_state]],
#                                                                       device=device),
#                                                          self.rl.mdp.nStates),
#                                                  one_hot(torch.tensor([[it_action]],
#                                                                       device=device),
#                                                          self.rl.mdp.nActions))
#                                                
#                                            V_next=torch.sum(F.softmax(sample_s_next[0],-1)*V).item()
#                                            sample_reward=sample_reward[0][0].item()
#                                            temporal_error+=(sample_reward+\
#                                                               self.rl.mdp.discount*V_next-\
#                                                                 Q[it_action][it_state])
#                                    Q[it_action][it_state] = Q[it_action,it_state] + \
#                                      (1./(n_table[it_action,it_state]*quantum_limit_time))*\
#                                        temporal_error
                                            
#                                        done=(np.random.rand() < \
#                                              torch.sigmoid(done[0][0]).item())
                                
#                        world = self.universe.world(Temp=1)
#                        Q,_,_=self.qLearning(s0,Q,1,epsilon,temperature,
#                                             world,1)
                r_table[s,action] += (reward-r_table[s,action])/n_table[action,s]
                n_table[action,s] += 1
                s=s_next
                if(done):
                    break
            reward_ep.append(reward_cum)
            print("\nEpisodes: {} reward: {}".format(epId,reward_cum))
            print(Q.detach().data)
        Q=Q.detach().cpu().numpy()
        policy = Q.argmax(0)
        return [Q,policy, reward_ep]
        
        
        
    def qLearning(self,s0,initialQ,nEpisodes,epsilon=0.1,temperature=1,
                  world=None,lr=1):
        '''qLearning algorithm.  Epsilon exploration and Boltzmann exploration
        are combined in one procedure by sampling a random action with 
        probabilty epsilon and performing Boltzmann exploration otherwise.  
        When epsilon and temperature are set to 0, there is no exploration.

        Inputs:
        s0 -- initial state
        initialQ -- initial Q function (|A|x|S| array)
        nEpisodes -- # of episodes (one episode consists of a trajectory of nSteps that starts in s0
        nSteps -- # of steps per episode
        epsilon -- probability with which an action is chosen at random
        temperature -- parameter that regulates Boltzmann exploration

        Outputs: 
        Q -- final Q function (|A|x|S| array)
        policy -- final policy
        '''

        # temporary values to ensure that the code compiles until this
        # function is coded
        Q = initialQ
        n_table = np.zeros(Q.shape,dtype=int)
        learning_rate = 0
        lr = lr
        episodeId = 0
        reward_episodes = []
        if world is None:
            world = self.rl.mdp.sampleRewardAndNextState
        while (episodeId < nEpisodes):
          episodeId += 1
          s=s0
          reward_cum=0
          discount_factor = 1
          for i in range(100):
            action = 0
            discount_factor *= self.rl.mdp.discount
            if (np.random.rand(1) < epsilon):
              action = np.random.randint(self.rl.mdp.nActions)
            elif (temperature != 0):
              boltz_state = Q[:,s].flatten()
              boltz_state = np.exp( (boltz_state-np.max(boltz_state)) / temperature)
              boltz_state = np.cumsum(boltz_state / boltz_state.sum())
              action = np.where(boltz_state >= np.random.rand(1))[0][0]
            else:
              action = Q[:,s].argmax()
            with torch.no_grad():
                [s_next, reward, done] =\
                world(one_hot(torch.tensor([[s]],device=device),self.rl.mdp.nStates),
                      one_hot(torch.tensor([[action]],device=device),self.rl.mdp.nActions),
                      EVAL=True)
                    
                s_next = torch.multinomial(F.softmax(s_next[0],-1),1)[0].item()
                reward=reward[0][0].item()
                done=(np.random.rand() < torch.sigmoid(done[0][0]).item())
            n_table[action,s] += 1
            learning_rate = float(lr) / n_table[action,s]
            Q[action,s] = Q[action,s] + learning_rate*(reward + self.rl.mdp.discount*np.max(Q[:,s_next].flatten())-Q[action,s])
            s = s_next
            reward_cum += reward
#            reward_cum += discount_factor * reward
            if(done):
                break
          reward_episodes.append(reward_cum)
#          print("Sim Episodes: {} reward: {}".format(episodeId,reward_cum))
        policy = Q.argmax(0).flatten()
        return [Q,policy, reward_episodes]

if __name__ == "__main__":
    import time
    import matplotlib.pyplot as plt
    seed=1234567891
    np.random.seed(seed)
    # Discount factor: scalar in [0,1)
    discount = 0.95
    ax=plt.figure(figsize=(15,10))
    plt.grid()
    plt.xlabel("Episodes")
    plt.ylabel("Rewards")
    H=np.load('QLearning100.npy')
    plt.plot(range(len(H)),H,label="Q")
    
    H = []
    for i in range(10):
        start = time.time()
                
        # MDP object
        [T,R,E] = maze_generator()
        mdp = MDP(T,R,E,discount)
        
        # RL problem
        rlProblem = RL(mdp,np.random.normal)
        universe = Universe(10,5,rlProblem.mdp.nStates,rlProblem.mdp.nActions,
                            forbidden_fruit=50,adult=100,blank_paper_ad=1,
                            Temperature1=2,Temperature2=1,
                            granularity=4)
        bayesSampling = BayesSampling(rlProblem,universe)
        q,p,h = bayesSampling.bayesSampling(10000,128,s0=0,nEpisodes=200,
                                            epsilon=0.1,temperature=1,
                                            computation_cost_train=100.,
                                            computation_cost_plan=100,
                                            quantum_limit_time=10)
        H.append(h)
        print("Experiment {} 100 episodes_mean {} time {}"\
              .format(i,np.mean(h[-100:]),time.time()-start))
    np.save('H2',np.array(H))
#    H = np.load('H.npy')
    H=np.mean(H,0)
    print(q)
    plt.plot(range(len(H)),H,label="BQ2")
    plt.savefig("BQ2.png")
    plt.show()
    #np.save('QLearning100',np.array(H))
    