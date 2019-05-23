#%%
from __future__ import print_function
import torch

#%%
x=torch.randn([2,3])
print(x.size())
print(torch.cuda.is_available())

#%%
x = torch.ones(2,2,requires_grad=True)
print(x)

#%%
import torch.nn as nn
from torch.nn.utils.rnn import pack_sequence as pks
import torch.nn.functional as F
class DeepQN(nn.Module):
    
    def __init__(self,fbi,fsi,fso):
        super(DeepQN,self).__init__()
        self.belief1 = nn.Linear(fbi,fbi)
        self.belief2 = nn.Linear(fbi,fbi)
        self.belief_interm = nn.Linear(fbi,fbi)
        
        self.bs1 = nn.Linear(fbi+fsi,fbi+fsi)
        self.bs2 = nn.Linear(fbi+fsi,fbi+fsi)
        self.bs3 = nn.Linear(fbi+fsi,fso)
        self.bs4 = nn.Linear(fso,fso)
    
    def forward(self,belief,state):
        x = F.hardtanh(self.belief1(belief))
        x = F.hardtanh(self.belief2(x))
        x = F.hardtanh(self.belief_interm(x))
        x = torch.cat((x,state),0)
        
        x = F.hardtanh(self.bs1(x))
        x = F.hardtanh(self.bs2(x))
        x = F.softmax(self.bs4(self.bs3(x)))
        return x
    
dpqn = DeepQN(2,2,2)
import torch
import numpy as np
batch = (((2,3),(4,5),(6,9)),((4,7),(5,9),(4,4)))
batch = zip(*[list(zip(*ep)) for ep in batch])
batch = [torch.tensor(x) for x in batch]
list(batch)[0][:,0]
torch.tensor([2]).size()[0]
def one_hot(ts,dim):
    res=torch.zeros([ts.size()[0],dim])
    for i in range(ts.size()[0]):
        res[i,ts[i].type(torch.int)]=1
    return res
one_hot(torch.tensor([3,4,0]),5)
torch.div(torch.tensor([2.]), torch.tensor([4.]))
torch.tensor([1e-8])
def reward_lc(y_pred,y_targ):
    diff = torch.abs((y_pred - y_targ) / 
                     torch.clamp(torch.abs(y_targ),min=1e-8))
    return 100. * torch.mean(diff,-1)
reward_lc(torch.tensor([2,3],dtype=torch.float),torch.tensor([1,2],dtype=torch.float))
x=torch.tensor([[4,1],[3.,4.],[10,9]])
x.argmax(1)
x.gather(1,torch.tensor([[1],[0],[1]]))
torch.cat((torch.tensor([1]),torch.tensor([2]),torch.tensor([3])))
torch.multinomial(torch.tensor([[0.1,0.2,0.7]]),1,replacement=True)
torch.randn(1,1,2,3)
torch.tensor(list(zip(*[list(range(x)) for x in [3,3,3]])))
x[0::2]
import random

random.shuffle(list(range(4)))
a=torch.tensor([[2,4,5],[5,3,2],[2,5,0]],dtype=torch.float)
b=torch.tensor([3,9,3,2])
sample_s_next = torch.tensor([[2,3,4,5.,5.]])
sample_s_next = torch.max(sample_s_next[0],0)
print(sample_s_next)
#sample_s_next = sample_s_next[random.randint(sample_s_next.size()[0])].item()
#%%
from bayesSamplling import BayesHypo
from torchsummary import summary
model = BayesHypo('LSTM',1,10,4,7).to(device)
b=model(torch.zeros(100,10).to(device),torch.zeros(100,4).to(device))
#print(len(list(model.parameters())))
#for i,(a,b) in enumerate(model.named_parameters()):
#    print(i,a,b)
print(type(model.state_dict()))
print({'a':1,'b':2})
#%%
import numpy as np
import os
import matplotlib.pyplot as plt
ax=plt.figure(figsize=(15,10))
x=list(range(1000))
hq = np.load('/home/haobei/Project/SL/BayesianReasoning/dq.npy')
hq=hq[:1000]
lq = np.polyfit(x,hq,4)
plt.plot(x,hq,'y',label='Deep Q Learning with pre-trained deep Q net')

hbs = np.load('/home/haobei/Project/SL/BayesianReasoning/bdq.npy')
hbs=hbs[:1000]
lbs = np.polyfit(x,hbs,4)
plt.plot(x,hbs,'c',label='Bayesian belief based deep Q Learning with pre-trained deep Q net')

#
#hmb = np.load('/home/haobei/Project/SL/BayesianReasoning/MBLearning.npy')
#hmb=hmb[:200]
#lmb = np.polyfit(x,hmb,4)
#plt.plot(x,hmb,'g',label='Model based tabular Q Learning with frequentist model')
#
plt.plot(x,np.poly1d(lq)(x),'y',linewidth=4,label='(Polyfit) Deep Q Learning with pre-trained deep Q net')
plt.plot(x,np.poly1d(lbs)(x),'c',linewidth=4,label='(Polyfit) Bayesian belief based deep Q Learning with pre-trained deep Q net')
##plt.plot(x,np.poly1d(lmb)(x),'g',linewidth=4,label='(Polyfit) Model based tabular Q Learning with frequentist model')

a=np.sum(hq)
b=np.sum(hbs)
print((b-a)/a)
plt.grid()
plt.xlabel("Episodes (averaged over 100 runs)")
plt.ylabel("Cumulative rewards (discounted)")
plt.legend()
plt.savefig("BQL4.png")
plt.show()
#%%
import random
random.randint( 1,4)
#%%
import numpy as np