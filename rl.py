import numpy as np
from mdp import MDP
class RL:
    def __init__(self,mdp,sampleReward):
        '''Constructor for the RL class

        Inputs:
        mdp -- Markov decision process (T, R, discount)
        sampleReward -- Function to sample rewards (e.g., bernoulli, Gaussian).
        This function takes one argument: the mean of the distributon and 
        returns a sample from the distribution.
        '''

        self.mdp = mdp
        self.sampleReward = sampleReward
        self.COUNT = 0

    def sampleRewardAndNextState(self,state,action):
        '''Procedure to sample a reward and the next state
        reward ~ Pr(r)
        nextState ~ Pr(s'|s,a)

        Inputs:
        state -- current state
        action -- action to be executed

        Outputs: 
        reward -- sampled reward
        nextState -- sampled next state
        '''

        cumProb = np.cumsum(self.mdp.T[action,state,:])
        nextState = np.where(cumProb >= np.random.rand(1))[0][0]
        reward = self.sampleReward(self.mdp.R[action,nextState])
        done = False
        self.COUNT+=1
        if(nextState in self.mdp.E):
            self.COUNT = 0
            done = True
        elif(self.COUNT == 100):
            self.COUNT = 0
            done = True
        return [nextState,reward,done]

    def sampleSoftmaxPolicy(self,policyParams,state):
        '''Procedure to sample an action from stochastic policy
        pi(a|s) = exp(policyParams(a,s))/[sum_a' exp(policyParams(a',s))])
        This function should be called by reinforce() to selection actions

        Inputs:
        policyParams -- parameters of a softmax policy (|A|x|S| array)
        state -- current state

        Outputs: 
        action -- sampled action
        '''

        # temporary value to ensure that the code compiles until this
        # function is coded
        pi = policyParams[:,state]
        pi = pi - np.max(pi)
        pi = np.exp(pi) / np.sum(np.exp(pi))
        action = np.where(np.cumsum(pi) >= np.random.rand(1))[0][0]
                          
        return action

    def modelBasedRL(self,s0,defaultT,initialR,nEpisodes,nSteps,epsilon=0):
        '''Model-based Reinforcement Learning with epsilon greedy 
        exploration.  This function should use value iteration,
        policy iteration or modified policy iteration to update the policy at each step

        Inputs:
        s0 -- initial state
        defaultT -- default transition function when a state-action pair has not been vsited
        initialR -- initial estimate of the reward function
        nEpisodes -- # of episodes (one episode consists of a trajectory of nSteps that starts in s0
        nSteps -- # of steps per episode
        epsilon -- probability with which an action is chosen at random

        Outputs: 
        V -- final value function
        policy -- final policy
        '''

        # temporary values to ensure that the code compiles until this
        # function is coded
        count_triple = np.ones([self.mdp.nActions, self.mdp.nStates, self.mdp.nStates])
        cumu_reward_lst = np.zeros(nEpisodes)
        V = np.zeros(self.mdp.nStates)
        policy = np.zeros(self.mdp.nStates,int)
        mdp_tmp = MDP(defaultT, initialR, self.mdp.E,self.mdp.discount)
        
        for iterEp in range(nEpisodes):
            state = s0
            for iterSt in range(nSteps):
                action = 0
                if np.random.rand(1) < epsilon:
                    action = np.random.randint(self.mdp.nActions)
                else:
                    action = policy[state]
                [nextState,reward,done] = self.sampleRewardAndNextState(state,action)
                cumu_reward_lst[iterEp] += self.mdp.discount**iterSt * reward
                count_triple[action,state,nextState] += 1
                count_double = np.sum(count_triple[action,state,:])
                mdp_tmp.T[action,state,:] = count_triple[action,state,:] /  count_double
                mdp_tmp.R[action, state] = (reward + (count_double-1) * mdp_tmp.R[action,state]) / count_double
                [policy, V, iterId] = mdp_tmp.policyIteration(policy)
                state = nextState
                if(done):
                    break
        return [V,policy,cumu_reward_lst] 

    def reinforce(self,s0,initialPolicyParams,nEpisodes,nSteps):
        '''reinforce algorithm.  Learn a stochastic policy of the form
        pi(a|s) = exp(policyParams(a,s))/[sum_a' exp(policyParams(a',s))]).
        This function should call the function sampleSoftmaxPolicy(policyParams,state) to select actions

        Inputs:
        s0 -- initial state
        initialPolicyParams -- parameters of the initial policy (array of |A|x|S| entries)
        nEpisodes -- # of episodes (one episode consists of a trajectory of nSteps that starts in s0)
        nSteps -- # of steps per episode

        Outputs: 
        policyParams -- parameters of the final policy (array of |A|x|S| entries)
        '''

        # temporary values to ensure that the code compiles until this
        # function is coded
        policyParams = initialPolicyParams
        cumu_reward_lst = []
        count_lst = np.zeros([self.mdp.nActions, self.mdp.nStates])
        
        for iterEp in range(nEpisodes):
            G = np.zeros(nSteps)
            state = s0
            state_lst = []
            action_lst = []
            reward_lst = []
            for iterSt in range(nSteps):
                action = self.sampleSoftmaxPolicy(policyParams, state)
                [next_state,reward,done] = self.sampleRewardAndNextState(state, action)
                state_lst.append(state)
                action_lst.append(action)
                reward_lst.append(reward)
                state = next_state
            G[-1] = reward_lst[-1]
            for iter in range(nSteps-2, -1, -1):
                G[iter]=G[iter+1]*self.mdp.discount+reward_lst[iter]
            cumu_reward_lst.append(G[0])
            for iter in range(nSteps):
                count_lst[action_lst[iter], state_lst[iter]] += 1
                pi = policyParams[:,state_lst[iter]]
                pi -= np.max(pi)
                pi = np.exp(pi)
                pi = pi / np.sum(pi)
                gradient = -pi
                gradient[action_lst[iter]] += 1
                policyParams[:,state_lst[iter]] += 0.01 * (self.mdp.discount ** iter) * G[iter] * gradient
        return policyParams, cumu_reward_lst   

    def qLearning(self,s0,initialQ,nEpisodes,epsilon=0.05,temperature=1):
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
        episodeId = 0
        reward_episodes = []
        while (episodeId < nEpisodes):
          episodeId += 1
          s=s0
          reward_cum=0
          discount_factor = 1
          while (True):
            action = 0
            if (np.random.rand(1) < epsilon):
              action = np.random.randint(self.mdp.nActions)
            elif (temperature != 0):
              boltz_state = Q[:,s].flatten()
              boltz_state = np.exp((boltz_state-np.max(boltz_state)) / temperature)
              boltz_state = np.cumsum(boltz_state / boltz_state.sum())
              action = np.where(boltz_state >= np.random.rand(1))[0][0]
            else:
              action = Q[:,s].argmax()
            [s_next, reward, done] = self.sampleRewardAndNextState(s,action)
            n_table[action,s] += 1
            learning_rate = 1 / n_table[action,s]
            Q[action,s] = Q[action,s] + learning_rate*(reward + self.mdp.discount*np.max(Q[:,s_next].flatten())-Q[action,s])
            s = s_next
#            reward_cum += reward
            reward_cum += discount_factor * reward
            discount_factor *= self.mdp.discount
            if(done):
                break
          reward_episodes.append(reward_cum)
        
        policy = Q.argmax(0).flatten()

        return [Q,policy, reward_episodes]