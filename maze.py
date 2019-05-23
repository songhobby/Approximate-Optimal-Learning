''' Construct a simple maze MDP

  Grid world layout:

  -----------------------------------------
  |  0 |  1 |  2 |  3 |  4 |  5 |  6 |  7 |
  -----------------------------------------
  |  8 |  9 | 10 | 11 | 12 | 13 | 14 | 15 |
  -----------------------------------------
  ...
  -----------------------------------------
  | 56 | 57 | 58 | 59 | 60 | 61 | 62 | 63 |

  4 Goal states
  4 Bad states
  
  Game ends when the any goal state is reached
  
  4 actions (up-0, down-1, left-2, right-3).'''

# Transition function: |A| x |S| x |S'| array
import numpy as np
MAZE_SIDE_LENGTH=8
FAVOR_FACTOR=4
ACTION_SIZE=4
UP=0
DOWN=1
LEFT=2
RIGHT=3

def multi_nomial_prob_generator(action):
    alpha=[1]*(ACTION_SIZE+1)
    alpha[np.random.randint(ACTION_SIZE)]=FAVOR_FACTOR
    return np.random.dirichlet(alpha)
                              
def look_around(state):
    return [state-MAZE_SIDE_LENGTH,state+MAZE_SIDE_LENGTH,
            state-1,state+1,state]
    
def naive_pred(state,action):
    if(action==UP):
        s_n = state - 8
    elif(action==DOWN):
        s_n = state + 8
    elif(action==LEFT):
        s_n = state - 1
    elif(action==RIGHT):
        s_n = state + 1
    else:
        raise Exception('Unknown Action')
    if(s_n < 0 or s_n > 63):
        s_n = state
    elif(state % 8 == 7 and s_n % 8 == 1):
        s_n = state
    elif(state % 8 == 1 and s_n % 8 == 7):
        s_n = state
    return s_n
    
def maze_generator():
    T = np.zeros([ACTION_SIZE,MAZE_SIDE_LENGTH**2,MAZE_SIDE_LENGTH**2])
    
    # 0
    for action_i in range(ACTION_SIZE):
        prob=multi_nomial_prob_generator(action_i)
        T[action_i][0][1]=prob[RIGHT]
        T[action_i][0][MAZE_SIDE_LENGTH]=prob[DOWN]
        T[action_i][0][0]=1-prob[RIGHT]-prob[DOWN]
    # 7
    for action_i in range(ACTION_SIZE):
        prob=multi_nomial_prob_generator(action_i)
        T[action_i][MAZE_SIDE_LENGTH-1][MAZE_SIDE_LENGTH-2]=prob[LEFT]
        T[action_i][MAZE_SIDE_LENGTH-1][2*MAZE_SIDE_LENGTH-1]=prob[DOWN]
        T[action_i][MAZE_SIDE_LENGTH-1][MAZE_SIDE_LENGTH-1]=\
        1-prob[LEFT]-prob[DOWN]
        
    # 56
    for action_i in range(ACTION_SIZE):
        prob=multi_nomial_prob_generator(action_i)
        T[action_i][MAZE_SIDE_LENGTH**2-MAZE_SIDE_LENGTH]\
        [MAZE_SIDE_LENGTH**2-2*MAZE_SIDE_LENGTH]=prob[UP]
        T[action_i][MAZE_SIDE_LENGTH**2-MAZE_SIDE_LENGTH]\
        [MAZE_SIDE_LENGTH**2-MAZE_SIDE_LENGTH+1]=prob[RIGHT]
        T[action_i][MAZE_SIDE_LENGTH**2-MAZE_SIDE_LENGTH]\
        [MAZE_SIDE_LENGTH**2-MAZE_SIDE_LENGTH]=1-prob[UP]-prob[RIGHT]
        
    # 63
    for action_i in range(ACTION_SIZE):
        prob=multi_nomial_prob_generator(action_i)
        T[action_i][MAZE_SIDE_LENGTH**2-1]\
        [MAZE_SIDE_LENGTH**2-MAZE_SIDE_LENGTH-1]=prob[UP]
        T[action_i][MAZE_SIDE_LENGTH**2-1]\
        [MAZE_SIDE_LENGTH**2-2]=prob[LEFT]
        T[action_i][MAZE_SIDE_LENGTH**2-1]\
        [MAZE_SIDE_LENGTH**2-1]=1-prob[UP]-prob[LEFT]
        
    # 1-6
    for state_i in range(1,MAZE_SIDE_LENGTH-1):
        for action_i in range(ACTION_SIZE):
            prob=multi_nomial_prob_generator(action_i)
            la=look_around(state_i)
            for la_i in range(ACTION_SIZE):
                if(la[la_i] >= 0):
                    T[action_i][state_i][la[la_i]]=prob[la_i]
            T[action_i][state_i][state_i] = prob[UP]+prob[-1]
    
    # 57-62
    for state_i in range(MAZE_SIDE_LENGTH**2-MAZE_SIDE_LENGTH+1,
                         MAZE_SIDE_LENGTH**2-1):
        for action_i in range(ACTION_SIZE):
            prob=multi_nomial_prob_generator(action_i)
            la=look_around(state_i)
            for la_i in range(ACTION_SIZE):
                if(la[la_i] <= MAZE_SIDE_LENGTH**2-1):
                    T[action_i][state_i][la[la_i]]=prob[la_i]
            T[action_i][state_i][state_i] = prob[DOWN]+prob[-1]
            
    # (8,56,8)
    for state_i in range(MAZE_SIDE_LENGTH,
                         MAZE_SIDE_LENGTH**2-MAZE_SIDE_LENGTH,
                         MAZE_SIDE_LENGTH):
        for action_i in range(ACTION_SIZE):
            prob=multi_nomial_prob_generator(action_i)
            la=look_around(state_i)
            for la_i in range(ACTION_SIZE):
                if(la[la_i] % MAZE_SIDE_LENGTH != (MAZE_SIDE_LENGTH-1)):
                    T[action_i][state_i][la[la_i]]=prob[la_i]
            T[action_i][state_i][state_i] = prob[LEFT]+prob[-1]
            
    # (15,63,8)
    for state_i in range(2*MAZE_SIDE_LENGTH-1,
                         MAZE_SIDE_LENGTH**2-1,
                         MAZE_SIDE_LENGTH):
        for action_i in range(ACTION_SIZE):
            prob=multi_nomial_prob_generator(action_i)
            la=look_around(state_i)
            for la_i in range(ACTION_SIZE):
                if(la[la_i] % MAZE_SIDE_LENGTH != 0):
                    T[action_i][state_i][la[la_i]]=prob[la_i]
            T[action_i][state_i][state_i] = prob[RIGHT]+prob[-1]

    # for the rest of the states 
    for start_i in range(MAZE_SIDE_LENGTH+1,
                         MAZE_SIDE_LENGTH**2-MAZE_SIDE_LENGTH+1,
                         MAZE_SIDE_LENGTH):
        for state_i in range(start_i,start_i+MAZE_SIDE_LENGTH-2):
            for action_i in range(ACTION_SIZE):
                prob=multi_nomial_prob_generator(action_i)
                la=look_around(state_i)
                for la_i in range(ACTION_SIZE):
                    T[action_i][state_i][la[la_i]]=prob[la_i]
                T[action_i][state_i][state_i] = prob[-1]
    # Reward function: |A| x |S| array
    R = -1 * np.ones([ACTION_SIZE,MAZE_SIDE_LENGTH**2]);
    
    spec=np.random.permutation(np.arange(1,MAZE_SIDE_LENGTH**2))[:8]
    # set rewards
    # goal states
    R[:,spec[0]] = 100
    R[:,spec[1]] = 100
    R[:,spec[2]] = 200
    R[:,spec[3]] = 200
    E = spec[0:4]
    # bad states
    R[:,spec[4]] = -10
    R[:,spec[5]] = -10
    R[:,spec[6]] = -20
    R[:,spec[7]] = -20
    return [T,R,E]
