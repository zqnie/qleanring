# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 23:37:33 2019

@author: Thinkpad
"""


from IPython.display import clear_output
from collections import deque
import progressbar
import numpy as np
import pandas as pd
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Embedding, Reshape
from tensorflow.keras.optimizers import Adam

class env:
    def __init__(self,act,state_size,ticker1,ticker2,Return,Price,Value):
        """
        act: list of all possible actions
        state_size: size of input state
        ticker1,ticker2: name of both stocks
        Return: df of all stock returns
        Price: df of all stock prices
        Value: inital total value of money
        """
        self.action_size=len(act)
        self.action=act
        self.state_size=state_size
        self.r1,self.r2=np.array(Return[ticker1]),np.array(Return[ticker2])
        self.p1,self.p2=np.array(Price[ticker1]),np.array(Price[ticker2])
        self.Value=Value
        self.total_reward=0
#        self.n=int((state_size-3)/2) #how many days of history we are using
        self.n=5
    
    def Reset(self,T=0,random=1):
        """
        Reset initial state
        n: number of days of history
        T: reset weights for time T, typically the starting date of training
        """
        if random==1:
            w0=np.random.uniform(0,1)
        else:
            w0=0.5
        self.w=[w0,1-w0]
        TT=self.n+1+T # so you should just start at T=0
        self.total_reward=0
        
        his1=list(self.r1[TT-self.n-1:TT-1])
        his2=list(self.r2[TT-self.n-1:TT-1])
        
        initial=np.concatenate((his1,his2), axis=None)
        
        return initial
        
    def random_act(self):
        """
        randomly choose an action
        """
        return self.action.index(np.random.choice(self.action))

    def step(self,now_state,action,T):
        """
        return next_state, reward, terminated information if we take action from T-1 to T
        """
        now_state = now_state[0]
        #sell low beta, buy in corresponding high beta
        T=T
        w1=now_state[0:2]
#        total1=now_state[2] #previous state total value
#        
#                
        #update weights
        new_w=w1[0]+self.action[action]
        if new_w<0: new_w = 0
        if new_w>0: new_w = 1
        w2=[new_w,1-new_w]
#        
#        num1=w2[0]*total1/self.p1[T-1]
#        num2=w2[1]*total1/self.p2[T-1]
#        
#        
#        value1=num1*self.p1[T]
#        value2=num2*self.p2[T]
#        total2=value1+value2 # current total value
        

        his1=self.r1[T-self.n:T]
        his2=self.r2[T-self.n:T]
        
        #update state 
        next_state=np.concatenate((his1,his2), axis=None)
        
        #reward function
        t = 30 # rolling window
        ret1 = self.r1[T-t:T]
        ret2 = self.r2[T-t:T]
        exp_ret = np.matrix([[np.mean(ret1)],[np.mean(ret2)]])
        cov_mat = np.cov(ret1,ret2)
        reward = np.dot(w2,exp_ret)/np.sqrt(np.dot(np.dot(w2,cov_mat),np.matrix(w2).T))  # use new weights or old weights?
        self.total_reward=self.total_reward+reward  #?
        #print("take action:",action," and got reward:",reward)
        
        #judge termination state:
        if T==len(self.p1)-3:
            terminated=True
        else:
            terminated=False
        
        return next_state,reward,terminated,(reward>0)
        

