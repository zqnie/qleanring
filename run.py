#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 21:47:25 2019

@author: niyu
"""

from IPython.display import clear_output
from collections import deque
import progressbar
import numpy as np
import pandas as pd
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Embedding, Reshape
from tensorflow.keras.optimizers import Adam
import random

from environment import env
from agent import Agent
import os

os.chdir('C:\\Users\\Thinkpad\\Desktop\\proj')  

if __name__=='__main__':
    
    test = True
    highticker = 'TXN'
    lowticker = 'HCP'

    high_prices = pd.DataFrame(pd.read_csv(".\TXN.csv",index_col = 0)['Adj Close'])
    high_prices.columns = ['TXN']
    high_returns = np.log(high_prices).diff().dropna()

    low_prices = pd.DataFrame(pd.read_csv('.\HCP.csv',index_col = 0)['Adj Close'])
    low_prices.columns = ['HCP']
    low_returns = np.log(low_prices).diff().dropna()
    dates = list(set(high_prices.index).intersection(set(high_returns.index)))
    dates.sort()
    
    train_days = int(np.round(len(dates)*0.7))
    Return = high_returns.loc[dates,:].iloc[:train_days,]
    Return = pd.concat([Return,low_returns.loc[dates,:].iloc[:train_days,]],axis =1)
    Price = high_prices.loc[dates,:].iloc[:train_days,]
    Price = pd.concat([Price,low_prices.loc[dates,:].iloc[:train_days,]],axis =1)
    init_value = 100
    value = init_value
    optimizer = Adam(learning_rate=0.01)
    environment = env([-0.15,-0.1,-0.05,0,0.05,0.1,0.15],10,'TXN','HCP',Return,Price,value)
    agent = Agent(environment, optimizer)
    
    batch_size = 50
    num_of_episodes = 10
    timesteps_per_episode = int(len(Return)/num_of_episodes)
    agent.q_network.summary()
    
    for e in range(0, num_of_episodes):
        # Reset the environment
        
        if e == 0: start = 30
        else: start = 0

        state = environment.Reset(random = 1)
        state = np.reshape(state, [1, 10])
        
        # Initialize variables
        reward = 0
        terminated = False
        
#        bar = progressbar.ProgressBar(maxval=timesteps_per_episode/10, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
#        bar.start()
        
        for timestep in range(timesteps_per_episode*e+start,timesteps_per_episode*(e+1)):
            # Run Action
            action = agent.act(state)
            
            # Take action    
            next_state, reward, terminated, info = environment.step(state,action,timestep) 
            next_state = np.reshape(next_state, [1, 10])
            agent.store(state, action, reward, next_state, terminated)
            
            state = next_state
            if terminated:
                agent.alighn_target_model()
                break

            
        if len(agent.expirience_replay) > batch_size:
            agent.retrain(batch_size)
            
            
        
#        bar.finish()
        print("**********************************")
        print("Episode: {}".format(e + 1))
#            environment.render()
        print("**********************************")
    
    # test data
    if test:
        import seaborn as sns
        import datetime
        import matplotlib.pyplot as plt
        test_return = high_returns.loc[dates,:].iloc[train_days:,:]
        test_return = pd.concat([test_return,low_returns.loc[dates,:].iloc[train_days:,:]],axis=1 )
        test_price = high_prices.loc[dates,:].iloc[train_days:,]
        test_price = pd.concat([test_price,low_prices.loc[dates,:].iloc[train_days:,:]],axis=1 )
        ticker1 = 'TXN'
        ticker2 = 'HCP'
        r1 = test_return[ticker1]
        r2 = test_return[ticker2]
        w1 = 0.5
        w2 = 1-w1
        actions = [-0.15,-0.1,-0.05,0,0.05,0.1,0.15]
        val = []
        value = init_value
        freq=22
        for i in range(5,len(test_return)):
            
            state = list(r1[(i-5):i])+list(r2[i-5:i])
            state = np.reshape(state,[1,10])
            
            w1 = w1+actions[np.argmax(agent.target_network.predict(state)[0])]
            if w1>1:w1=1
            if w1<0:w1=0
            print(w1)
            w2 = 1-w1
            
            value = (1+(w1*r1[i]+w2*r2[i]))*value
            val.append(value)
        
        nothing = np.cumprod(1+0.5*r1[5:]+0.5*r2[5:])*init_value
        test_result = pd.concat([nothing,pd.Series(val,index = nothing.index)],axis = 1)
        test_result.index = [datetime.datetime.strptime(x,"%Y-%m-%d") for x in test_result.index]
        sns.set(style="darkgrid")
        sns.lineplot(x= range(len(test_result)),y=test_result[0])    
        sns.lineplot(x= range(len(test_result)),y=test_result[1],color = 'grey')
        plt.title('Portfolio Value')
        plt.legend(['Do nothing, end with '+str(round(test_result[0][-1],2)),'DQN, end with '+str(round(test_result[1][-1],2))])
        print('******* Annualized return is : ******')
        print((val[-1]/init_value)**(1/(len(val)/252)) -1)
        print('******* Rebalance frequency is :*****')
        print(freq)
        # short sell
        # rebalance freq
        # state
        #...