# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 23:38:35 2019

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
import random
class Agent:
    def __init__(self, env, optimizer):
        
        # Initialize atributes
        self.env=env
        self._state_size = self.env.state_size
        self._action_size = self.env.action_size
        self._optimizer = optimizer
        
        self.expirience_replay = deque(maxlen=2000)
        
        # Initialize discount and exploration rate
        self.gamma = 0.9
        self.epsilon = 0.1
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        # Build networks
        self.q_network = self._build_compile_model()
        self.target_network = self._build_compile_model()
        self.alighn_target_model()

    def store(self, state, action, reward, next_state, terminated):
        self.expirience_replay.append((state, action, reward, next_state, terminated))
    
    def _build_compile_model(self):
        model = Sequential()
        model.add(Dense(20, input_dim = self._state_size, activation='relu'))
        model.add(Dense(15, activation='relu'))
        model.add(Dense(self._action_size, activation='linear'))
        
        model.compile(loss='mse', optimizer=self._optimizer)
        return model

    def alighn_target_model(self):
        self.target_network.set_weights(self.q_network.get_weights())
    
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return self.env.random_act()
        
        q_values = self.q_network.predict(state) # choose best action according to the Q 
        return np.argmax(q_values[0]) # returning the corresponding action

    def retrain(self, batch_size):
        
        
        minibatch = random.sample(self.expirience_replay, batch_size)
        
        
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
              target = reward + self.gamma * \
                       np.amax(self.q_network.predict(next_state)[0])
            target_f = self.q_network.predict(state)
            target_f[0][action] = target
            self.q_network.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay