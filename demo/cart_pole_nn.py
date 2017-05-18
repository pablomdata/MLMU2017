#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 08:38:47 2017

@author: pablo
"""

import gym
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop
import random

class NNAgent(object):
    def __init__(self, state_size=4, action_size=2):
        self.state_size = state_size
        self.action_size = action_size
        self.model = self._build_model()
        self.memory = []
        self.learning_rate = 0.01
        self.gamma = 0.9    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.e_decay = .99
        self.e_min = 0.05

    def remember(self, state, action, reward, next_state, done):
            self.memory.append((state, action, reward, next_state, done))         
        
    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(20, input_dim=self.state_size, activation='tanh'))
        #model.add(Dense(20, activation='tanh', init='uniform'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=RMSprop(lr=0.01))
        return model    
    
    def move(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action
    
    
    def train_nn(self):
        batch_size = len(self.memory)
        minibatch = self.memory
        X = np.zeros((batch_size, self.state_size))
        Y = np.zeros((batch_size, self.action_size))
        for i in range(batch_size):
            state, action, reward, next_state, done = minibatch[i]
            target = self.model.predict(state)[0]
            if done:
                target[action] = reward
            else:
                target[action] = reward + self.gamma * \
                            np.amax(self.model.predict(next_state)[0])
            X[i], Y[i] = state, target
        self.model.fit(X, Y, batch_size=batch_size, epochs=1, verbose=0)
        if self.epsilon > self.e_min:
            self.epsilon *= self.e_decay        
    
    def train(self,n_episodes):
        env = gym.make('CartPole-v0')
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n    
        counter = 0
        for e in range(n_episodes):
            if e % 100 == 0:
                print("INFO: Episode ", e)
            counter +=1
            state = env.reset()
            state = np.reshape(state, [1, self.state_size])
            for time in range(1000):
                env.render()
                action = self.move(state)
                next_state, reward, done, _ = env.step(action)
                reward = reward if not done else -10
                next_state = np.reshape(next_state, [1, self.state_size])
                self.remember(state, action, reward, next_state, done)
                state = next_state
                if done or time == 999:
                    print("episode: {}/{}, score: {}, e: {:.2}"
                            .format(e+1, n_episodes, time, self.epsilon))
                    break
            self.train_nn()        
        return counter
    
if __name__ == "__main__":
    
#    # Which one is faster?
#    import matplotlib.pyplot as plt
    returns_nn = [NNAgent().train(n_episodes=200) for _ in range(200)]    