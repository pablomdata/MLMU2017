#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 18:41:48 2017

@author: pablo
"""

import gym
import numpy as np

def run_episode(env, weights, ep):
    s = env.reset()
    total_reward = 0
    for _ in range(200):
        a = 0 if np.dot(weights,s) < 0 else 1
        env.render()
        s, r, done, info = env.step(a)
        total_reward += r
        if done:
            print("episode: {}, score: {}".format(ep+1, total_reward))
            break
    return total_reward

def random_train(n_episodes):
    '''Random search'''
    env = gym.make('CartPole-v0')
    best_weights = None
    best_ep_reward = 0
    counter = 0
    # Random search
    for _ in range(n_episodes):
        counter += 1
        weights = 2*np.random.rand(4)-1
        ep_reward = run_episode(env, weights, _)
        
        if ep_reward > best_ep_reward: 
            best_weights = weights
            best_ep_reward = ep_reward
    
            if ep_reward == 200:
                break
                
    env.close()        
    return best_weights, best_ep_reward, counter
       

if __name__ == "__main__":
    random_train(200)



