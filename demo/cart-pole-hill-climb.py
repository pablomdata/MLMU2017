# -*- coding: utf-8 -*-
"""
Created on Tue May 16 18:42:29 2017

@author: pc
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

def hill_train(n_episodes):
    '''Hill-climbing algorithm for training''' 
    env = gym.make('CartPole-v0')
    alpha = 0.01
    best_weights = 2*np.random.rand(4)-1
    best_ep_reward = 0
    counter = 0 
    
    for _ in range(n_episodes):
        counter+=1
        weights_update = best_weights+alpha*(2*np.random.rand(4)-1)
        ep_reward = run_episode(env,weights_update, _)
        if ep_reward > best_ep_reward:
            best_ep_reward = ep_reward
            best_weights = weights_update
        if ep_reward == 200:
            break
    env.close()        
    return best_weights, best_ep_reward, counter


if __name__ == "__main__":
    hill_train(200)
