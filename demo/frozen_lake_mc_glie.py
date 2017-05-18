API_KEY = 'sk_nrENBskESHfG1zUasGj3w'


import gym 
import numpy as np
import random
import time           

problem = 'FrozenLake-v0'
algo_name = 'mc_glie'
env = gym.make(problem)
env = gym.wrappers.Monitor(env, algo_name, force=True)

Q = np.random.rand(env.observation_space.n, env.action_space.n)
R = np.zeros([env.observation_space.n, env.action_space.n])
N = np.zeros([env.observation_space.n, env.action_space.n])
policy = {}

#Initialize policy randomly
for s in range(env.observation_space.n): 
    policy[s] = 1 if random.random()<0.5 else 2

n_episodes = 10000
gamma = 1
j = 0


start = time.time()
## Train
while j<n_episodes:
    j += 1
    s = env.reset()
    d = False
    ep_reward = 0
    it = 0
    episode = []
    
    if j % 1000 == 00 and j>0:
        print("Now playing episode: ", j)
    
    # Generate samples from the episode
    while not d:
        it +=1
        a = policy[s]
        s1, r, d, _ = env.step(a)
        episode.append((s,a))
        s = s1

    # Update value functions after the sampling from the episode
    for s,a in episode:
        N[s,a] +=1
        G = gamma**it*r # delayed reward
        Q[s,a] += 1/N[s,a]*(G-Q[s,a])
    
    # Epsilon-greedy update for the policy
    random.seed(42)
    ep = random.random()
    for s,_ in episode:
        policy[s] = np.argmax(Q[s,:]) if ep>1000/(j+1) else random.randint(0,3)
        
env.close()    
gym.upload(algo_name, 
               api_key=API_KEY)


stop = time.time()
print("Training completed... Showtime")
print("It took: {0} episodes and {1} minutes".format(j,(stop-start)/60))

    
