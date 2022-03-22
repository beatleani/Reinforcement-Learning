import sys
import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ddpg import DDPGagent
from utils import *

env = gym.make("Pendulum-v1")
#env = gym.make("Ant-v2")

agent = DDPGagent(env)
noise = 0.1#OUNoise(env.action_space)
batch_size = 64
rewards = []
avg_rewards = []

for episode in range(1000):
    state = env.reset()
    #noise.reset()
    episode_reward = 0
    
    for step in range(1000):
        action = [agent.get_action(state)]
       # print(action)
        #action = noise.get_action(action, step)
        new_state, reward, done, _ = env.step(action) 
        agent.memory.push(state, action, reward, new_state, done)
       # print(len(agent.memory))
        if len(agent.memory) > batch_size:
            agent.update(batch_size)        
        
        state = new_state
        episode_reward += reward

        if done:
            sys.stdout.write("episode: {}, reward: {}, average _reward: {} \n".format(episode, np.round(episode_reward, decimals=2), np.mean(rewards[-10:])))
            break

    rewards.append(episode_reward)
    avg_rewards.append(np.mean(rewards[-10:]))

plt.plot(rewards)
plt.plot(avg_rewards)
plt.plot()
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.show()