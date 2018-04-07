import gym,sys,numpy as np
# import tensorflow as tf


env = gym.make('FrozenLake-v0')

print(env.observation_space)
print(env.action_space)

q_learning_table = np.zeros([env.observation_space.n,env.action_space.n])
print(q_learning_table)

print(env.render())





# -- end code --