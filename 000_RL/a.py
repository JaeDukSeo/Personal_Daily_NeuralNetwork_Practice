import gym,sys,numpy as np
# import tensorflow as tf


env = gym.make('CartPole-v0')

print(env.observation_space)
print(env.action_space)


sys.exit()
env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample()) 