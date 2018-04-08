import gym,sys,numpy as np
import tensorflow as tf

from gym.envs.registration import register
register(
    id='FrozenLakeNotSlippery-v0',
    entry_point='gym.envs.toy_text:FrozenLakeEnv',
    kwargs={'map_name' : '4x4', 'is_slippery': False},
    max_episode_steps=100,
    reward_threshold=0.78, # optimum = .8196
)

# make the env
env = gym.make('FrozenLake-v0')
# print(env.observation_space)
# print(env.action_space)
# q_learning_table = np.zeros([env.observation_space.n,env.action_space.n])
# print(q_learning_table)
# print(env.render())

# perfect step
env = gym.make('FrozenLakeNotSlippery-v0')
s = env.reset()
perfect_step = [1,1,2,2,1,2]
for x in perfect_step:
    env.step(x)
    env.render()




# -- end code --