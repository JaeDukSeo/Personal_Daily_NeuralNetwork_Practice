import gym
import numpy as np

env = gym.make('FrozenLake-v0')

print(env)
print(dir(env))
print(type(env.render()))

for x in dir(env):
    try:
        temp = getattr(env,x)
        print(x,temp)
    except:
        pass

# -- end code --