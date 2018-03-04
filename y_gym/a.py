# import gym
# env = gym.make('Copy-v0')
# env.reset()
# env.render()

# import gym
# env = gym.make('SpaceInvaders-v0')
# env.reset()
# env.render()

# import gym
# env = gym.make('LunarLander-v2')
# env.reset()
# env.render()

# import gym
# env = gym.make('CartPole-v0')
# env.reset()
# env.render()

# import gym
# env = gym.make('Humanoid-v2')
# env.reset()
# env.render()

# import gym
# env = gym.make('HandManipulateBlock-v0')
# env.reset()
# env.render()

import gym
# env = gym.make('FrozenLake-v0')
env = gym.make('Humanoid-v2')

env.reset()
env.render()

for _ in range(1000000): # run for 1000 steps
    action = env.action_space.sample() # pick a random action
    env.step(action) # take action
    env.render()