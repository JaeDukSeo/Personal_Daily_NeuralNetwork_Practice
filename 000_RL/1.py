import gym,sys
import numpy as np

env = gym.make('FrozenLake-v0')



#  there are 16 and 4
print(env.observation_space)
print(env.action_space)

sys.exit()

#Initialize table with all zeros
Q = np.zeros([env.observation_space.n,env.action_space.n])

# hyper
lr = .8
y = .95
num_episodes = 1

# Reward per each step
rList = []
print(rList)
print(Q)
print('-------')

# this seems to be like some thing of num_epoch
for i in range(num_episodes):

    s = env.reset()
    rAll = 0
    d = False
    j = 0

    for j in range(19):
        
        #Choose an action by greedily (with noise) picking from Q table
        a = np.argmax(Q[s,:] + np.random.randn(1,env.action_space.n)*(1./(i+1)))

        print(a)

        #Get new state and reward from environment (observation,reward,done)
        s1,r,d,_ = env.step(a)

        #Update Q-Table with new knowledge
        Q[s,a] = Q[s,a] + lr * (r + y * np.max(Q[s1,:]) - Q[s,a] )
        rAll += r
        # print(s,'  ', d, ' ', s1)
        s = s1
        if d == True:
            break
    rList.append(rAll)

print('========================')
print(len(rList))
print(Q)
print('-------')


# -- end code --