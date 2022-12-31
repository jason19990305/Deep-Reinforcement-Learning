import gym
import time
from matplotlib import pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential,clone_model
from tensorflow.keras.layers import Dense,Activation
from tensorflow.keras.optimizers import Adam
from collections import deque
import random
import DQN


env_name = "CartPole-v1"

env = gym.make(env_name)


env.reset()
num_actions = env.action_space.n
num_states = env.observation_space.shape[0]
print("Action Space:",num_actions)
print("Observation Space : ",num_states)

learning_rate = 0.001

# Creat ANN
model = Sequential()
model.add(Dense(16,input_shape=(1,num_states)))
model.add(Activation('relu'))

model.add(Dense(32))
model.add(Activation('relu')) 

# output == action_space
model.add(Dense(num_actions))
model.add(Activation('linear'))
model.compile(loss='mse',optimizer=(Adam(lr=learning_rate)))

print(model.summary())

class EnvForTraining(gym.Env):
    def __init__(self,env_name):
        self.env = gym.make(env_name)    # The wrapper encapsulates the gym env
        self.count = 0
    def step(self, action):
        obs, reward, done,truncted, info = self.env.step(action)   # calls the gym env methods
        if done:
            reward = -10
        if self.count > 500:
            truncted = True
        self.count += 1
        return obs, reward, done,truncted, info

    def reset(self):
        self.count = 0
        obs = self.env.reset()   # same for reset
        return obs
env = EnvForTraining(env_name)

dqn_para = DQN.QParameter(num_states,
                        num_actions, 
                        epsilon = 1.0,
                        e_decay=0.99 ,
                        gamma = 0.99,
                        batch_size = 32, 
                        buffer_size = 2000,
                        update_target_model=1000)

dqn_agent = DQN.DQNAgent(model=model,env=env,dqn_para=dqn_para,learning_rate = 0.001)


total_reward_list,loss_list = dqn_agent.fit(env,episodes=800)

plt.plot(total_reward_list)
plt.title('model score')
plt.ylabel('Total reward')
plt.xlabel('Epoch')
plt.show()


env = gym.make(env_name,render_mode="human")
dqn_agent.test(env,episodes=10000)

