import gym
import time
from matplotlib import pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential,clone_model
from tensorflow.keras.layers import Dense,Activation,Flatten
from tensorflow.keras.optimizers import Adam
from collections import deque
import random

from rl.agents.dqn import DQNAgent
from rl.memory import SequentialMemory
from rl.policy import LinearAnnealedPolicy,EpsGreedyQPolicy


policy = LinearAnnealedPolicy(EpsGreedyQPolicy(),
                              attr='eps',#epsilon
                              value_max=1.0,
                              value_min=0.1,
                              value_test=0.05,
                              nb_steps = 20000)
memory = SequentialMemory(limit=20000,window_length = 1)

env_name = 'CartPole-v1'
env = gym.make(env_name)
num_actions = env.action_space.n
num_observations = env.observation_space.shape[0]
print("Action Space:",num_actions)
print("Observation Space : ",num_observations)

class EnvForTraining(gym.Env):
    def __init__(self):
        self.env = gym.make(env_name)    # The wrapper encapsulates the gym env
        
    def step(self, action):
        obs, reward, done,truncted, info = self.env.step(action)   # calls the gym env methods
        return obs, reward, done, info

    def reset(self):
        obs = self.env.reset()[0]    # same for reset
        return obs

# Environment 

env = EnvForTraining()



# Hyperparameter

Epochs = 50
epsilon = 1.0
epsilon_reduce = 0.98
learning_rate = 0.001
gamma = 0.9
batch_size = 32

# Creat ANN
model = Sequential()
model.add(Flatten(input_shape=(1,num_observations)))
model.add(Dense(16))
model.add(Activation('relu'))

model.add(Dense(32))
model.add(Activation('relu')) 

model.add(Dense(64))
model.add(Activation('relu')) 

model.add(Dense(32))
model.add(Activation('relu')) 
# output == action_space
model.add(Dense(num_actions))
model.add(Activation('linear'))
model.compile(loss='mse',optimizer=(Adam(lr=learning_rate)))
print(model.summary())

dqn = DQNAgent(model=model,
              nb_actions=num_actions,
              memory=memory,
              nb_steps_warmup=10,
              target_model_update=100,
              policy=policy)
dqn.compile(Adam(learning_rate=learning_rate),metrics=['mse'])


dqn.fit(env,nb_steps=50000,visualize=False,verbose=2)



class EnvForHuman(gym.Env):
    def __init__(self):
        self.env = gym.make(env_name,render_mode='human')    # The wrapper encapsulates the gym env
        
    def step(self, action):
        obs, reward, done,truncted, info = self.env.step(action)   # calls the gym env methods
        return obs, reward, done, info
    def render(self,mode):
        return
    def reset(self):
        obs = self.env.reset()[0]    # same for reset
        return obs

# Environment 

env = EnvForHuman()


dqn.test(env,nb_episodes=100,visualize=True)

env.close()