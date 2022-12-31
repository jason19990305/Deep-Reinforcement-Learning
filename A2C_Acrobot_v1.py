import gym
import time
from matplotlib import pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential,clone_model
from tensorflow.keras.layers import Dense,Activation,Input
from tensorflow.keras.optimizers import Adam
from collections import deque
import random
import A2C_v0


env_name = "Acrobot-v1"

env = gym.make(env_name)


env.reset()
num_actions = env.action_space.n
num_states = env.observation_space.shape[0]
print("Action Space:",num_actions)
print("Observation Space : ",num_states)

learning_rate = 0.001

# Create Actor
actor = Sequential()
actor.add(Input(shape=(num_states,)))
actor.add(Dense(16,activation='relu'))
actor.add(Dense(32,activation='relu'))
actor.add(Dense(32,activation='relu'))
actor.add(Dense(32,activation='sigmoid'))
actor.add(Dense(num_actions,activation='softmax'))

# Create Critic
critic = Sequential()
critic.add(Input(shape=(num_states,)))
critic.add(Dense(16,activation='relu'))
critic.add(Dense(32,activation='relu'))
critic.add(Dense(32,activation='relu'))
critic.add(Dense(1,activation='linear'))


class EnvForTraining(gym.Env):
    def __init__(self,env_name):
        self.env = gym.make(env_name)    # The wrapper encapsulates the gym env
        self.count = 0
    def step(self, action):
        obs, reward, done,truncted, info = self.env.step(action)   # calls the gym env methods
        if self.count < 1000:
            truncted = False
        if done:
            reward = +5
        self.count += 1
        return obs, reward, done,truncted, info

    def reset(self):
        self.count = 0
        obs = self.env.reset()   # same for reset
        return obs



a2c = A2C_v0.A2C(state_space=num_states,
                action_space=num_actions,
                gamma=0.98)# the size of replay buffer

a2c_agent = A2C_v0.A2C_Agent(a2c=a2c,
                            actor=actor,
                            critic=critic)

env = EnvForTraining(env_name)
total_reward_list,loss_list = a2c_agent.fit(env,episodes=10000)


plt.plot(total_reward_list)
plt.title('Model train')
plt.ylabel('Total reward')
plt.xlabel('Epoch')
plt.show()


env = gym.make(env_name,render_mode="human")
a2c_agent.test(env,episodes=10000)
