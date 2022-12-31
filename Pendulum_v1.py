import gym
import random
import numpy as np
from collections import deque
import tensorflow
from tensorflow.keras.models import Sequential ,clone_model 
from tensorflow.keras.layers import Dense ,Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow import keras
import A2C_v1

env = gym.make("Pendulum-v1",render_mode='human')


env.reset()
print("Action Space:",env.action_space.shape[0])
print("Observation Space : ",env.observation_space.shape[0])
num_states = env.action_space.shape[0]
num_actions = env.observation_space.shape[0]
learning_rate = 0.001

# Create Actor
actor = Sequential()
actor.add(Input(shape=(num_states,)))
actor.add(Dense(16,activation='relu'))
actor.add(Dense(32,activation='relu'))
actor.add(Dense(32,activation='relu'))
actor.add(Dense(32,activation='sigmoid'))
actor.add(Dense(num_actions,activation='linear'))

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
        if self.count < 5000:
            truncted = False
        if done:
            reward = -10
        self.count += 1
        return obs, reward, done,truncted, info

    def reset(self):
        self.count = 0
        obs = self.env.reset()   # same for reset
        return obs

a2c = A2C_v1.A2C(state_space=num_states,
                action_space=num_actions,
                gamma=0.98)# the size of replay buffer

a2c_agent = A2C_v1.A2C_Agent(a2c=a2c,
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

