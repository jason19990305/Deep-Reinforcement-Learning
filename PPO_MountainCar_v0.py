import gym
import time
from matplotlib import pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential,clone_model
from tensorflow.keras.layers import Dense,Activation,Input
from tensorflow.keras.optimizers import Adam
from collections import deque
import random
import PPO


env_name = "MountainCar-v0"

env = gym.make(env_name)


env.reset()
num_actions = env.action_space.n
num_states = env.observation_space.shape[0]
print("Action Space:",num_actions)
print("Observation Space : ",num_states)

learning_rate = 0.001

# Creat ANN
state_input = Input(shape=(num_states,))
dense1 = Dense(16,activation='tanh')(state_input)
dense2 = Dense(32,activation='tanh')(dense1)
dense3 = Dense(32,activation='tanh')(dense2)
dense4 = Dense(32,activation='tanh')(dense3)
action_output = Dense(num_actions,activation='softmax')(dense4)
#model.compile(loss='mse',optimizer=(Adam(lr=learning_rate)))


class EnvForTraining(gym.Env):
    def __init__(self,env_name):
        self.env = gym.make(env_name)    # The wrapper encapsulates the gym env
        self.count = 0
    def step(self, action):
        obs, reward, done,truncted, info = self.env.step(action)   # calls the gym env methods
        if truncted:
            reward = -3
        if done:
            reward = 3
        self.count += 1
        return obs, reward, done,truncted, info

    def reset(self):
        self.count = 0
        obs = self.env.reset()   # same for reset
        return obs


ppo = PPO.PPO(state_space=num_states,
              action_space=num_actions,
              gamma=0.98,
              iteration=10,
              buffer_size=2000)

ppo_agent = PPO.PPOAgent(state_input=state_input,action_output=action_output
,env=env,ppo=ppo,optimizer=Adam(lr=learning_rate),update_model = 1)

env = EnvForTraining(env_name)
total_reward_list,loss_list = ppo_agent.fit(env,episodes=1000)


plt.plot(total_reward_list)
#plt.plot(loss_list)
plt.title('Model train')
plt.ylabel('Total reward')
plt.xlabel('Epoch')
plt.show()


env = gym.make(env_name,render_mode="human")
ppo_agent.test(env,episodes=10000)

