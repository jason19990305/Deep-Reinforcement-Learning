import gym
import time
from matplotlib import pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential,clone_model
from tensorflow.keras.layers import Dense,Activation,Input
from tensorflow.keras.optimizers import Adam
from collections import deque
import random
import DPG


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
dense1 = Dense(16,activation='relu')(state_input)
dense2 = Dense(32,activation='relu')(dense1)
dense3 = Dense(32,activation='relu')(dense2)
action_output = Dense(num_actions,activation='softmax')(dense3)
#model.compile(loss='mse',optimizer=(Adam(lr=learning_rate)))


class EnvForTraining(gym.Env):
    def __init__(self,env_name):
        self.env = gym.make(env_name)    # The wrapper encapsulates the gym env
        self.count = 0
    def step(self, action):
        obs, reward, done,truncted, info = self.env.step(action)   # calls the gym env methods
        if self.count < 1000:
            truncted = False
        if done:
            reward = -3
        self.count += 1
        return obs, reward, done,truncted, info

    def reset(self):
        self.count = 0
        obs = self.env.reset()   # same for reset
        return obs
env = EnvForTraining(env_name)

dpg = DPG.DPG(advantage=DPG.Discount(),
              state_space=num_states,
              action_space=num_actions,     
              training_episode=1)

dqn_agent = DPG.DPGAgent(state_input=state_input,action_output=action_output
,env=env,dpg=dpg,optimizer=Adam(lr=learning_rate))


total_reward_list,loss_list = dqn_agent.fit(env,episodes=2000)

plt.plot(total_reward_list)
#plt.plot(loss_list)
plt.title('Model train')
plt.ylabel('Total reward')
plt.xlabel('Epoch')
plt.show()


env = gym.make(env_name,render_mode="human")
dqn_agent.test(env,episodes=10000)

