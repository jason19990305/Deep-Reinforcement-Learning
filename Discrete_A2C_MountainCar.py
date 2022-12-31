import gym
import time
from matplotlib import pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential,clone_model
from tensorflow.keras.layers import Dense,Activation,Input
from tensorflow.keras.optimizers import Adam
from collections import deque
import random
import Discrete_A2C


env_name = "MountainCar-v0"

env = gym.make(env_name)


env.reset()
num_actions = env.action_space.n
num_states = env.observation_space.shape[0]
print("Action Space:",num_actions)
print("Observation Space : ",num_states)

learning_rate = 0.001


state_input = Input(shape=(num_states,))
dense1 = Dense(16,activation='relu')(state_input)
dense2 = Dense(32,activation='relu')(dense1)
dense3 = Dense(32,activation='relu')(dense2)
dense4 = Dense(32,activation='relu')(dense3)
actor_dense = Dense(32,activation='relu')(dense4)
actor_output = Dense(num_actions,activation='softmax')(actor_dense)
critic_dense = Dense(32,activation='relu')(dense4)
critic_output = Dense(1,activation='linear')(critic_dense)


class EnvForTraining(gym.Env):
    def __init__(self,env_name):
        self.env = gym.make(env_name)    # The wrapper encapsulates the gym env
        self.count = 0
    def step(self, action):
        obs, reward, done,truncted, info = self.env.step(action)   # calls the gym env methods
        if self.count <= 800:
            truncted = False
        if done:
            reward = 10
        self.count += 1
        return obs, reward, done,truncted, info

    def reset(self):
        self.count = 0
        obs = self.env.reset()   # same for reset
        return obs

a2c = Discrete_A2C.A2C(state_space=num_states,
                action_space=num_actions,
                iteration=30,
                warm_up=100)# the size of replay buffer

a2c_agent = Discrete_A2C.A2C_Agent(a2c=a2c,
                            state_input=state_input,
                            actor_output=actor_output,
                            critic_output=critic_output
                            )

env = EnvForTraining(env_name)
total_reward_list,loss_list = a2c_agent.fit(env,episodes=500)


plt.plot(total_reward_list)
plt.title('Model train')
plt.ylabel('Total reward')
plt.xlabel('Epoch')
plt.show()


env = gym.make(env_name,render_mode="human")
a2c_agent.test(env,episodes=10000)

