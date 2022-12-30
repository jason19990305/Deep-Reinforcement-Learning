import gym
import time
from matplotlib import pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential,clone_model
from tensorflow.keras.layers import Dense,Activation,Input
from tensorflow.keras.optimizers import Adam
from collections import deque
import random
import PPO2


env_name = "Acrobot-v1"

env = gym.make(env_name)


env.reset()
num_actions = env.action_space.n
num_states = env.observation_space.shape[0]
print("Action Space:",num_actions)
print("Observation Space : ",num_states)

learning_rate = 0.001


state_input = Input(shape=(num_states,))
dense1 = Dense(512,activation='relu')(state_input)
dense2 = Dense(256,activation='relu')(dense1)
dense3 = Dense(256,activation='relu')(dense2)
dense4 = Dense(128,activation='relu')(dense3)
actor_dense = Dense(64,activation='relu')(dense4)
actor_output = Dense(num_actions,activation='softmax')(actor_dense)
critic_dense = Dense(64,activation='relu')(dense4)
critic_output = Dense(1,activation='linear')(critic_dense)


class EnvForTraining(gym.Env):
    def __init__(self,env_name):
        self.env = gym.make(env_name)    # The wrapper encapsulates the gym env
        self.count = 0
    def step(self, action):
        obs, reward, done,truncted, info = self.env.step(action)   # calls the gym env methods
        #if self.count < 10000:
        #    truncted = False
        if done:
            reward = 10
        self.count += 1
        return obs, reward, done,truncted, info

    def reset(self):
        self.count = 0
        obs = self.env.reset()   # same for reset
        return obs

ppo2 = PPO2.PPO2(state_space=num_states,
                action_space=num_actions,
                iteration=10,
                warm_up=5)# the size of replay buffer

ppo2_agent = PPO2.PPO2_Agent(ppo2=ppo2,
                            state_input=state_input,
                            actor_output=actor_output,
                            critic_output=critic_output
                            )

env = EnvForTraining(env_name)
total_reward_list,avg_reward_list = ppo2_agent.fit(env,episodes=1000)


plt.plot(total_reward_list)
plt.plot(avg_reward_list)
plt.title('Model train')
plt.ylabel('Total reward')
plt.xlabel('Epoch')
plt.show()


env = gym.make(env_name,render_mode="human")
ppo2_agent.test(env,episodes=10000)

