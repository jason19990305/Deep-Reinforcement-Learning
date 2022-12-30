import gym
import time
from matplotlib import pyplot as plt
import numpy as np
from tensorflow.keras.models import Sequential,clone_model
from tensorflow.keras.layers import Dense,Activation,Input,Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import Orthogonal,Zeros
from collections import deque
import random
import Continuous_PPO2


env_name = "BipedalWalker-v3"
#env_name = "Pendulum-v1"
env = gym.make(env_name)


env.reset()
num_actions = env.action_space.shape[0]
num_states = env.observation_space.shape[0]
print("Action Space:",num_actions)
print("Observation Space : ",num_states)
#print(env.action_space)
learning_rate = 0.001

#Actor
actor_state_input = Input(shape=(num_states,))
actor_dense1 = Dense(512,activation='tanh'    ,kernel_initializer=Orthogonal(gain=np.sqrt(2)) ,bias_initializer=Zeros())(actor_state_input)
actor_dense2 = Dense(256,activation='tanh'    ,kernel_initializer=Orthogonal(gain=np.sqrt(2)) ,bias_initializer=Zeros())(actor_dense1)
actor_dense3 = Dense(64,activation='tanh' ,kernel_initializer=Orthogonal(gain=np.sqrt(2)) ,bias_initializer=Zeros())(actor_dense2)
actor_output = Dense(num_actions,activation='tanh',kernel_initializer=Orthogonal(gain=0.01) ,bias_initializer=Zeros())(actor_dense3)
sigma_output =  Dense(num_actions,activation='softplus',kernel_initializer=Orthogonal(gain=0.01) ,bias_initializer=Zeros())(actor_dense3)
#Critic

'''critic_state_input = Input(shape=(num_states,))
critic_dense1 = Dense(512,activation='tanh'    ,kernel_initializer=Orthogonal(gain=np.sqrt(2)) ,bias_initializer=Zeros())(critic_state_input)
critic_dense2 = Dense(256,activation='tanh'    ,kernel_initializer=Orthogonal(gain=np.sqrt(2)) ,bias_initializer=Zeros())(critic_dense1)
critic_dense3 = Dense(64,activation='tanh' ,kernel_initializer=Orthogonal(gain=np.sqrt(2)) ,bias_initializer=Zeros())(actor_dense2)'''
critic_dense = Dense(64,activation='tanh' ,kernel_initializer=Orthogonal(gain=np.sqrt(2)) ,bias_initializer=Zeros())(actor_dense3)
critic_output = Dense(1,activation='linear' ,kernel_initializer=Orthogonal(gain=np.sqrt(2)) ,bias_initializer=Zeros())(critic_dense)



class EnvForTraining(gym.Env):
    def __init__(self,env_name):
        self.env = gym.make(env_name)    # The wrapper encapsulates the gym env
        self.count = 0
    def step(self, action):
        obs, reward, done,truncted, info = self.env.step(action)   # calls the gym env methods
        self.count += 1
        return obs, reward, done,truncted, info

    def reset(self):
        self.count = 0
        obs = self.env.reset()   # same for reset
        return obs

ppo2 = Continuous_PPO2.PPO2(state_space=num_states,
                action_space=num_actions,
                iteration=10,
                warm_up=5,
                batch_size = 256,
                stop_score=250)# the size of replay buffer

ppo2_agent = Continuous_PPO2.PPO2_Agent(ppo2=ppo2,
                            actor_state_input=actor_state_input,
                            critic_state_input=actor_state_input,
                            actor_output=actor_output,
                            critic_output=critic_output,
                            sigma_output=sigma_output
                            )
env = EnvForTraining(env_name)
total_reward_list,avg_reward_list = ppo2_agent.fit(env,episodes=50000)

plt.plot(total_reward_list)
plt.plot(avg_reward_list)
plt.title('Model train')
plt.ylabel('Total reward')
plt.xlabel('Epoch')
plt.show()


ppo2_agent.save_weight("BipedalWalker-v3_12_29.h5")
env = gym.make(env_name,render_mode="human")
ppo2_agent.test(env,episodes=10000)

