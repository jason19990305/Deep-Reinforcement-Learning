import gym
import time
import numpy as np
import matplotlib.pyplot as plot

env = gym.make("CartPole-v1")

from gym.utils import play

action_size = env.action_space.n


Epochs = 90000#episodes , how many times the agent plays the game
Alpha = 0.8 #learning rate
Gamma = 0.95 #Discount rate   gamma^2 r + gamma^3 ...

epsilon = 1.0 #initial to 1,becuse first we let agent random exploeration
max_epsilon = 0.0  
min_epsilon = 0.01
decay_rate = 0.0001 

def create_bins(num_bins_per_obs=10):# create a mapping table to convert observation to discretize data.
    bins_cart_position = np.linspace(-4.8,4.8,num_bins_per_obs)
    bins_cart_velocity = np.linspace(-5,5,num_bins_per_obs)
    bins_pole_angle = np.linspace(-0.418,0.418,num_bins_per_obs)
    bins_pole_angular_velocity = np.linspace(-5,5,num_bins_per_obs)
    bins = np.array([bins_cart_position,
                    bins_cart_velocity,
                    bins_pole_angle,
                    bins_pole_angular_velocity])
    return bins
def discretize_observation(observations,bins):# convert observations to discretize data
    binned_observations = []
    for i,observation in enumerate(observations):
        discretized_observation = np.digitize(observation,bins[i])# output range index of the mapping table
        binned_observations.append(discretized_observation)
    return tuple(binned_observations)
    
    
def epsilon_greedy_action_selection(epsilon,q_table,discrete_state):
    random_number = np.random.random() #random get 0~1
    
    #Exploitation (choose the action that maximizes Q)
    if random_number > epsilon:
        state_row = q_table[discrete_state][:] #return the row of the Q value
        action = np.argmax(state_row) # return the index of the max value of the row
        
    else: #Exploitation (choose the action that random action)
        action = env.action_space.sample()        
        
    return action

def compute_next_q_value(old_q_value,reward,next_optimal_q_value):

    return old_q_value + Alpha * (reward + Gamma * next_optimal_q_value - old_q_value)
    
def reduce_epsilon(epsilon,epoch):
    
    return min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate*epoch)
    

NUM_BINS = 30
BINS = create_bins(NUM_BINS-1)
# BINS
print(BINS.shape)
observation = env.reset()
q_table_shape = (NUM_BINS,NUM_BINS,NUM_BINS,NUM_BINS,env.action_space.n)
q_table = np.zeros(q_table_shape)
for episode in range(Epochs):
    
    total_rewards = 0
    state = env.reset()[0]
    discretized_state = discretize_observation(state,BINS)
    print("Episode : ",episode)
    done = False;
    rewards = 0
    while not done:
    
        #Action
        action = epsilon_greedy_action_selection(epsilon,q_table,discretized_state)
        # state,reward ... env.step()        
        next_state, reward, done, truncated, info = env.step(action)
        if done and total_rewards < 150:
            reward = -150     
        # Old(Current) Q value note the state
        next_state_discretized = discretize_observation(next_state,BINS)
        old_q_value = q_table[discretized_state+(action,)]

        
        # Get next optimal Q value (max Q value for this state)
        next_optimal_q_value = np.max(q_table[next_state_discretized]) # Using the next state to find the optimal q value
        
        # Compute the next Q Value
        next_q = compute_next_q_value(old_q_value,reward,next_optimal_q_value)
        
        # Upate the table
        q_table[discretized_state+(action,)] = next_q
        
        # Track rewards
        total_rewards += reward
        
        # Update the state to new state
        discretized_state = next_state_discretized
        
        
        if done or truncated or total_rewards > 1000:
            break
    epsilon = reduce_epsilon(epsilon,episode)    
    print("Reward :",total_rewards)

env = gym.make("CartPole-v1",render_mode="human")
for episode in range(Epochs):
    
    total_rewards = 0
    state = env.reset()[0]
    print("Episode : ",episode)
    print("----------------------------------------")
    done = False;
    while not done:
        #Action
        discretized_state = discretize_observation(state,BINS)
        state_row = q_table[discretized_state] #return the row of the Q value
        action = np.argmax(state_row) # return the index of the max value of the row
        
        state, reward, done, truncated, info = env.step(action)  
        total_rewards += reward
        if done or truncated:
            break
    print("Total reward : ",total_rewards)
env.close()
