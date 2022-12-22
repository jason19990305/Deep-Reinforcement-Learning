import gym
import time
import numpy as np
import matplotlib.pyplot as plot

env = gym.make("FrozenLake-v1",is_slippery=True)

from gym.utils import play

#This is a play module that allows you to interact with some of the games.
#**It will need install pygame**
#Not every environment in OpenAi gym is interactive or allows you to play as a human.

#Hyperparameter initial
#rows-->States  colums-->Action
action_size = env.action_space.n
state_size = env.observation_space.n
q_table = np.zeros([state_size,action_size])
print("Q Table size:",q_table.shape)


Epochs = 40000#episodes , how many times the agent plays the game
Alpha = 0.8 #learning rate
Gamma = 0.95 #Discount rate   gamma^2 r + gamma^3 ...

epsilon = 1.0 #initial to 1,becuse first we let agent random exploeration
max_epsilon = 1.0  
min_epsilon = 0.01
decay_rate = 0.001 

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

rewards = []
for episode in range(Epochs):
    
    total_rewards = 0
    state = env.reset()[0]
    print("Episode : ",episode)
    done = False;
    while not done:
    
        #Action
        action = epsilon_greedy_action_selection(epsilon,q_table,state)
        # state,reward ... env.step()        
        next_state, reward, done, truncated, info = env.step(action)
        if reward == 1:
            print("You got the treasure!!")       
        
        
        # Old(Current) Q value note the state
        old_q_value = q_table[state,action]
        
        # Get next optimal Q value (max Q value for this state)
        next_optimal_q_value = np.max(q_table[next_state,:]) # Using the next state to find the optimal q value
        
        # Compute the next Q Value
        next_q = compute_next_q_value(old_q_value,reward,next_optimal_q_value)
        
        # Upate the table
        q_table[state,action] = next_q
        
        # Track rewards
        total_rewards += reward
        rewards.append(reward)
        
        # Update the state to new state
        state = next_state
        
        
        if done or truncated:
            break
    epsilon = reduce_epsilon(epsilon,episode)    
    rewards.append(total_rewards)
            

env = gym.make("FrozenLake-v1",render_mode="human",is_slippery=True)
for episode in range(Epochs):
    
    total_rewards = 0
    state = env.reset()[0]
    print("Episode : ",episode)
    print("----------------------------------------")
    done = False;
    while not done:
        #Action
        state_row = q_table[state][:] #return the row of the Q value
        
        action = np.argmax(state_row) # return the index of the max value of the row
        if action == 0:
            print("Left")
        if action == 1:
            print("Down")
        if action == 2:
            print("Right")
        if action == 3:
            print("Up")       
        state, reward, done, truncated, info = env.step(action)
        if reward == 1:
            print("You got the treasure!!")   
        
        if done or truncated:
            break
env.close()