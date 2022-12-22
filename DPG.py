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
#tensorflow.config.experimental_run_functions_eagerly(True)
tf.compat.v1.disable_eager_execution()
class DPG():
    def __init__(self,advantage,state_space, action_space,training_episode = 1):
        self.state_space = state_space
        self.action_space = action_space
        self.training_episode = training_episode  
        self.advantage = advantage    
    
class Discount():
    def __init__(self,gamma=0.99):
        self.gamma = gamma
              
    def estimate_function(self,rewards):
        G = np.zeros_like(rewards)
        for t in range(len(rewards)):
            G_sum = 0
            discount = 1
            for k in range(t,len(rewards)):
                G_sum += rewards[k]*discount
                discount *= self.gamma
            G[t] = G_sum
        mean = np.mean(G)
        std = np.std(G) if np.std(G)>0 else 1
        G = (G-mean)/std
        
        
        
        return np.array(G)

                    
class TotalReward():
    def __init__(self):
        return          
    def estimate_function(self,rewards):
        G_sum = np.sum(rewards)
        G = np.ones_like(rewards)
        G = G*G_sum
        return G

              
class DPGAgent():

    def __init__(self,state_input,action_output,env,dpg,optimizer = Adam(lr=0.001)):
        self.dpg = dpg
        self.env = env
        self.optimizer = optimizer
        advantages = Input(shape=[1])
        
        self.policy = Model([state_input,advantages],[action_output])
        #**********
        
        def custom_loss(y_true,y_pred):      
            y_pred = K.clip(y_true*y_pred,1e-10,1-1e-10)            
            return -K.sum(K.log(y_pred)*advantages)
            
            
        def policy_gradient_loss(y_true, y_pred):
            policy_loss = -keras.losses.logcosh(y_true, y_pred)
            return policy_loss
            
        #self.policy.compile(optimizer=optimizer, loss=custom_loss)
        self.model = Model([state_input],[action_output])
        self.policy.compile(optimizer=optimizer, loss=custom_loss)
        tf.keras.utils.plot_model(self.policy, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
        #***************
        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []
        self.trajectory_memory = []
        self.loss_list = []
        
    def store_transition(self,state,action,reward):
        self.state_memory.append(state)
        self.action_memory.append(action)
        self.reward_memory.append(reward)
        
    def store_trajectory(self):
        self.trajectory_memory.append((self.state_memory, self.action_memory, self.reward_memory))
        self.clear_memory()
        
    def clear_memory(self):
        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []
        
    
        
    
        
    def softmax(self,x):   
        return np.exp(x) / np.sum(np.exp(x), axis=0)
    
    def replay(self):      
    
        total_loss = []
        for trajectory in self.trajectory_memory: 
        
            (states , actions ,rewards) = trajectory
            
            
            labels = np.zeros([len(actions),self.dpg.action_space])
            labels[np.arange(len(actions)) , actions] = 1  # Setting the max acion probilbality    to 1,other action is 0                         
            
            #print(labels*actions)
            advantage = self.dpg.advantage.estimate_function(rewards)
            
            cost = self.policy.train_on_batch([np.array(states),advantage],labels)
            #print(cost)
          
            
        self.trajectory_memory = []
        self.clear_memory()
        #print( np.sum(total_loss)/len(total_loss))
        return cost
        #return np.sum(total_loss)/len(total_loss)
        
    def selecte_action(self,state):
        state = state[np.newaxis,:]
        probabilities = self.model.predict(state)[0]
        #print(probabilities)
        try:
            action = np.random.choice(self.dpg.action_space,p=probabilities)
        except:
            print(probabilities)
            return None
        return action
        
    def save_weight(self,file_name ='dqn_weights.h5'):
        self.model.save_weights('./'+file_name)
        
    def fit(self,env,episodes=1000):
    
        total_reward_list = []
        loss_list = []
        
        for episode in range(episodes):
            state = env.reset()[0]
            state = np.array(state)
            #state = state.reshape([1,self.dpg.state_space])
            done = False            
            total_reward = 0
            
            
            while not done:
            
                action = self.selecte_action(state)
                if action == None:
                    break
                next_state, reward, done, truncated, info = env.step(action) 
                #next_state = next_state.reshape([1,self.dpg.state_space])   
                next_state = np.array(next_state)
                self.store_transition(state ,action ,reward)
                
                
                state = next_state
                
                total_reward += reward                                
                
                if done or truncated:                    
                    break;
            self.store_trajectory()
            if episode % self.dpg.training_episode == 0:
                total_loss = self.replay()# Training
                #print(total_loss)
                loss_list.append(total_loss)
            print("episode : %d/%d , score: %d " %(episode,episodes,total_reward))
            total_reward_list.append(total_reward)
            
        env.close()
        return total_reward_list,loss_list
    def load_weights(self,file_name ='dpg_weights.h5'):
        self.eval_model.load_weights('./'+file_name)
        self.target_model.load_weights('./'+file_name)
    
    def test(self,env,episodes):
        total_reward_list = []
        for episode in range(episodes):
            state = env.reset()[0]
            
            #state = state.reshape([1,self.dpg.state_space])
            done = False            
            total_reward = 0
            print("Episode :",episode)
            
            while not done:
                #print(state)
                action = self.selecte_action(state)
                
                next_state, reward, done, truncated, info = env.step(action) 
                #next_state = next_state.reshape([1,self.dpg.state_space])  
                state = next_state               
                total_reward += reward
                                
                
                if done or truncated:                    
                    break;
            print("Total reward : ",total_reward)
            total_reward_list.append(total_reward)
        env.close()
        return total_reward_list
        
        