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

tf.compat.v1.disable_eager_execution()
class Actor():
    def __init__(self,state_space,action_space,model,optimizer=Adam(lr=0.001)):
        self.model = model
        self.action_space = action_space
        self.state_space = state_space        
        def custom_loss(y_true,y_pred):
            advantages, actions = y_true[:, :1], y_true[:, 1:]            
            y_pred = K.clip(y_pred*actions,1e-10,1-1e-10)            
            loss = -K.sum(K.log(y_pred)*advantages)            
            return loss                        
        self.model.compile(optimizer=optimizer, loss=custom_loss)
        
    def fit(self,replay_buffer,advantages):
        zipped_samples = list(zip(*replay_buffer))        
        states , actions ,rewards , next_states ,dones = zipped_samples        
        labels = np.zeros([len(actions),self.action_space])
        labels[np.arange(len(actions)) , actions] = 1 
        
        advantages = np.vstack(advantages)        
        labels = np.vstack(labels)
        y_true = np.hstack([np.array(advantages), labels])
        loss = self.model.train_on_batch(np.array(states),y_true)
        return loss
                
    def selecte_action(self,state):
        state = state[np.newaxis,:]
        probabilities = self.model.predict(state)[0]
        action = np.random.choice(self.action_space,p=probabilities)
        return action
        
class Critic():
    def __init__(self,state_space,model,optimizer= Adam(lr=0.001), gamma=0.98):
        self.model = model
        self.model.compile(optimizer=optimizer,loss='mse')
        self.gamma = gamma
        
    def fit(self,replay_buffer):
    
        zipped_samples = list(zip(*replay_buffer))        
        states , actions ,rewards , next_states ,dones = zipped_samples        
        critic_values = self.model.predict(np.array(states))
        next_critic_values = self.model.predict(np.array(next_states))
                
        y_true = []
        advantages = []
        for i in range(len(rewards)):
            # critic
            critic_value = critic_values[i]
            next_critic_value = next_critic_values[i][0]
            
            if dones[i] : 
                target = rewards[i]
            else:
                target = rewards[i] + self.gamma * next_critic_value
            y_true.append(target)
            # actor
            advantage = rewards[i] + self.gamma * next_critic_value - critic_value[0]
            advantages.append(advantage)
        loss = self.model.train_on_batch(np.array(states),np.array(y_true))
        return loss,advantages
        
class A2C():
    def __init__(self,state_space,action_space,gamma=0.98):
        self.state_space = state_space
        self.action_space = action_space
        self.gamma = gamma
        
class A2C_Agent():
    def __init__(self,a2c,actor,critic):
        self.a2c = a2c
        state_space = self.a2c.state_space
        action_space = self.a2c.action_space
        gamma = self.a2c.gamma
        self.actor = Actor(state_space,action_space,actor)
        self.critic = Critic(state_space,critic,gamma=gamma)
        self.replay_buffer = []
    def fit(self,env,episodes=1000):
    
        total_reward_list = []
        loss_list = []
        mean_reward=0
        for episode in range(episodes):
            state = env.reset()[0]
            state = np.array(state)
            done = False          
            total_reward = 0
            while not done:
            
                action = self.actor.selecte_action(state)
                next_state, reward, done, truncated, info = env.step(action)    
                next_state = np.array(next_state)
                
                self.store_transition(state ,action ,reward,next_state,done)
                
                state = next_state
                
                total_reward += reward                                
                
                if done or truncated:                    
                    break
            mean_reward += total_reward
            total_loss = self.replay()
            
            loss_list.append(total_loss)
            if total_reward > 3000:
                break
            print("episode : %d/%d , score: %d " %(episode,episodes,total_reward))
            if episode % 5 == 0:
                total_reward_list.append(mean_reward/5)
                mean_reward=0
            
        env.close()
        return total_reward_list,loss_list
        
    def store_transition(self,state,action,reward,next_state,done):
        self.replay_buffer.append((state,action,reward,next_state,done))
        
    def replay(self):
        loss,advantages = self.critic.fit(self.replay_buffer)
        loss = self.actor.fit(self.replay_buffer,advantages)
        self.replay_buffer = []
    
    def test(self,env,episodes):
        total_reward_list = []
        for episode in range(episodes):
            state = env.reset()[0]
            state = np.array(state)
            
            done = False            
            total_reward = 0
            print("Episode :",episode)
            
            while not done:
                action = self.actor.selecte_action(state)
                #print(action)
                next_state, reward, done, truncated, info = env.step(action) 
                state = np.array(next_state)             
                total_reward += reward
                                
                
                if done or truncated:                    
                    break
            print("Total reward : ",total_reward)
            total_reward_list.append(total_reward)
        env.close()
        return total_reward_list
    