import gym
import random
import numpy as np
from collections import deque
from tensorflow.keras.models import Sequential   ,clone_model
from tensorflow.keras.layers import Dense 
from tensorflow.keras.optimizers import Adam

class QParameter():
    def __init__(self, state_space, action_space, epsilon = 0.01,e_decay=0.99 ,gamma = 0.95, batch_size=20,buffer_size = 2000,update_target_model = 10):
        self.state_space = state_space
        self.action_space = action_space
        self.epsilon = epsilon
        self.e_decay = e_decay    
        self.gamma = gamma
        self.replay_buffer = deque(maxlen = buffer_size)
        self.batch_size = batch_size
        self.update_target_model = update_target_model
    def epsilon_redece(self):
        self.epsilon *= self.e_decay
        
    def store_transition(self,state,action,reward,next_state,done):
        self.replay_buffer.append((state,action,reward,next_state,done))
    
class DQNAgent():

    def __init__(self,model,env,dqn_para,learning_rate = 0.001):
        self.dqn_para = dqn_para
        self.learning_rate = learning_rate
        self.env = env
        self.eval_model = model
        self.target_model = clone_model(model)
        
        self.loss_list = []
    def replay(self):
        if len(self.dqn_para.replay_buffer) < self.dqn_para.batch_size:
            return
        minibatch = random.sample(self.dqn_para.replay_buffer, self.dqn_para.batch_size)
        
        target_batch = []
        
        zipped_samples = list(zip(*minibatch))
        
        states , actions ,rewards , next_states ,dones = zipped_samples
        
        #targets = self.target_model.predict(np.array(states))
        next_q_rows = self.eval_model.predict(np.array(next_states))
        q_value_rows = self.eval_model.predict(np.array(states))        
        
        for i in range(self.dqn_para.batch_size):
            
            q_value = q_value_rows[i]     
            target = rewards[i]
            if not dones[i]:
                next_q_value =  np.amax(next_q_rows[i])
                target = rewards[i] + self.dqn_para.gamma * next_q_value
            q_value[0][actions[i]] = target
            target_batch.append(q_value)
        history = self.eval_model.fit(np.array(states),np.array(target_batch),epochs=1,verbose = 0)
        
        # Keeping track of loss
        loss = history.history['loss'][0]
        self.loss_list.append(loss)
        #print("Loss:",loss)
        return loss
            
    def target_replacement(self):
        self.target_model.set_weights(self.eval_model.get_weights())  
        
    def selecte_action(self,state):
        if random.random() < self.dqn_para.epsilon:
            return random.choice(range(self.dqn_para.action_space))
        predict = self.eval_model.predict(state)
        return np.argmax(predict[0])
    
    def save_weight(self,file_name ='dqn_weights.h5'):
        self.eval_model.save_weights('./'+file_name)
        
    def fit(self,env,episodes=1000):
    
        total_reward_list = []
        
        for episode in range(episodes):
            state = env.reset()[0]
            
            state = state.reshape([1,self.dqn_para.state_space])
            done = False            
            total_reward = 0
            
            
            while not done:
                
                action = self.selecte_action(state)
                
                next_state, reward, done, truncated, info = env.step(action) 
                next_state = next_state.reshape([1,self.dqn_para.state_space])   
                
                self.dqn_para.store_transition(state ,action ,reward ,next_state ,done)
                
                self.replay()# Training
                state = next_state
                
                total_reward += reward                                
                
                if done or truncated:                    
                    break;
            if episode % self.dqn_para.update_target_model == 0:
                self.target_replacement()
            print("episode : %d/%d , score: %d , epsilon: %f " %(episode,episodes,total_reward,self.dqn_para.epsilon))
            total_reward_list.append(total_reward)
            self.dqn_para.epsilon_redece()
        env.close()
        return total_reward_list,self.loss_list
    def load_weights(self,file_name ='dqn_weights.h5'):
        self.eval_model.load_weights('./'+file_name)
        self.target_model.load_weights('./'+file_name)
    
    def test(self,env,episodes):
        self.dqn_para.epsilon = 0
        total_reward_list = []
        for episode in range(episodes):
            state = env.reset()[0]
            
            state = state.reshape([1,self.dqn_para.state_space])
            done = False            
            total_reward = 0
            print("Episode :",episode)
            
            while not done:
                
                action = self.selecte_action(state)
                
                next_state, reward, done, truncated, info = env.step(action) 
                next_state = next_state.reshape([1,self.dqn_para.state_space])  
                state = next_state               
                total_reward += reward
                                
                
                if done or truncated:                    
                    break;
            print("Total reward : ",total_reward)
            total_reward_list.append(total_reward)
        env.close()
        return total_reward_list
        
        