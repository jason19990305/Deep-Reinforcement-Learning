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
#tensorflow.config.experimental_run_functions_eagerly(True)
tf.compat.v1.disable_eager_execution()
class PPO():
    def __init__(self,state_space, action_space,gamma=0.99,epsilon=0.2,batch_size=32,buffer_size=2000,iteration=80):
        self.state_space = state_space
        self.action_space = action_space
        #self.training_episode = training_episode   
        self.gamma = gamma
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.iteration =iteration
        self.buffer_size = buffer_size
class PPOAgent():

    def __init__(self,state_input,action_output,env,ppo,optimizer = Adam(lr=0.001),update_model=1):
        self.ppo = ppo
        self.env = env
        self.optimizer = optimizer
        self.update_model = update_model
        
        
        #self.kl_max = 0
        actions = Input(shape=[self.ppo.action_space])
        advantages = Input(shape=[1]) 
        p_old =  Input(shape=[self.ppo.action_space])  
          
        self.policy = Model([state_input],[action_output])# for training
        
        #**********
        
        def custom_loss(y_true,y_pred):   
            advantages, p_old, actions = y_true[:, :1], y_true[:, 1:1 + self.ppo.action_space], y_true[:, 1 + self.ppo.action_space:]     
            
            
            old_prob = actions*p_old # The action agent picked
            new_prob = actions*y_pred# The action agent picked
            
            new_prob = K.clip(new_prob,1e-10,1-1e-10)
            old_prob = K.clip(old_prob,1e-10,1-1e-10)
            
            ratio = new_prob/(old_prob+1e-10) 
            p1 = ratio * advantages
            p2 = K.clip(ratio,min_value = 1 - self.ppo.epsilon,max_value=1 + self.ppo.epsilon)* advantages
            loss = -K.sum(K.minimum(p1,p2))
            
            return loss
            #ENTROPY_LOSS = 5e-3
            #loss = - K.mean(K.minimum(p1,p2) + ENTROPY_LOSS * - (new_prob * K.log(new_prob+1e-10)))
            #log_lik =y_true*K.log(out) # total reward * log( p_(a|s))
        self.model = Model([state_input],[action_output])
        self.old_model = clone_model(self.model)
        self.policy.compile(optimizer=optimizer, loss=custom_loss)
        tf.keras.utils.plot_model(self.policy, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
        #***************
        
        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []
        self.replay_buffer = deque(maxlen = self.ppo.buffer_size)
        
        self.trajectory_memory = []
        self.probability_old_memory = []
        
        self.loss_list = []
        
    def store_transition(self,state,action,reward,p_old):
        self.state_memory.append(state)
        self.action_memory.append(action)
        self.reward_memory.append(reward)
        self.probability_old_memory.append(p_old)
        
    def store_replay_buffer(self):
        advantage = self.advantage(self.reward_memory)
        #print(advantage)
        for i in range(len(advantage)):
            step = (self.state_memory[i], self.action_memory[i], self.reward_memory[i],self.probability_old_memory[i],advantage[i])
            self.replay_buffer.append(step)
        self.clear_memory()
        
    def clear_memory(self):
        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []
        self.probability_old_memory = []
        
    def advantage(self,rewards):
        G = np.zeros_like(rewards)
        for t in range(len(rewards)):
            G_sum = 0
            discount = 1
            for k in range(t,len(rewards)):
                G_sum += rewards[k]*discount
                discount *= self.ppo.gamma
            G[t] = G_sum
        mean = np.mean(G)
        std = np.std(G) if np.std(G)>0 else 1
        G = (G-mean)/std
        return np.array(G)
    
    def replay(self,train=True):      
    
        total_loss = []
        if len(self.replay_buffer) < self.ppo.batch_size:
            return
        # replay_buffer = list()
        
        
        total_cost = 0
        for i in range(self.ppo.iteration):
        
            minibatch = random.sample(self.replay_buffer, self.ppo.batch_size)# random sample at replay_buffer
            zipped_samples = list(zip(*minibatch))
            states , actions ,rewards , p_olds ,advantages = zipped_samples
            action_picked = np.zeros([len(actions),self.ppo.action_space])
            action_picked[np.arange(len(actions)) , actions] = 1 
            
            # reshape            
            advantages = np.vstack(advantages) # (batch_size , 1)
            p_olds = np.vstack(p_olds)         # (batch_size , action num)
            action_picked = np.vstack(action_picked)# (batch_size , action num)
            
            y_true = np.hstack([advantages, p_olds, action_picked])
            cost = self.policy.train_on_batch(np.array(states),y_true)
            total_cost += cost
        
            
        #print( np.sum(total_loss)/len(total_loss))
        return total_cost
        #return np.sum(total_loss)/len(total_loss)
        
    def selecte_action(self,state):
        state = state[np.newaxis,:]
        probabilities = self.old_model.predict(state)[0]
        #probabilities = self.model.predict(state)[0]
        #print(probabilities)
        try:
            action = np.random.choice(self.ppo.action_space,p=probabilities)
        except Exception as e:
            print(e)
            return None
        
        return action,probabilities
        
    def save_weight(self,file_name ='dqn_weights.h5'):
        self.model.save_weights('./'+file_name)
        
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
            
                action,p_old = self.selecte_action(state)
                if action == None:
                    break
                next_state, reward, done, truncated, info = env.step(action)    
                next_state = np.array(next_state)
                
                self.store_transition(state ,action ,reward,p_old)
                
                
                
                state = next_state
                
                total_reward += reward                                
                
                if done or truncated:                    
                    break;
            mean_reward += total_reward
            self.store_replay_buffer()
            total_loss = self.replay()
            loss_list.append(total_loss)
            if episode % self.update_model == 0:
                print("Reset old Model")
                self.old_model.set_weights(self.model.get_weights()) 
                
            print("episode : %d/%d , score: %d " %(episode,episodes,total_reward))
            if episode % 5 == 0:
                total_reward_list.append(mean_reward/5)
                mean_reward=0
            
        env.close()
        return total_reward_list,loss_list
    def load_weights(self,file_name ='ppo_weights.h5'):
        self.eval_model.load_weights('./'+file_name)
        self.target_model.load_weights('./'+file_name)
    
    def test(self,env,episodes):
        total_reward_list = []
        self.old_model.set_weights(self.model.get_weights()) 
        for episode in range(episodes):
            state = env.reset()[0]
            state = np.array(state)
            
            done = False            
            total_reward = 0
            print("Episode :",episode)
            
            while not done:
                action,_ = self.selecte_action(state)
                #print(action)
                next_state, reward, done, truncated, info = env.step(action) 
                state = np.array(next_state)             
                total_reward += reward
                                
                
                if done or truncated:                    
                    break;
            print("Total reward : ",total_reward)
            total_reward_list.append(total_reward)
        env.close()
        return total_reward_list
        
        