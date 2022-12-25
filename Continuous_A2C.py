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
    def __init__(self,state_space,action_space,model,epsilon,sigma,sigma_reduce,optimizer=Adam(lr=0.001)):
        self.model = model
        self.action_space = action_space
        self.state_space = state_space    
        self.epsilon = epsilon  
        self.sigma = sigma
        self.sigma_reduce = sigma_reduce
        self.min_sigma = 0.1
        def pdf(mu,sigma,x):
            p1 = 1/(np.sqrt(2*np.pi)*sigma**2)
            p2 = -0.5*(((x-mu)/sigma)**2)
            prob = p1*K.exp(p2)
            return prob
        def custom_loss(y_true,y_pred):
            advantages, sigma , actions ,p_olds = y_true[:, :1], y_true[:, 1:2]  , y_true[:, 2:2+self.action_space],  y_true[:, 2+self.action_space : 2+self.action_space*2]
            #advantages, p_old, actions = y_true[:, :1], y_true[:, 1:1 + self.action_space], y_true[:, 1 + self.action_space:]    


            old_prob = p_olds
            new_prob = pdf(y_pred,sigma,actions)
            r = new_prob/(old_prob+1e-10)
            
            p1 = r * advantages
            p2 = K.clip(r,min_value = 1 - self.epsilon,max_value=1 + self.epsilon)* advantages

            # summation over the min(p1,p2)
            loss = - K.sum(K.minimum(p1,p2))
            return loss                        
        self.model.compile(optimizer=optimizer, loss=custom_loss)
    def pdf_np(self,mu,sigma,x):
            p1 = 1/(np.sqrt(2*np.pi)*sigma**2)
            p2 = -0.5*(((x-mu)/sigma)**2)
            prob = p1*np.exp(p2)
            return prob       

    def selecte_action(self,state):
        state = state[np.newaxis,:]
        probabilities = self.model.predict(state)[0]
        mu = probabilities
        action = np.random.normal(mu,self.sigma,(self.action_space))
        action = np.clip(action,-1,1)

        p_old = self.pdf_np(mu,self.sigma,action)
        
        return action,p_old
    def reduce_sigma(self):
        print("Sigma:",self.sigma)
        if self.sigma >= self.min_sigma:
            self.sigma *= self.sigma_reduce

    def train(self,states,actions,old_probs,advantages):
        # One hot encode. Get the probability of agent picked
        sigma = np.ones([len(actions)]) * self.sigma
        #print("Old prob:",old_probs)
        advantages = np.vstack(advantages)      
        sigmas = np.vstack(sigma)  
        actions = np.vstack(actions)
        old_probs = np.vstack(old_probs)

        
        actor_y_true = np.hstack([np.array(advantages),sigmas,actions,old_probs])# [ advantages, sigma , actions ,p_olds ]
        loss = self.model.train_on_batch(np.array(states),actor_y_true)        

        return loss
class Critic():
    def __init__(self,state_space,model,optimizer= Adam(lr=0.001), gamma=0.98):
        self.model = model
        self.model.compile(optimizer=optimizer,loss='mse')
        self.gamma = gamma
        
    def train(self,states,rewards,dones,next_states):
        # Get the target value for training critic
        critic_values = self.model.predict(np.array(states))
        next_critic_values = self.model.predict(np.array(next_states))
        y_true = []
        advantages = []
        for i in range(len(rewards)):
            # critic
            critic_value = critic_values[i]
            next_critic_value = next_critic_values[i][0] # The next value 
                
            if dones[i] : 
                target = rewards[i]
            else:
                target = rewards[i] + self.gamma * next_critic_value

            y_true.append(target)

            # advantage for training actor
            advantage = target - critic_value[0]
            advantages.append(advantage)
        loss = self.model.train_on_batch(np.array(states),np.array(y_true))

        return loss,advantages

class A2C():
    def __init__(self,state_space,action_space,gamma=0.98,buffer_size=2000,epsilon=0.2,iteration=10,batch_size=32,warm_up=10,sigma=3,sigma_reduce=0.99):
        self.state_space = state_space
        self.action_space = action_space
        self.gamma = gamma
        self.buffer_size = buffer_size
        self.epsilon = epsilon
        self.iteration = iteration
        self.batch_size = batch_size
        self.warm_up = warm_up
        self.sigma = sigma
        self.sigma_reduce = sigma_reduce

class A2C_Agent():
    def __init__(self,a2c,state_input,actor_output,critic_output):
        # Parameter
        self.a2c = a2c
        self.state_space = self.a2c.state_space
        self.action_space = self.a2c.action_space
        self.gamma = self.a2c.gamma 
        self.epsilon = self.a2c.epsilon
        self.warm_up = self.a2c.warm_up
        self.iteration = self.a2c.iteration
        self.batch_size = self.a2c.batch_size
        self.buffer_size = self.a2c.buffer_size
        self.sigma = self.a2c.sigma
        self.sigma_reduce = self.a2c.sigma_reduce
        #Building actor、critic network
        actor =  Model([state_input],[actor_output])
        critic = Model([state_input],[critic_output])
        self.actor = Actor(self.state_space,self.action_space,actor,self.epsilon,self.sigma,self.sigma_reduce)
        self.critic = Critic(self.state_space,critic,gamma=self.gamma)
        #initial replay buffer
        self.replay_buffer = deque(maxlen = self.buffer_size)

        # Output the image of actor-critic
        model = Model([state_input],[actor_output,critic_output])
        tf.keras.utils.plot_model(model, to_file='actor_critic_model_plot.png', show_shapes=True, show_layer_names=True)

    def fit(self,env,episodes=1000,test_visual=False):
        
        total_reward_list = []
        loss_list = []
        mean_reward=0
        for episode in range(episodes):
            state = env.reset()[0]
            state = np.array(state)
            done = False          
            total_reward = 0
            while not done:
                # Selecte action
                action,old_prob = self.actor.selecte_action(state)
                next_state, reward, done, truncated, info = env.step(action)    
                next_state = np.array(next_state)
                # Store data in replay buffer
                self.store_transition(state ,action ,reward,next_state,done,old_prob)
                
                state = next_state
                
                total_reward += reward                                
                
                if done or truncated:  
                                  
                    break
            mean_reward += total_reward
            total_loss = self.replay(episode)
            
            loss_list.append(total_loss)
            
            print("episode : %d/%d , score: %d " %(episode,episodes,total_reward))
            if episode % 3 == 0:
                total_reward_list.append(mean_reward/3)
                mean_reward=0
        env.close()
        return total_reward_list,loss_list
        
    def store_transition(self,state,action,reward,next_state,done,old_prob):
        self.replay_buffer.append((state,action,reward,next_state,done,old_prob))
    
    def replay(self,episode):
        if len(self.replay_buffer) < self.batch_size:
            return
        
        for epoch in range(self.iteration):
            # Random Sample data
            minibatch = random.sample(self.replay_buffer, self.batch_size)
            #minibatch = self.replay_buffer
            zipped_samples = list(zip(*minibatch))                     
            states , actions ,rewards , next_states ,dones,old_probs = zipped_samples              

            # Update Critic                
            loss ,advantages= self.critic.train(states,rewards,dones,next_states)            
            # Update Actor
            if episode < self.warm_up :
                continue
            loss = self.actor.train(states,actions,old_probs,advantages)

        # reduce sigma when end of update
        self.actor.reduce_sigma()

    def test(self,env,episodes):

        total_reward_list = []
        for episode in range(episodes):
            state = env.reset()[0]
            state = np.array(state)
            done = False            
            total_reward = 0
            print("Episode :",episode)
            
            while not done:
                action,old_prob = self.actor.selecte_action(state)
                next_state, reward, done, truncated, info = env.step(action) 
                state = np.array(next_state)             
                total_reward += reward
                                
                
                if done or truncated:                    
                    break
            print("Total reward : ",total_reward)
            total_reward_list.append(total_reward)
        env.close()
        return total_reward_list
    