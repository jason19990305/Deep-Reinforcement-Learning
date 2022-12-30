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
import copy
from matplotlib import pyplot as plt

tf.compat.v1.disable_eager_execution()
class Actor():
    def __init__(self,state_space,action_space,model,epsilon,optimizer=Adam(lr=0.001)):
        self.model = model
        self.action_space = action_space
        self.state_space = state_space    
        self.epsilon = epsilon
        def custom_loss(y_true,y_pred):        
            advantages, p_old, actions = y_true[:, :1], y_true[:, 1:1 + self.action_space], y_true[:, 1 + self.action_space:]  
            ENTROPY_LOSS = 0.001
            old_prob = actions*p_old
            new_prob = actions*y_pred
            new_prob = K.clip(new_prob,1e-10,1-1e-10)
            old_prob = K.clip(old_prob,1e-10,1-1e-10)

            ratio = K.exp(K.log(new_prob)-K.log(old_prob))
            
            p1 = ratio * advantages
            p2 = K.clip(ratio,min_value = 1 - self.epsilon,max_value=1 + self.epsilon)* advantages
            actor_loss = -K.mean(K.minimum(p1, p2))
            entropy = -(y_pred * K.log(y_pred + 1e-10))
            entropy = ENTROPY_LOSS * K.mean(entropy)            
            total_loss = actor_loss - entropy
            return total_loss                        
            
        self.model.compile(optimizer=optimizer, loss=custom_loss)       
        
    def selecte_action(self,state):
        state = state[np.newaxis,:]
        probabilities = self.model.predict(state)[0]
        action = np.random.choice(self.action_space,p=probabilities)
        return action,probabilities
    
    def train(self,states,actions,old_probs,advantages,iteration):
        # One hot encode. Get the probability of agent picked
        labels = np.zeros([len(actions),self.action_space])
        labels[np.arange(len(actions)) , actions] = 1 
               
        advantages = np.vstack(advantages)        
        labels = np.vstack(labels)

        # advantages normalization 
        #mean = np.mean(advantages)
        #std = np.std(advantages) 
        #advantages = (advantages-mean)/(std+10e-8)
        #print("advantages:",advantages)
        
        actor_y_true = np.hstack([np.array(advantages),old_probs, labels])# [ advantages, p_old, actions ]
        loss = self.model.fit(np.array(states),actor_y_true,epochs=iteration,verbose=0,shuffle=True)
        
        return loss
class Critic():
    def __init__(self,state_space,model,optimizer= Adam(lr=0.001), gamma=0.98):
        self.model = model

        def custom_loss(y_true, y_pred):
            y_true, old_value = y_true[:,:1],y_true[:,1:2]
            LOSS_CLIPPING = 0.2
            clipped_value_loss = old_value + K.clip(y_pred - old_value, -LOSS_CLIPPING, LOSS_CLIPPING)
            v_loss1 = (y_true - clipped_value_loss) ** 2
            v_loss2 = (y_true - y_pred) ** 2
            
            value_loss = 0.5 * K.mean(K.maximum(v_loss1, v_loss2))
            return value_loss

        self.model.compile(optimizer=optimizer,loss=custom_loss)
        self.gamma = gamma

    def generalized_advantage_estimation(self, rewards, dones, values, next_values, gamma = 0.99, lamda = 0.9, normalize=True):
        deltas = [r + gamma * (1 - d) * nv - v for r, d, nv, v in zip(rewards, dones, next_values, values)]
        deltas = np.stack(deltas)
        gaes = copy.deepcopy(deltas)
        for t in reversed(range(len(deltas) - 1)):
            gaes[t] = gaes[t] + (1 - dones[t]) * gamma * lamda * gaes[t + 1]

        target = gaes + values
        if normalize:
            gaes = (gaes - gaes.mean()) / (gaes.std() + 1e-8)
        return np.vstack(gaes), np.vstack(target)

    def train(self,states,rewards,dones,next_states,iteration):
        # Get the target value for training critic
        critic_values = self.model.predict(np.array(states))
        next_critic_values = self.model.predict(np.array(next_states))

        advantages, target = self.generalized_advantage_estimation(rewards, dones, np.squeeze(critic_values), np.squeeze(next_critic_values))
        
        # put it together
        target = np.vstack(target)        
        critic_values = np.vstack(critic_values)
        y_true = np.hstack([target,critic_values])# [ target, old_value]
        #print(y_true)
        #loss = self.model.train_on_batch(np.array(states),y_true)
        loss = self.model.fit(np.array(states),y_true,epochs=iteration,verbose=0,shuffle=True)
        return loss,advantages

class PPO2():
    def __init__(self,state_space,action_space,gamma=0.98,buffer_size=2000,epsilon=0.2,iteration=10,batch_size=32,warm_up=20,stop_score=1000):
        self.state_space = state_space
        self.action_space = action_space
        self.gamma = gamma
        self.buffer_size = buffer_size
        self.epsilon = epsilon
        self.iteration = iteration
        self.batch_size = batch_size
        self.warm_up = warm_up
        self.stop_score =stop_score
class PPO2_Agent():
    def __init__(self,ppo2,actor_state_input,critic_state_input,actor_output,critic_output):
        # Parameter
        self.ppo2 = ppo2
        self.state_space = self.ppo2.state_space
        self.action_space = self.ppo2.action_space
        self.gamma = self.ppo2.gamma 
        self.epsilon = self.ppo2.epsilon
        self.warm_up = self.ppo2.warm_up
        self.iteration = self.ppo2.iteration
        self.batch_size = self.ppo2.batch_size
        self.buffer_size = self.ppo2.buffer_size
        self.stop_score = self.ppo2.stop_score
        #Building actor、critic network
        actor =  Model([actor_state_input],[actor_output])
        critic = Model([critic_state_input],[critic_output])
        self.actor = Actor(self.state_space,self.action_space,actor,self.epsilon)
        self.critic = Critic(self.state_space,critic,gamma=self.gamma)
        #initial replay buffer
        #self.replay_buffer = deque(maxlen = self.buffer_size)
        self.replay_buffer = []
        # Output the image of actor-critic
        #model = Model([state_input],[actor_output,critic_output])
        tf.keras.utils.plot_model(actor, to_file='actor_critic_model_plot1.png', show_shapes=True, show_layer_names=True)
        tf.keras.utils.plot_model(critic, to_file='actor_critic_model_plot2.png', show_shapes=True, show_layer_names=True)

    def fit(self,env,episodes=1000,test_visual=False):
        
        total_reward_list = []
        reward_list = []
        average_reward_list = []
        loss_list = []
        advantage_list = np.array([])
        for episode in range(episodes):
            state = env.reset()[0]
            state = np.array(state)
            done = False          
            total_reward = 0
            #reward_list = []
            while not done:
                # Selecte action
                action,old_prob = self.actor.selecte_action(state)
                next_state, reward, done, truncated, info = env.step(action)    
                next_state = np.array(next_state)

                # Store data in replay buffer
                self.store_transition(state ,action ,reward,next_state,done,old_prob)
                
                state = next_state
                
                total_reward += reward                                
                #reward_list.append(total_reward)
                if done or truncated:                                    
                    break
            
            advantages = self.replay(episode)
            reward_list.append(total_reward)
            avg_reward = sum(reward_list[-50:])/len(reward_list[-50:])
            average_reward_list.append(avg_reward)
            if avg_reward > self.stop_score:
                break

            print("episode : %d/%d\t,score: %d\t,avg_score:%f" %(episode,episodes,total_reward,avg_reward))
            
            
        env.close()
        return reward_list,average_reward_list
        
    def store_transition(self,state,action,reward,next_state,done,old_prob):
        self.replay_buffer.append((state,action,reward,next_state,done,old_prob))
    
    def replay(self,episode):
        zipped_samples = list(zip(*self.replay_buffer))                     
        states , actions ,rewards , next_states ,dones,old_probs = zipped_samples       
        loss ,advantages= self.critic.train(states,rewards,dones,next_states,self.iteration)
        
        loss = self.actor.train(states,actions,old_probs,advantages,self.iteration)
        self.replay_buffer = []
        return advantages
    
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
    