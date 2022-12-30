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
def Mini_Batch(buffer):
    minibatch = random.sample(buffer)
    zipped_samples = list(zip(*minibatch))                     
    states , actions ,rewards , p_olds ,advantages = zipped_samples
class Actor():
    def __init__(self,state_space,action_space,model,epsilon=0.2,optimizer=Adam(lr=0.00025),batch_size=32,):
        self.model = model
        self.action_space = action_space
        self.state_space = state_space    
        self.epsilon = epsilon
        self.batch_size = batch_size
        def custom_loss(y_true,y_pred):        
            advantages, p_old, actions = y_true[:, :1], y_true[:, 1:1 + self.action_space], y_true[:, 1 + self.action_space:]  
            ENTROPY_LOSS = 0.001
            old_prob = actions*p_old
            new_prob = actions*y_pred
            new_prob = K.clip(new_prob,1e-10,1)
            old_prob = K.clip(old_prob,1e-10,1)

            ratio = K.exp(K.log(new_prob)-K.log(old_prob))
            
            p1 = ratio * advantages
            p2 = K.clip(ratio,min_value = 1 - self.epsilon,max_value=1 + self.epsilon)* advantages
            actor_loss = -K.mean(K.minimum(p1, p2))

            # policy Entropy 
            entropy = -(y_pred * K.log(y_pred + 1e-10))
            entropy = ENTROPY_LOSS * K.mean(entropy)
            
            total_loss = actor_loss - entropy

            return total_loss                        
            

        # Learning rate decay/scheduling
        lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=3e-4,
        decay_steps=10000,
        decay_rate=0.98)
        
        # gradient clipping  , elsilon=1e-5  ,beta = 0.9
        self.model.compile(optimizer=Adam(lr=0.00025,clipvalue=0.5,epsilon=1e-5,beta_1=0.9), loss=custom_loss)       
        
    def selecte_action(self,state):
        state = state[np.newaxis,:]
        probabilities = self.model.predict(state)[0]
        action = np.random.choice(self.action_space,p=probabilities)
        return action,probabilities

    def normalization(self,A):
        mean = np.mean(A)
        std = np.std(A)
        A = (A-mean)/(std+10e-8)
        return A

    def Mini_Batch(self,buffer):
        print(buffer.shape)
        minibatch = random.sample(buffer,self.batch_size)
        zipped_samples = list(zip(*minibatch))                     
        advantages,labels,old_probs = zipped_samples

        return advantages,labels,old_probs

    def train(self,states,actions,old_probs,advantages,iteration):
        # One hot encode. Get the probability of agent picked
        labels = np.zeros([len(actions),self.action_space])
        labels[np.arange(len(actions)) , actions] = 1 
        
        # advantages normalization 
        advantages = self.normalization(advantages)

        

        # put it together
        advantages = np.vstack(advantages)        
        labels = np.vstack(labels)
        old_probs = np.vstack(old_probs)
       
        
        actor_y_true = np.hstack([np.array(advantages),old_probs, labels])# [ advantages, p_old, actions ]


        # random mini-batch 
        #if len(advantages) > self.batch_size:            
        #    advantages,labels,old_probs = self.Mini_Batch(actor_y_true)


        loss = self.model.fit(np.array(states),actor_y_true,epochs=iteration,verbose=0,shuffle=True)
        
        return loss

class Critic():
    def __init__(self,state_space,model,optimizer= Adam(lr=0.00025), gamma=0.99,lamda=0.9,batch_size=32):
        self.model = model
        self.gamma = gamma
        self.lamda  =lamda
        self.batch_size = batch_size
        def custom_loss(y_true, y_pred):
            y_true, old_value = y_true[:,:1],y_true[:,1:2]
            
            # Loss Clipping
            LOSS_CLIPPING = 0.2
            clipped_value_loss = old_value + K.clip(y_pred - old_value, -LOSS_CLIPPING, LOSS_CLIPPING)
            v_loss1 = (y_true - clipped_value_loss) ** 2
            v_loss2 = (y_true - y_pred) ** 2
    
            value_loss = 0.5 * K.mean(K.maximum(v_loss1, v_loss2))
            return value_loss#K.sum((y_true-y_pred)**2)


        self.model.compile(optimizer=Adam(lr=0.00025,clipvalue=0.5,epsilon=1e-5,beta_1=0.9),loss=custom_loss)

    # Generalized Advantage estimation
    def GAE(self, rewards, dones, values, next_values, gamma = 0.99, lamda = 0.9, normalize=True):
        deltas = [r + gamma * (1 - d) * nv - v for r, d, nv, v in zip(rewards, dones, next_values, values)]
        deltas = np.stack(deltas)
        gaes = copy.deepcopy(deltas)
        for t in reversed(range(len(deltas) - 1)):
            gaes[t] = gaes[t] + (1 - dones[t]) * gamma * lamda * gaes[t + 1] 

        target = gaes + values
        return gaes,target
    
    # Critic training "iteration" times
    def train(self,states,target,iteration):

        # Prepare value 
        critic_values = self.model.predict(np.array(states))


        # put it together
        target = np.vstack(target)        
        critic_values = np.vstack(critic_values)
        y_true = np.hstack([target,critic_values])# [ target, old_value]

        loss = self.model.fit(np.array(states),y_true,epochs=iteration,verbose=0,shuffle=True)
        return loss

class RunningMeanStd:
    # Dynamically calculate mean and std
    def __init__(self, shape):  # shape:the dimension of input data
        self.n = 0
        self.mean = np.zeros(shape)
        self.S = np.zeros(shape)
        self.std = np.sqrt(self.S)

    def update(self, x):
        x = np.array(x)
        self.n += 1
        if self.n == 1:
            self.mean = x
            self.std = x
        else:
            old_mean = self.mean.copy()
            self.mean = old_mean + (x - old_mean) / self.n
            self.S = self.S + (x - old_mean) * (x - self.mean)
            self.std = np.sqrt(self.S / self.n )

class RewardScaling:
    def __init__(self, shape=1, gamma=0.99):
        self.shape = shape  # reward shape=1
        self.gamma = gamma  # discount factor
        self.running_ms = RunningMeanStd(shape=self.shape)
        self.R = np.zeros(self.shape)

    def __call__(self, x):
        self.R = self.gamma * self.R + x
        self.running_ms.update(self.R)
        x = x / (self.running_ms.std + 1e-8)  # Only divided std
        return x

    def reset(self):  # When an episode is done,we should reset 'self.R'
        self.R = np.zeros(self.shape)

class Normalization:
    def __init__(self, shape):
        self.running_ms = RunningMeanStd(shape=shape)

    def __call__(self, x, update=True):
        # Whether to update the mean and std,during the evaluating,update=Flase
        if update:  
            self.running_ms.update(x)
        x = (x - self.running_ms.mean) / (self.running_ms.std + 1e-8)

        return x

class PPO2():
    def __init__(self,state_space,action_space,gamma=0.99,lamda=0.9,buffer_size=2000,batch_size=128,epsilon=0.2,iteration=10,warm_up=20,stop_score=1000,reward_scaling=True):
        self.state_space = state_space
        self.action_space = action_space
        self.gamma = gamma
        self.buffer_size = buffer_size
        self.epsilon = epsilon
        self.iteration = iteration
        self.batch_size = batch_size
        self.warm_up = warm_up
        self.stop_score =stop_score
        self.lamda = lamda
        self.reward_scaling = reward_scaling
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
        self.lamda = self.ppo2.lamda
        self.reward_scaling_flag = self.ppo2.reward_scaling
        self.batch_size = self.ppo2.batch_size
        self.state_normalization = Normalization(self.state_space)
        self.reward_scaling = RewardScaling()
        #Building actor、critic network
        actor =  Model([actor_state_input],[actor_output])
        critic = Model([critic_state_input],[critic_output])
        self.actor = Actor(self.state_space,self.action_space,actor,self.epsilon,batch_size=self.batch_size)
        self.critic = Critic(self.state_space,critic,gamma=self.gamma,lamda=self.lamda,batch_size=self.batch_size)
        #initial replay buffer
        self.batch_buffer = deque(maxlen = self.batch_size*4)
        self.replay_buffer = []
        # Output the image of actor-critic
        #model = Model([state_input],[actor_output,critic_output])
        tf.keras.utils.plot_model(actor, to_file='actor_critic_model_plot1.png', show_shapes=True, show_layer_names=True)
        tf.keras.utils.plot_model(critic, to_file='actor_critic_model_plot2.png', show_shapes=True, show_layer_names=True)

    def fit(self,env,episodes=1000,test_visual=False):
        
        reward_list = []
        average_reward_list = []
        for episode in range(episodes):

            # reset reward scaling
            self.reward_scaling.reset()
            
            state = env.reset()[0]
            state = np.array(state)
            # state normalization
            #state = self.state_normalization(state)


            done = False          
            total_reward = 0

            

            while not done:

                # Selecte action and return probability
                action,old_prob = self.actor.selecte_action(state)

                # interaction with environment
                next_state, reward, done, truncated, info = env.step(action)    
                next_state = np.array(next_state)

                # state normalization
                #state = self.state_normalization(state)
                #print(state)

                #reward scaling
                
                s_reward = self.reward_scaling(reward)[0]

                #print(s_reward)
                #storage data
                if self.reward_scaling_flag:
                    self.storage_transition(state ,action ,reward,next_state,done,old_prob)
                else:
                    self.storage_transition(state ,action ,s_reward,next_state,done,old_prob)
                state = next_state
                
                total_reward += reward                                
                #reward_list.append(total_reward)
                if done or truncated:                                    
                    break

            self.replay(episode)
            reward_list.append(total_reward)
            avg_reward = sum(reward_list[-50:])/len(reward_list[-50:])
            average_reward_list.append(avg_reward)
            if avg_reward > self.stop_score:
                break

            print("episode : %d/%d\t,score: %d\t,avg_score:%f" %(episode,episodes,total_reward,avg_reward))
            
            
        env.close()
        return reward_list,average_reward_list
        
    def storage_transition(self,state,action,reward,next_state,done,old_prob):
        self.replay_buffer.append((state,action,reward,next_state,done,old_prob))
    def storage_Mini_Batch(self,state,action,old_probs,advantages,target):
        for i in range(len(action)):            
            self.batch_buffer.append((state[i],action[i],old_probs[i],advantages[i],target[i]))

    def replay(self,episode):
        
        zipped_samples = list(zip(*self.replay_buffer))                     
        states , actions ,rewards , next_states ,dones,old_probs = zipped_samples   

        # Prepare value  next_value
        critic_values = self.critic.model.predict(np.array(states))
        next_critic_values = self.critic.model.predict(np.array(next_states))

        # TD error and GAE(gen)
        advantages, target = self.critic.GAE(rewards, dones, np.squeeze(critic_values), np.squeeze(next_critic_values))
        # Storage states advantage actions 
        self.storage_Mini_Batch(states,actions,old_probs,advantages,target)
        self.replay_buffer = []

        # Mini batch
        if len(self.batch_buffer) < self.batch_size:
            return
        for i in range(self.iteration):
            minibatch = random.sample(self.batch_buffer, self.batch_size)
            zipped_samples = list(zip(*minibatch))                     
            states,actions,old_probs,advantages,target = zipped_samples 

            loss = self.critic.train(states,target,1)   

            # actor warm up
            if episode < self.warm_up:
                return advantages 
            loss = self.actor.train(states,actions,old_probs,advantages,1)

        
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
    