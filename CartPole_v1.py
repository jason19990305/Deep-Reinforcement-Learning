import gym
import time
import matplotlib.pyplot as plot

env = gym.make("CartPole-v1",render_mode='human')


env.reset()
print("Action Space:",env.action_space.n)
print("Observation Space : ",env.observation_space)
for steps in range(500):
    time.sleep(0.01)
    random_action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(random_action)
    
    
    
    
    if terminated or truncated:
        observation, info = env.reset()
    
env.close()