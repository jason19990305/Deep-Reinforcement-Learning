import gym
import time
import matplotlib.pyplot as plot

env = gym.make("Breakout-v0",render_mode='human')

from gym.utils import play

#This is a play module that allows you to interact with some of the games.
#**It will need install pygame**
#Not every environment in OpenAi gym is interactive or allows you to play as a human.

#play.play(env,zoom=3)
env.reset()
print("Action Space:",env.action_space.n)

for steps in range(500):
    time.sleep(0.01)
    random_action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(random_action)
    
    print("Reward:",reward)
    print("Done:",terminated)
    print("Info:",info)
    #print("Observation:",observation)
    
    
    if terminated or truncated:
        observation, info = env.reset()
    
env.close()