import gym
import time

def simple_agent(observation):
    #observation: [Position , Velocity]
    #Position -1.2 ~ 0.6
    #Velocity -0.07 ~ 0.07
    print("Observation:",observation)
    position , velocity = observation
    print("Position:",position)
    print("Velocity:",velocity)
    # 0   <=====
    # 1   None
    # 2   =====>
    if -0.1 < position and position < 0.4:
        action = 2
    elif velocity < 0 and position < -0.2:
        action = 0
    else:
        action = 1
    return action

env_name = 'MountainCar-v0'

env = gym.make(env_name,render_mode='human')

observation = env.reset()[0]
time.sleep(3)
for step in range(600):
    action = simple_agent(observation)
    observation,reward,done,truncated,info = env.step(action)
    #print("Reward:",reward)
    #print("Done:",done)
    #print("Truncated:",truncated)
    #print("Info:",info)
    print("Observation:",observation)
    time.sleep(0.01)
    
