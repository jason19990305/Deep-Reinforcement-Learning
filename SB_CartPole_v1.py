from stable_baselines3 import PPO2
import gym

# Parallel environments
env = gym.make("CartPole-v1")

model = PPO(policy = "MlpPolicy",env =  env, verbose=1)
model.learn(total_timesteps=2000)

obs = env.reset()
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
      obs = env.reset()