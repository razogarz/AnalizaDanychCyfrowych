import gymnasium as gym
import matplotlib.pyplot as plt

from stable_baselines3 import A2C

env = gym.make("BipedalWalker-v3", render_mode="rgb_array")

model = A2C("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=50_000)

# Let's visualize the agent's behavior
rewards = []
infos = []

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(5000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render("human")
    print(reward, info)
    # saving rewards and info
    rewards.append(reward)
    infos.append(info)

# Plotting the rewards
plt.plot(list(rewards))
plt.xlabel("Time steps")
plt.ylabel("Reward")
plt.title("Reward over time")
plt.show()
