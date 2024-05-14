import gymnasium as gym
from classes import Params, EpsilonGreedy, Qlearning
from pathlib import Path

params = Params(
    total_episodes=2000,
    learning_rate=0.8,
    gamma=0.9,
    epsilon=0.1,
    seed=123,
    n_runs=20,
    action_size=None,
    state_size=None,
    savefig_folder=Path("./img/"),
)

env = gym.make("LunarLander-v2", render_mode="rgb_array")
observation, info = env.reset(seed=42)
for _ in range(1000):
   action = env.action_space.sample()  # this is where you would insert your policy
   observation, reward, terminated, truncated, info = env.step(action)

   if terminated or truncated:
      observation, info = env.reset()

env.close()