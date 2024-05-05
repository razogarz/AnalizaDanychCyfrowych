from gymnasium.envs.registration import register
from gym_examples.envs.snake import SnakeEnv

register(
		id="gym_examples/snake-v0",
		entry_point="gym_examples.envs:SnakeEnv",
		max_episode_steps=300,
	)