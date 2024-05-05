import gym
from gym import spaces
import pygame
import numpy as np

# Game settings
WINDOW_SIZE = 800
BLOCK_SIZE = 10
SPEED = 10
NUMBER_OF_OBSTACLES = 10
MAX_SNAKE_LENGTH = 100
MAX_OBSTACLE_SIZE = 10

class SnakeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size=5):
        self.size = size
        self.window_size = 800
        self.observation_space = spaces.Dict(
            {
                "snake": spaces.Box(0, size - 1, shape=(MAX_SNAKE_LENGTH, 2), dtype=int),
                "food": spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "obstacles": spaces.Box(0, size - 1, shape=(MAX_OBSTACLE_SIZE, 2), dtype=int),
            }
        )
        self.action_space = spaces.Discrete(4)
        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        
        self._obstacles = []
        self.window = None
        self.clock = None

    def _get_obs(self):
        return {
            "snake": self._snake,
            "food": self._food_location,
            "obstacles": self._obstacles,
        }
    
    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._snake[0] - self._food_location, ord=1
            )
        }
    
    def reset(self, seed=None):
        super().reset(seed=seed)

        self.np_random, seed = gym.utils.seeding.np_random(seed)
        self._snake = [self.np_random.integers(0, self.size, size=2, dtype=int)]
        
        # Generate food and obstacles, checking for collisions with the snake
        self._food_location = self._snake[0]
        while np.array_equal(self._food_location, self._snake[0]):
            self._food_location = self._generate_food()
        
        self._obstacles = self._snake
        while any(np.array_equal(obstacle, self._snake[0]) for obstacle in self._obstacles):
            self._obstacles = self._generate_obstacles()

        observation = self._get_obs()
        info = {}  # Add any auxiliary information here

        if self.render_mode == "human":
            self.render()

        return observation, info
    
    
    def _generate_obstacles(self):
        # Generate obstacles here. This is just an example.
        return [self.np_random.integers(0, self.size, size=2, dtype=int) for _ in range(5)]

    def _generate_food(self):
        while True:
            food_location = self.np_random.integers(0, self.size, size=2, dtype=int)
            if all(not np.array_equal(food_location, pos) for pos in self._snake):
                return food_location

    def step(self, action):
        direction = self._action_to_direction[action]
        new_head = self._snake[0] + direction

        terminated = any(np.array_equal(new_head, pos) for pos in self._snake) or \
                    any(np.array_equal(new_head, pos) for pos in self._obstacles) or \
                    not (0 <= new_head).all() or \
                    not (new_head < self.size).all()

        if terminated:
            return self._get_obs(), -1, True, self._get_info()

        self._snake.insert(0, new_head)
        if np.array_equal(new_head, self._food_location):
            self._food_location = self._generate_food()
            reward = 1
        else:
            self._snake.pop()
            reward = 0

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, False, self._get_info()

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = self.window_size / self.size

        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._food_location,
                (pix_square_size, pix_square_size),
            ),
        )
        for pos in self._snake:
            pygame.draw.circle(
                canvas,
                (0, 0, 255),
                (pos + 0.5) * pix_square_size,
                pix_square_size / 3,
            )

        for pos in self._obstacles:  # Draw obstacles
            pygame.draw.rect(
                canvas,
                (0, 255, 0),
                pygame.Rect(
                    pix_square_size * pos,
                    (pix_square_size, pix_square_size),
                ),
            )

        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()


if __name__ == "__main__":
    env = SnakeEnv(render_mode="human", size=10)
    env.reset()
    for _ in range(100):
        env.step(env.action_space.sample())
        env.render()
    env.close()

