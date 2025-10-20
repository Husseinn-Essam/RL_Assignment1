import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np

class GridMazeEnv(gym.Env):
    """
    Custom Gym Environment for a 5x5 Grid Maze. 

    The objective is to navigate from a start cell 'S' to a goal cell 'G'
    while avoiding two bad cells 'X'.  The positions of S, G, and
    the two X cells are randomized at the start of each episode. 
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, size=5, render_mode=None):
        super(GridMazeEnv, self).__init__()
        self.size = size
        self.window_size = 512  # The size of the PyGame window

        # The state is the agent's (row, col) position.
        # This is a simplification for the tabular Policy Iteration algorithm.
        # The full state would include G and X positions, making the state space
        # too large for this method.
        self.observation_space = spaces.Discrete(self.size * self.size)
        
        # 4 discrete actions: 0:Right, 1:Up, 2:Left, 3:Down [cite: 47]
        self.action_space = spaces.Discrete(4)
        
        # Action to coordinate change mapping
        self._action_to_direction = {
            0: np.array([0, 1]),   # Right
            1: np.array([-1, 0]),  # Up
            2: np.array([0, -1]),   # Left
            3: np.array([1, 0]),    # Down
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

    def _get_obs(self):
        return self._agent_location[0] * self.size + self._agent_location[1]

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._goal_location, ord=1
            )
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Choose S, G, and two X locations randomly without replacement 
        locations = self.np_random.choice(
            self.size * self.size, size=4, replace=False
        )
        self._agent_location = np.array([locations[0] // self.size, locations[0] % self.size])
        self._start_location = self._agent_location.copy()
        self._goal_location = np.array([locations[1] // self.size, locations[1] % self.size])
        self._bad_locations = [
            np.array([locations[2] // self.size, locations[2] % self.size]),
            np.array([locations[3] // self.size, locations[3] % self.size])
        ]

        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), self._get_info()

    def step(self, action):
        # Stochastic movement: 70% intended, 15% perpendicular 
        p = self.np_random.random()
        
        if p < 0.70:
            chosen_action = action # Intended direction
        elif p < 0.85:
            chosen_action = (action + 1) % 4 # Perpendicular 1
        else:
            chosen_action = (action - 1 + 4) % 4 # Perpendicular 2
        
        direction = self._action_to_direction[chosen_action]
        
        # We use `np.clip` to make sure we don't leave the grid
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )

        # Check for termination
        terminated = np.array_equal(self._agent_location, self._goal_location)
        is_bad_location = any(np.array_equal(self._agent_location, bl) for bl in self._bad_locations)
        if is_bad_location:
            terminated = True
            
        # Reward Function Design [cite: 54]
        # +10 for reaching the goal: A strong positive incentive.
        # -10 for hitting a bad cell: A strong deterrent.
        # -0.1 for each step: Encourages finding the shortest path.
        if terminated and not is_bad_location:
            reward = 10.0
        elif terminated and is_bad_location:
            reward = -10.0
        else:
            reward = -0.1
        
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, False, info

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
        pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # Draw Goal
        pygame.draw.rect(
            canvas,
            (0, 255, 0),
            pygame.Rect(
                pix_square_size * self._goal_location[1],
                pix_square_size * self._goal_location[0],
                pix_square_size,
                pix_square_size,
            ),
        )

        # Draw Bad Cells
        for bl in self._bad_locations:
            pygame.draw.rect(
                canvas,
                (255, 0, 0),
                pygame.Rect(
                    pix_square_size * bl[1],
                    pix_square_size * bl[0],
                    pix_square_size,
                    pix_square_size,
                ),
            )
        
        # Draw Agent
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location[::-1] + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Draw grid lines
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
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
