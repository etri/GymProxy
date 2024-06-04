from typing import Optional

import gymnasium as gym
from gymnasium import spaces
import numpy as np

class SimpleEnv(gym.Env):
    def __init__(self, render_mode: Optional[str] = None):
        super(SimpleEnv, self).__init__()
        # Define action and observation space
        # Example: action space has two actions (0 and 1)
        self.action_space = spaces.Discrete(2)
        # Example: observation space is a single value between 0 and 1
        self.observation_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        self.render_mode = render_mode

    def reset(self):
        # Reset the state of the environment to an initial state
        self.state = np.random.rand(1)
        return self.state, {}

    def step(self, action):
        # Execute one time step within the environment
        if action == 1:
            self.state = np.random.rand(1)  # Example: random new state
            reward = 1.0  # Example: fixed reward
        else:
            self.state = np.random.rand(1)
            reward = 0.0  # Example: fixed reward
        done = np.random.rand() > 0.95  # Example: 5% chance to end episode
        return self.state, reward, done, {}, {}

    def render(self):
        # Render the environment (optional)
        pass

# Register the environment with Gymnasium
gym.register(
    id='SimpleEnv-v0',
    entry_point='__main__:SimpleEnv',
)

from ray import tune

tune.run("PPO",
         config={"env": SimpleEnv,    # Instead of strings e.g. "CartPole-v1", we pass the custom env class
                 "evaluation_interval": 1000,
                 # Each episode uses different shop params. Need lots of samples to gauge agent's performance
                 "evaluation_duration_unit": 10000,
                 },
         checkpoint_freq=1000,
         )