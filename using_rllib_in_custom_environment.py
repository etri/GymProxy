from typing import Optional

import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np
from gymnasium.utils.env_checker import check_env
from numpy.random import default_rng


class InventoryEnv(gym.Env):
    def __init__(self, render_mode: Optional[str] = None):
        """
        Must define self.observation_space and self.action_space here
        """

        # Define action space: bounds, space type, shape

        # Bound: Shelf space is limited
        print("render_mode:", render_mode)
        self.max_capacity = 4000

        # Space type: Better to use Box than Discrete, since Discrete will lead to too many output nodes in the NN
        # Shape: rllib cannot handle scalar actions, so turn it into a numpy array with shape (1,)
        self.action_space = Box(low=np.array([0]), high=np.array([self.max_capacity]))

        # Define observation space: bounds, space type, shape

        # Shape: The lead time controls the shape of observation space
        self.lead_time = 5
        self.obs_dim = self.lead_time + 4

        # Bounds: Define high of the remaining observation space elements
        self.max_mean_daily_demand = 200
        self.max_unit_selling_price = 100
        self.max_daily_holding_cost_per_unit = 5

        obs_low = np.zeros((self.obs_dim,))
        obs_high = np.array([self.max_capacity for _ in range(self.lead_time)] +
                            [self.max_mean_daily_demand, self.max_unit_selling_price,
                             self.max_unit_selling_price, self.max_daily_holding_cost_per_unit
                             ]
                            )
        self.observation_space = Box(low=obs_low, high=obs_high)
        # config["observation_space"] = self.observation_space

        # The random number generator that will be used throughout the environment
        self.rng = default_rng()

        # All instance variables are defined in the __init__() method
        self.current_obs = None
        self.episode_length_in_days = 90
        self.day_num = None
        self.render_mode = render_mode

    def reset(self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,)-> (object, dict):
        """
        Returns: the observation of the initial state
        Reset the environment to initial state so that a new episode (independent of previous ones) may start
        """
        super().reset(seed=seed)

        # Sample parameter values from the parameter space

        # Set mean daily demand (lambda)
        mean_daily_demand = self.rng.uniform() * self.max_mean_daily_demand

        # Set selling price
        selling_price = self.rng.uniform() * self.max_unit_selling_price

        # Set buying price: buying price cannot be higher than selling price
        buying_price = self.rng.uniform() * selling_price

        # Set daily holding cose per unit: holding cost cannot be higher than buying_price
        daily_holding_cost_per_unit = self.rng.uniform() * min(buying_price,
                                                               self.max_daily_holding_cost_per_unit
                                                               )

        # Return the first observation
        self.current_obs = np.array([0 for _ in range(self.lead_time)] +
                                    [mean_daily_demand, selling_price, buying_price,
                                     daily_holding_cost_per_unit,
                                     ]
                                    )
        # self.current_obs = np.zeros()

        print(self.current_obs)
        self.day_num = 0
        return self.current_obs, {}

    def step(self, action):
        """
        Returns: Given current obs and action, returns the next observation, the reward, done and optionally additional info
        """
        # Action looks like np.array([20.0]). We convert that to float 20.0 for easier calculation
        buys = min(action[0], self.max_capacity - np.sum(self.current_obs[:self.lead_time]))

        # Compute next obs
        demand = self.rng.poisson(self.current_obs[self.lead_time])
        next_obs = np.concatenate((self.current_obs[1: self.lead_time],
                                   np.array([buys]),
                                   self.current_obs[self.lead_time:]
                                   )
                                  )
        next_obs[0] += max(0, self.current_obs[0] - demand)

        # Compute reward
        reward = (self.current_obs[self.lead_time + 1] * (self.current_obs[0] + self.current_obs[1] - next_obs[0]) -
                  self.current_obs[self.lead_time + 2] * buys -
                  self.current_obs[self.lead_time + 3] * (next_obs[0] - self.current_obs[1])
                  )

        # Compute done
        self.day_num += 1
        done = False
        truncated = False
        if self.day_num >= self.episode_length_in_days:
            done = True

        self.current_obs = next_obs

        # info must be a dict
        return self.current_obs, reward, done, truncated, {}

    def render(self, mode="human"):
        """
        Returns: None
        Show the current environment state e.g. the graphical window in `CartPole-v1`
        This method must be implemented, but it is OK to have an empty implementation if rendering is not
        important
        """
        pass

    def close(self):
        """
        Returns: None
        This method is optional. Used to cleanup all resources (threads, graphical windows) etc.
        """
        pass

    def seed(self, seed=None):
        """
        Returns: List of seeds
        This method is optional. Used to set seeds for the environment's random number generator for
        obtaining deterministic behavior
        """
        return


# env = InventoryEnv()
# check_env(env, warn=False)


from ray import tune

tune.run("PPO",
         config={"env": InventoryEnv,    # Instead of strings e.g. "CartPole-v1", we pass the custom env class
                 "evaluation_interval": 1000,
                 # Each episode uses different shop params. Need lots of samples to gauge agent's performance
                 "evaluation_duration_unit": 10000,
                 },
         checkpoint_freq=10,
         )