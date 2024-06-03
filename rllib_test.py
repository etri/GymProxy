from typing import Optional

import gymnasium
from gymnasium.spaces import Box
import numpy as np
from gymnasium.utils.env_checker import check_env
from numpy.random import default_rng
from ray.tune import register_env


class InventoryEnv(gymnasium.Env):
    def __init__(self, config: Optional[dict] = None):
        """
        Must define self.observation_space and self.action_space here
        """

        # Define action space: bounds, space type, shape

        # Bound: Shelf space is limited
        print("config:", config)
        # print("render_mode:", render_mode)
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
        # self.render_mode = render_mode

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


# env = gymnasium.make(id='InventoryEnv-v0')
# print("env", env)
# obs, info = env.reset(seed=126, options={})

import logging
import numpy as np

from examples.gamblers_problem import *

# Setting logger
FORMAT = "[%(asctime)s|%(levelname)s|%(name)s] %(message)s"
DATE_FMT = "%H:%M:%S %Y-%m-%d"
log_level = logging.INFO
logging.basicConfig(format=FORMAT, datefmt=DATE_FMT, level=log_level)
logger = logging.getLogger('main')

# Environment configuration parameters.
NUM_STEPS = 10
PROB_HEAD = 0.5
INITIAL_CAPITAL = 10
WINNING_CAPITAL = 100

NUM_EPISODES = 1

def main():
    """Main routine of testing GamblersProblem gym-type environment.
    """
    gymnasium.register(id='InventoryEnv-v0', entry_point='rllib_test:InventoryEnv')
    register_env("InventoryEnv-v0", InventoryEnv)

    config = {'num_steps': NUM_STEPS,
              'prob_head': PROB_HEAD,
              'initial_capital': INITIAL_CAPITAL,
              'winning_capital': WINNING_CAPITAL}

    # metadata_ = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    env = gym.make(id='InventoryEnv-v0')
    #print("env", env)


    for i in range(0, NUM_EPISODES):
        obs, info = env.reset(seed=i, options={})
        log_step(0, 0, obs, 0.0, False, False, info, {})
        # print(obs)
        capital = INITIAL_CAPITAL
        print("1")
        j = 0
        # obs, info = env.reset(seed=126, options={})
        #print(obs, info)
        #logger.info(str(obs))
        while True:
            env.render()
            # action = env.action_space.sample()  # Means random agent
            action = [100.0]
            # Amount of betting should be less than current capital.
            # action = min(obs[0].item(), WINNING_CAPITAL - capital)
            #print(action, obs[0].item(), WINNING_CAPITAL-capital)

            obs, reward, terminated, truncated, info = env.step(action)

            # if info["flip_result"] == "head":
            #     capital += action
            # else:
            #     capital -= action

            log_step(i, j, obs, reward, terminated, truncated, info, action)
            j = j + 1
            if terminated:
                break
    env.close()

def log_step(episode: int, step: int, obs: np.ndarray, reward: float, terminated: bool, truncated:bool, info: dict, action: int):
    """Utility function for printing logs.

    :param episode: Episode number.
    :param step: Time-step.
    :param obs: Observation of the current environment.
    :param reward: Reward from the current environment.
    :param done: Indicates whether the episode ends or not.
    :param info: Contains auxiliary diagnostic information (helpful for debugging, and sometimes learning).
    :param action: An action provided by the agent.
    """
    capital = obs[0].item()
    if action != 0 :
        bet = action
    else:
        bet = 0
    step_str = '{}-th step in {}-th episode / '.format(step, episode)
    obs_str = 'obs: {} / '.format(capital)
    reward_str = 'reward: {} / '.format(reward)
    done_str = 'terminated: {} / '.format(terminated)
    truncated_str = 'truncated: {} / '.format(truncated)
    info_str = 'info: {} / '.format(info)
    action_str = 'action: {}'.format(bet)
    result_str = step_str + obs_str + reward_str + done_str + truncated_str + info_str + action_str
    logger.info(result_str)

if __name__ == "__main__":
    main()


# from ray import tune
#
# NUM_STEPS = 100
# PROB_HEAD = 0.5
# INITIAL_CAPITAL = 10
# WINNING_CAPITAL = 100
#
# config = {'num_steps': NUM_STEPS,
#           'prob_head': PROB_HEAD,
#           'initial_capital': INITIAL_CAPITAL,
#           'winning_capital': WINNING_CAPITAL}
#
# tune.run("PPO",
#          config={"env": "InventoryEnv-v0",    # Instead of strings e.g. "CartPole-v1", we pass the custom env class
#                  "env_config": {"config":config},
#                  "evaluation_interval": 1000,
#                  # Each episode uses different shop params. Need lots of samples to gauge agent's performance
#                  "evaluation_duration_unit": 10000,
#                  },
#          checkpoint_freq=10,
#          )

