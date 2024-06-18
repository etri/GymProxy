# Author: Seungjae Shin <sjshin0505@{etri.re.kr, gmail.com}>

"""Gym-type environment for simulating gambler's problem example. It is implemented based on the following reference:
R. S. Sutton and A. G. Barto, Reinforcement Learning - An Introduction, 2nd ed., 2018 (Example 4.3: Gambler's Problem).
"""
import logging
import sys
from typing import Optional

import gymnasium
import numpy as np
from gymnasium.spaces import Box

from examples.gamblers_problem.actual_env import GamblersProblemActualEnv
from gymproxy import BaseEnv

logger = logging.getLogger('gamblers_problem')

class GamblersProblem(BaseEnv):
    """Class defining observation and action spaces of gym-type GamblersProblem environment.
    """
    def __init__(self, config: Optional[dict] = None):
        """Constructor.

        :param kwargs: Dictionary of keyword arguments.
        """
        # Environment configuration parameters.
        NUM_STEPS = 100
        PROB_HEAD = 0.6
        INITIAL_CAPITAL = 10
        WINNING_CAPITAL = 100

        from examples.gamblers_problem.gym_env import GamblersProblem
        # GamblersProblem.update_action_space(self, obs=INITIAL_CAPITAL)

        if config is None:
            logger.info("config is None.")
            config = {'num_steps': NUM_STEPS,
                      'prob_head': PROB_HEAD,
                      'initial_capital': INITIAL_CAPITAL,
                      'winning_capital': WINNING_CAPITAL}

        BaseEnv.actual_env_class = GamblersProblemActualEnv
        super().__init__(config)

    @staticmethod
    def build_obs_space(kwargs: Optional[dict] = None) -> gymnasium.Space:
        """Builds observation space.

        :param kwargs: Dictionary of keyword arguments.
        :return: Observation space.
        """
        config = kwargs
        # print("winning:", config['winning_capital'])
        s_win = config.get('winning_capital', 100)
        result = Box(low=0, high=100, shape=(1,), dtype=np.int_)
        #print(result)
        return result

    @staticmethod
    def build_action_space(kwargs: Optional[dict] = None) -> gymnasium.Space:
        """Builds action space.

        :param kwargs: Dictionary of keyword arguments.
        :return: Action space.
        """
        config = kwargs
        s_win = config.get('winning_capital', 100)
        # result = Box(low=1., high=s_win, shape=(1,), dtype=np.int_)
        result = Box(low=0., high=1.0, shape=(1,), dtype=np.float_)
        #print(result)
        return result

    @staticmethod
    def update_action_space(self, obs: int):
        """Builds action space.

        :param kwargs: Dictionary of keyword arguments for building action space.
        """
        # obs = min(obs, 100 - obs)
        # obs = max(obs, 2)
        # self.action_space = Box(low=1., high=obs, shape=(1,), dtype=np.int_)
        logger.info("update action space {}".format(obs))
        # print("update action space", obs)

    def sample_action(self, cash_in_hand):
        cash_in_hand = max(cash_in_hand, 2)
        my_action = np.random.randint(1, cash_in_hand)
        # logger.info("my_action: {}".format(my_action))
        return my_action


# from ray import tune
#
# from examples.gamblers_problem import *
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
#          config={"env": GamblersProblem,    # Instead of strings e.g. "CartPole-v1", we pass the custom env class
#                  "env_config": {"config":config},
#                  "num_env_runners": 1,
#                  "evaluation_interval": 1000,
#                  # Each episode uses different shop params. Need lots of samples to gauge agent's performance
#                  "evaluation_duration_unit": 10000,
#                  },
#          checkpoint_freq=2,
#          num_samples=1,
#          )