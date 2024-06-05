# Author: Seungjae Shin <sjshin0505@{etri.re.kr, gmail.com}>

"""Gym-type environment for simulating gambler's problem example. It is implemented based on the following reference:
R. S. Sutton and A. G. Barto, Reinforcement Learning - An Introduction, 2nd ed., 2018 (Example 4.3: Gambler's Problem).
"""
import sys
from typing import Optional

import gymnasium
import numpy as np
from gymnasium.spaces import Box

from examples.gamblers_problem.actual_env import GamblersProblemActualEnv
from gymproxy import BaseEnv


class GamblersProblem(BaseEnv):
    """Class defining observation and action spaces of gym-type GamblersProblem environment.
    """
    def __init__(self, kwargs: Optional[dict] = None):
        """Constructor.

        :param kwargs: Dictionary of keyword arguments.
        """
        # Environment configuration parameters.
        NUM_STEPS = 100
        PROB_HEAD = 0.5
        INITIAL_CAPITAL = 10
        WINNING_CAPITAL = 100

        if kwargs is None:
            kwargs = {'num_steps': NUM_STEPS,
                      'prob_head': PROB_HEAD,
                      'initial_capital': INITIAL_CAPITAL,
                      'winning_capital': WINNING_CAPITAL}

        BaseEnv.actual_env_class = GamblersProblemActualEnv
        super().__init__(kwargs)

    @staticmethod
    def build_obs_space(kwargs: Optional[dict] = None) -> gymnasium.Space:
        """Builds observation space.

        :param kwargs: Dictionary of keyword arguments.
        :return: Observation space.
        """
        config = kwargs
        # print("winning:", config['winning_capital'])
        s_win = config.get('winning_capital', 100)
        result = Box(low=0, high=s_win, shape=(1,), dtype=np.int_)
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
        result = Box(low=0., high=s_win, shape=(1,), dtype=np.int_)
        #print(result)
        return result


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