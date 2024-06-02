# Author: Seungjae Shin <sjshin0505@{etri.re.kr, gmail.com}>

"""Gym-type environment for simulating gambler's problem example. It is implemented based on the following reference:
R. S. Sutton and A. G. Barto, Reinforcement Learning - An Introduction, 2nd ed., 2018 (Example 4.3: Gambler's Problem).
"""

import gymnasium as gym
import numpy as np

from examples.gamblers_problem.actual_env import GamblersProblemActualEnv
from gymnasium.spaces import Box, Discrete
from gymproxy import BaseEnv
from typing import Optional

# Environment configuration parameters.
NUM_STEPS = 100
PROB_HEAD = 0.5
INITIAL_CAPITAL = 10
WINNING_CAPITAL = 100

class GamblersProblem(BaseEnv):
    """Class defining observation and action spaces of gym-type GamblersProblem environment.
    """
    def __init__(self, kwargs: Optional[dict] = None):
        """Constructor.

        :param kwargs: Dictionary of keyword arguments.
        """
        #labry debug
        if kwargs is None:
            kwargs = {'num_steps': NUM_STEPS,
                      'prob_head': PROB_HEAD,
                      'initial_capital': INITIAL_CAPITAL,
                      'winning_capital': WINNING_CAPITAL}

        print('GamblersProblem __init__ kwargs: {}'.format(kwargs))
        BaseEnv.actual_env_class = GamblersProblemActualEnv
        super().__init__(kwargs)

    @staticmethod
    def build_obs_space(kwargs: Optional[dict] = None) -> gym.Space:
        """Builds observation space.

        :param kwargs: Dictionary of keyword arguments.
        :return: Observation space.
        """
        config = kwargs
        s_win = config['winning_capital']
        return Box(low=0, high=s_win, shape=(1,), dtype=np.int_)

    @staticmethod
    def build_action_space(kwargs: Optional[dict] = None) -> gym.Space:
        """Builds action space.

        :param kwargs: Dictionary of keyword arguments.
        :return: Action space.
        """
        config = kwargs
        s_win = config['winning_capital']
        return Box(low=1., high=s_win, shape=(1,), dtype=np.int_)
