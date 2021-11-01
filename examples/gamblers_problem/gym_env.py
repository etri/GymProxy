# Author: Seungjae Shin <sjshin0505@{etri.re.kr, gmail.com}>

"""Gym-type environment for simulating gambler's problem example. It is implemented based on the following reference:
R. S. Sutton and A. G. Barto, Reinforcement Learning - An Introduction, 2nd ed., 2018 (Example 4.3: Gambler's Problem).
"""

import gym
import numpy as np

from examples.gamblers_problem.actual_env import GamblersProblemActualEnv
from gym.spaces import Box, Discrete
from gymproxy import BaseEnv


class GamblersProblem(BaseEnv):
    """Class defining observation and action spaces of gym-type GamblersProblem environment.
    """
    def __init__(self, **kwargs):
        """Constructor.

        :param kwargs: Dictionary of keyword arguments.
        """
        BaseEnv.actual_env_class = GamblersProblemActualEnv
        super().__init__(**kwargs)

    @staticmethod
    def build_obs_space(**kwargs) -> gym.Space:
        """Builds observation space.

        :param kwargs: Dictionary of keyword arguments.
        :return: Observation space.
        """
        config = kwargs['config']
        s_win = config['winning_capital']
        return Box(low=0, high=s_win, shape=(1,), dtype=np.int)

    @staticmethod
    def build_action_space(**kwargs) -> gym.Space:
        """Builds action space.

        :param kwargs: Dictionary of keyword arguments.
        :return: Action space.
        """
        config = kwargs['config']
        s_win = config['winning_capital']
        return Box(low=1., high=s_win, shape=(1,), dtype=np.int)
