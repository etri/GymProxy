# Author: Seungjae Shin <sjshin0505@{etri.re.kr, gmail.com}>

"""Gym-type environment for simulating an access control example in a simple queuing system. It is implemented based on
the following reference:
R. S. Sutton and A. G. Barto, Reinforcement Learning - An Introduction, 2nd ed., 2018 (Example 10.2: An Access-Control
Queuing Task).
"""

import gym
import numpy as np

from examples.access_control_q.actual_env import AccessControlQueueActualEnv
from gym.spaces import Box, Discrete
from gymproxy import BaseEnv


class AccessControlQueue(BaseEnv):
    """Class defining observation and action spaces of gym-type AccessControlQueue environment.
    """
    def __init__(self, **kwargs):
        """Constructor.

        :param kwargs: Dictionary of keyword arguments.
        """
        BaseEnv.actual_env_class = AccessControlQueueActualEnv
        super().__init__(**kwargs)

    @staticmethod
    def build_obs_space(**kwargs) -> gym.Space:
        """Builds observation space.

        :param kwargs: Dictionary of keyword arguments.
        :return: Observation space.
        """
        return Box(low=0, high=np.inf, shape=(2,), dtype=np.int)

    @staticmethod
    def build_action_space(**kwargs) -> gym.Space:
        """Builds action space.

        :param kwargs: Dictionary of keyword arguments.
        :return: Action space.
        """
        return Discrete(2)
