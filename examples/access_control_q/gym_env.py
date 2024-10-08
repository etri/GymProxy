# Author: Sae Hyong Park <labry@etri.re.kr>, Seungjae Shin <sjshin0505@{etri.re.kr, gmail.com}>

"""Gym-type environment for simulating an access control example in a simple queuing system. It is implemented based on the following reference:
R. S. Sutton and A. G. Barto, Reinforcement Learning - An Introduction, 2nd ed., 2018 (Example 10.2: An Access-Control Queuing Task).
"""
import logging
from typing import Optional

import gymnasium as gym
import numpy as np

from examples.access_control_q.actual_env import AccessControlQueueActualEnv
from gymnasium.spaces import Box, Discrete
from gymproxy import GymEnv

logger = logging.getLogger('access-control-q')

class AccessControlQueue(GymEnv):
    """Class defining observation and action spaces of gym-type AccessControlQueue environment.
    """
    def __init__(self, config: Optional[dict] = None):
        """Constructor.

        :param kwargs: Dictionary of keyword arguments.
        """
        NUM_STEPS = 100
        NUM_SERVERS = 10
        SERVER_FREE_PROB = 0.06
        PRIORITIES = [1., 2., 4., 8.]
        if config is None:
            logger.info("config is None.")
            config = {'num_steps': NUM_STEPS,
                      'num_servers': NUM_SERVERS,
                      'server_free_probability': SERVER_FREE_PROB,
                      'priorities': PRIORITIES}

        GymEnv.actual_env_class = AccessControlQueueActualEnv
        super().__init__(config)

    @staticmethod
    def build_obs_space(kwargs: Optional[dict] = None) -> gym.Space:
        """Builds observation space.

        :param kwargs: Dictionary of keyword arguments.
        :return: Observation space.
        """
        return Box(low=0, high=np.inf, shape=(2,), dtype=np.int32)

    @staticmethod
    def build_action_space(kwargs: Optional[dict] = None) -> gym.Space:
        """Builds action space.

        :param kwargs: Dictionary of keyword arguments.
        :return: Action space.
        """
        return Discrete(2)
