# Author: Seungjae Shin <sjshin0505@{etri.re.kr, gmail.com}>

"""Gym-type environment for simulating Jack's car rental example. It is implemented based on the following reference:
R. S. Sutton and A. G. Barto, Reinforcement Learning - An Introduction, 2nd ed., 2018 (Example 4.2: Jack's Car Rental).
"""

import gym
import numpy as np

from examples.car_rental.actual_env import CarRentalActualEnv
from gym.spaces import Box, Discrete, Tuple
from gymproxy import BaseEnv


class CarRental(BaseEnv):
    """Class defining observation and action spaces of gym-type CarRental environment.
    """
    def __init__(self, **kwargs):
        """Constructor.

        :param kwargs: Dictionary of keyword arguments.
        """
        BaseEnv.actual_env_class = CarRentalActualEnv
        super().__init__(**kwargs)

    @staticmethod
    def build_obs_space(**kwargs) -> gym.Space:
        """Builds observation space.

        :param kwargs: Dictionary of keyword arguments.
        :return: Observation space.
        """
        config = kwargs['config']
        max_num_cars_per_loc = config['max_num_cars_per_loc']
        return Box(low=0, high=max_num_cars_per_loc, shape=(2,), dtype=np.int)

    @staticmethod
    def build_action_space(**kwargs) -> gym.Space:
        """Builds action space.

        :param kwargs: Dictionary of keyword arguments.
        :return: Action space.
        """
        config = kwargs['config']
        max_num_cars_per_loc = config['max_num_cars_per_loc']
        return Tuple((Discrete(2), Box(low=0, high=max_num_cars_per_loc, shape=(1,), dtype=np.int)))
