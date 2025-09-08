# Author: Seungjae Shin <sjshin0505@{etri.re.kr, gmail.com}>

"""Gym-type environment for simulating Jack's car rental example. It is implemented based on the following reference:
R. S. Sutton and A. G. Barto, Reinforcement Learning - An Introduction, 2nd ed., 2018 (Example 4.2: Jack's Car Rental).
"""
import logging
from typing import Optional

import gymnasium
import numpy as np
from gymnasium.spaces import Box, Discrete, MultiDiscrete, Tuple

from examples.car_rental.actual_env import CarRentalActualEnv
from gymproxy import GymEnv

logger = logging.getLogger('car_rental')

class CarRental(GymEnv):
    """Class defining observation and action spaces of gym-type CarRental environment.
    """
    def __init__(self, config: Optional[dict] = None):
        """Constructor.

        :param kwargs: Dictionary of keyword arguments.
        """
        NUM_STEPS = 100
        MAX_NUM_CARS_PER_LOC = 20
        RENTAL_CREDIT = 10.
        CAR_MOVING_COST = 2.
        RENTAL_RATE_0 = 4
        RENTAL_RATE_1 = 3
        RETURN_RATE_0 = 3
        RETURN_RATE_1 = 2

        if config is None:
            logger.info("config is None.")
            config = {'num_steps': NUM_STEPS,
                      'max_num_cars_per_loc': MAX_NUM_CARS_PER_LOC,
                      'rental_credit': RENTAL_CREDIT,
                      'car_moving_cost': CAR_MOVING_COST,
                      'rental_rate_0': RENTAL_RATE_0,
                      'rental_rate_1': RENTAL_RATE_1,
                      'return_rate_0': RETURN_RATE_0,
                      'return_rate_1': RETURN_RATE_1}

        GymEnv.actual_env_class = CarRentalActualEnv
        super().__init__(config)

    @staticmethod
    def build_obs_space(kwargs: Optional[dict] = None) -> gymnasium.Space:
        """Builds observation space.

        :param kwargs: Dictionary of keyword arguments.
        :return: Observation space.
        """
        config = kwargs
        max_num_cars_per_loc = config.get('max_num_cars_per_loc', 20)
        return Box(low=0, high=max_num_cars_per_loc, shape=(2,), dtype=np.int_)

    @staticmethod
    def build_action_space(kwargs: Optional[dict] = None) -> gymnasium.Space:
        """Builds action space.

        :param kwargs: Dictionary of keyword arguments.
        :return: Action space.
        """
        config = kwargs
        # max_num_cars_per_loc = config.get('max_num_cars_per_loc')
        max_num_cars_for_move = 5
        return MultiDiscrete([2, max_num_cars_for_move + 1])
        # return Tuple((Discrete(2), Box(low=0, high=max_num_cars_for_move, shape=(1,), dtype=np.int_)))
