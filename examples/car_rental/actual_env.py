# Author: Sae Hyong Park <labry@etri.re.kr>, Seungjae Shin <sjshin0505@{etri.re.kr, gmail.com}>

"""External environment for actually simulating Jack's car rental example. It is implemented based on the following
reference:
R. S. Sutton and A. G. Barto, Reinforcement Learning - An Introduction, 2nd ed., 2018 (Example 4.2: Jack's Car Rental).
"""

import copy
import logging
from typing import Optional

import numpy as np

from gymproxy.actual_env import ActualEnv, TerminateGymProxy

logger = logging.getLogger('car_rental_actual_env')


class CarRentalActualEnv(ActualEnv):
    """External environment class that actually simulates Jack's car rental.
    """

    def __init__(self, kwargs: Optional[dict] = None):
        """Constructor.

        :param kwargs: Dictionary of keyword arguments. It should have 'config' argument that is a dictionary for
        setting configuration parameters. kwargs['config'] should define following keys:
            num_steps (int): Number of time-steps.
            max_num_cars_per_loc (int): Maximum number of cars per location.
            rental_credit (float): Credit for each rental.
            car_moving_cost (float): Cost for each car moving between the two locations.
            rental_rate_0 (int): Mean arrival rate of rental request at the location 0 for each time-steps.
            rental_rate_1 (int): Mean arrival rate of rental request at the location 1 for each time-steps.
            return_rate_0 (int): Mean arrival rate of return at the location 0 for each time-steps.
            return_rate_1 (int): Mean arrival rate of return at the location 1 for each time-steps.
        """
        env_proxy = kwargs.get('env_proxy')
        ActualEnv.__init__(self, env_proxy)
        
        self._num_steps = kwargs['num_steps']
        self._max_num_cars_per_loc = kwargs['max_num_cars_per_loc']
        self._rental_credit = kwargs['rental_credit']
        self._car_moving_cost = kwargs['car_moving_cost']
        self._lambda_rental_0 = kwargs['rental_rate_0']
        self._lambda_rental_1 = kwargs['rental_rate_1']
        self._lambda_return_0 = kwargs['return_rate_0']
        self._lambda_return_1 = kwargs['return_rate_1']
        
        self._initial_cars = np.array([self._max_num_cars_per_loc] * 2)
        self._available_cars = np.copy(self._initial_cars)
        self._reward = 0.
        self._t = 0

    def run(self, seed_: int, kwargs: Optional[dict] = None):
        """Runs Jack's car rental environment.

        :param seed_: Seed for random number generator.
        :param kwargs: Dictionary of keyword arguments.
        """
        try:
            self._available_cars = np.copy(self._initial_cars)
            self._reward = 0.
            self._t = 0
            
            info = {}
            terminated = False
            truncated = False
            
            np.random.seed(seed_)
            
            while self._t < self._num_steps and not terminated:
                obs = np.array(self._available_cars, dtype=np.int64)
                
                # New rental requests arrive at two locations
                n_req_0, n_req_1 = self._get_new_rental_requests()
                
                # Calculate reward from rentals
                n_rentals = min(self._available_cars[0], n_req_0) + min(self._available_cars[1], n_req_1)
                self._reward = n_rentals * self._rental_credit
                
                # Rent cars
                self._rent_cars(n_req_0, n_req_1)
                
                # Get action from agent
                raw_action = CarRentalActualEnv.get_action(obs, self._reward, terminated, truncated, info)
                
                if raw_action is None:
                    ActualEnv.env_proxy.terminate_sync()
                    exit(1)
                
                # Process action (move cars)
                src = raw_action[0]
                dst = 1 - src
                n_moving = int(raw_action[1][0])
                
                # Apply moving cost
                self._reward -= n_moving * self._car_moving_cost
                
                # Return cars
                self._return_cars()
                
                # Move cars
                self._move_cars(src, dst, n_moving)
                
                self._t += 1
                
                if self._t >= self._num_steps:
                    terminated = True
                    truncated = True
            
            # Final observation
            obs = np.array(self._available_cars, dtype=np.int64)
            CarRentalActualEnv.set_obs_and_reward(obs, self._reward, terminated, truncated, info)
            
        except TerminateGymProxy:
            logger.info('Terminating car rental environment.')
            ActualEnv.env_proxy.terminate_sync()
            exit(1)
        except Exception as e:
            logger.exception(e)
            ActualEnv.env_proxy.terminate_sync()
            exit(1)

    def _get_new_rental_requests(self) -> tuple[int, int]:
        """Get new rental requests following Poisson distribution."""
        n_req_0 = np.random.poisson(self._lambda_rental_0)
        n_req_1 = np.random.poisson(self._lambda_rental_1)
        return n_req_0, n_req_1

    def _rent_cars(self, n_req_0: int, n_req_1: int):
        """Rent cars for requests at each location."""
        self._available_cars[0] = max(self._available_cars[0] - n_req_0, 0)
        self._available_cars[1] = max(self._available_cars[1] - n_req_1, 0)

    def _return_cars(self) -> tuple[int, int]:
        """Return cars following Poisson distribution."""
        n_return_0 = np.random.poisson(self._lambda_return_0)
        n_return_1 = np.random.poisson(self._lambda_return_1)
        self._available_cars[0] = min(self._available_cars[0] + n_return_0, self._max_num_cars_per_loc)
        self._available_cars[1] = min(self._available_cars[1] + n_return_1, self._max_num_cars_per_loc)
        return n_return_0, n_return_1

    def _move_cars(self, src: int, dst: int, n_moving: int):
        """Move cars from source to destination location."""
        actual_moving = min(n_moving, self._available_cars[src])
        self._available_cars[src] -= actual_moving
        self._available_cars[dst] = min(self._available_cars[dst] + actual_moving, self._max_num_cars_per_loc)

    def finish(self, kwargs: Optional[dict] = None):
        """Finishes car rental environment.

        :param kwargs: Dictionary of keyword arguments.
        """
        logger.debug("car rental finish")
        return