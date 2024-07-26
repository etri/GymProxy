# Author: Sae Hyong Park <labry@etri.re.kr>
# Author: Seungjae Shin <sjshin0505@{etri.re.kr, gmail.com}>

"""External environment for actually simulating Jack's car rental example. It is implemented based on the following
reference:
R. S. Sutton and A. G. Barto, Reinforcement Learning - An Introduction, 2nd ed., 2018 (Example 4.2: Jack's Car Rental).
"""

import logging
from typing import Optional

import numpy as np

from gymproxy.base_actual_env_cy import BaseActualEnv, TerminateGymProxy

logger = logging.getLogger('car_rental_actual_env')


class CarRentalActualEnv(BaseActualEnv):
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
        BaseActualEnv.__init__(self, env_proxy)
        config = kwargs.get('config')
        self._num_steps = kwargs['num_steps']
        self._max_num_cars_per_loc = kwargs['max_num_cars_per_loc']
        self._rental_credit = kwargs['rental_credit']
        self._car_moving_cost = kwargs['car_moving_cost']
        self._lambda_rental_0 = kwargs['rental_rate_0']
        self._lambda_rental_1 = kwargs['rental_rate_1']
        self._lambda_return_0 = kwargs['return_rate_0']
        self._lambda_return_1 = kwargs['return_rate_1']
        self._available_cars = [self._max_num_cars_per_loc] * 2
        self._reward = 0.
        self._t = 0

    def run(self, seed_:int, kwargs: Optional[dict] = None):
        """Runs Jack's car rental environment.

        :param kwargs: Dictionary of keyword arguments.
        """
        try:
            obs = None
            terminated = False
            truncated = False
            info = {}
            np.random.seed(seed_)
            self._reward = 0.
            self._t = 0
            self._available_cars = [self._max_num_cars_per_loc] * 2

            while self._t < self._num_steps and not terminated:

                if self._t != 0:
                    action = CarRentalActualEnv.get_action(obs, self._reward, terminated, truncated, info)
                else:
                    action = (0, np.array([0]))

                msg = None

                # New rental requests arrive at two locations for car rental. The arrival rates follow Poisson
                # distribution.
                n_req_0 = np.random.poisson(self._lambda_rental_0)
                n_req_1 = np.random.poisson(self._lambda_rental_1)

                # Identifies the number of possible rentals and reward.
                n_rentals = min(self._available_cars[0], n_req_0) + min(self._available_cars[1], n_req_1)
                self._reward = n_rentals * self._rental_credit * 0.005

                # Checks if there is out of car for the two locations
                if self._available_cars[0] < n_req_0:
                    msg = 'The business is lost because no available car at location 0'
                    self._reward = 0
                    terminated = True
                    truncated = True
                if self._available_cars[1] < n_req_1:
                    self._reward = 0
                    if terminated:
                        msg += ' and location 1'
                    else:
                        msg = 'The business is lost because no available car at location 1'
                        terminated = True
                        truncated = True
                if terminated:
                    msg += '.'

                self._available_cars[0] = max(self._available_cars[0] - n_req_0, 0)
                self._available_cars[1] = max(self._available_cars[1] - n_req_1, 0)

                # A number of notes in the information dictionary.
                info['rental_requests'] = [n_req_0, n_req_1]
                if msg:
                    info['msg'] = msg
                elif 'msg' in info.keys():
                    del info['msg']

                # Checks if the business is lost.
                if terminated:
                    continue

                # Some rented cars are returned. The return rates follow Poisson distribution. Note that the number of
                # available cars at each location should not be exceed _max_num_cars_per_loc.
                # np.random.seed(1)
                n_return_0 = np.random.poisson(self._lambda_return_0)
                n_return_1 = np.random.poisson(self._lambda_return_1)
                self._available_cars[0] = min(self._available_cars[0] + n_return_0, self._max_num_cars_per_loc)
                self._available_cars[1] = min(self._available_cars[1] + n_return_1, self._max_num_cars_per_loc)

                info['returns'] = [n_return_0, n_return_1]  # Note returns in the information dictionary.

                # Observation consists of the numbers of available cars at the two locations.

                obs = np.array(self._available_cars, dtype=np.int32)

                # logger.info("self._available_cars[0] {} - n_moving {}".format(self._available_cars[0], n_moving))
                # logger.info("self._available_cars[1] {} - n_moving {}".format(self._available_cars[1], n_moving))
                # Action consists of source location, from which cars should be moved, and number of cars to be moved.
                # where action takes place
                # print("action:", action)
                # print("type:", type(action))
                src = action[0]
                dst = 1 - src
                n_moving = action[1].item()
                # Moves cars from the source location to the destination location. Note that the number of available
                # cars at each location should not be exceed _max_num_cars_per_loc.
                self._available_cars[src] = max(self._available_cars[src] - n_moving, 0)
                self._available_cars[dst] = min(self._available_cars[dst] + n_moving, self._max_num_cars_per_loc)

                self._reward = self._reward - (n_moving * 0.005 * 2)


                self._t += 1

            # Arrives to the end of the episode (terminal state).
            terminated = True
            truncated = True
            CarRentalActualEnv.set_obs_and_reward(obs, self._reward, terminated, truncated, info)

        # Exception handling block.
        except TerminateGymProxy:
            # Means termination signal triggered by the agent.
            logger.info('Terminating CarRental environment.')
            BaseActualEnv.env_proxy.release_lock()
            BaseActualEnv.env_proxy.set_gym_env_event()
            exit(1)
        except Exception as e:
            logger.exception(e)
            BaseActualEnv.env_proxy.release_lock()
            BaseActualEnv.env_proxy.set_gym_env_event()
            exit(1)

    def finish(self, kwargs: Optional[dict] = None):
        """Finishes Jack's car rental environment.

        :param kwargs: Dictionary of keyword arguments.
        """
        logger.info("finish")
        return
