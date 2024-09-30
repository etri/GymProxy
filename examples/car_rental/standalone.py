# Author: Sae Hyong Park <labry@etri.re.kr>, Seungjae Shin <sjshin0505@{etri.re.kr, gmail.com}>

"""A standalone simulator for simulating the following reference:
R. S. Sutton and A. G. Barto, Reinforcement Learning - An Introduction, 2nd ed., 2018 (Example 4.2: Jack's car rental).
"""

import logging
import numpy as np

from gymnasium.spaces import Box, Discrete, Tuple
from examples.utilities import get_step_log_str_for_standalone

# Setting logger
FORMAT = "[%(asctime)s|%(levelname)s|%(name)s] %(message)s"
DATE_FMT = "%H:%M:%S %Y-%m-%d"
log_level = logging.INFO
logging.basicConfig(format=FORMAT, datefmt=DATE_FMT, level=log_level)
logger = logging.getLogger('main')


class CarRentalSimulator:
    """Standalone Simulator for Jack's car rental.
    """

    def __init__(self,
                 num_steps,
                 max_num_cars_per_loc,
                 rental_credit,
                 car_moving_cost,
                 rental_rate_0,
                 rental_rate_1,
                 return_rate_0,
                 return_rate_1):
        """Constructor.

        Args:
            num_steps (int): Number of time-steps.
            max_num_cars_per_loc (int): Maximum number of cars per location.
            rental_credit (float): Credit for each rental.
            car_moving_cost (float): Cost for each car moving between the two locations.
            rental_rate_0 (int): Mean arrival rate of rental request at the location 0 for each time-steps.
            rental_rate_1 (int): Mean arrival rate of rental request at the location 1 for each time-steps.
            return_rate_0 (int): Mean arrival rate of return at the location 0 for each time-steps.
            return_rate_1 (int): Mean arrival rate of return at the location 1 for each time-steps.
        """
        self._num_steps = num_steps
        self._max_num_cars_per_loc = max_num_cars_per_loc
        self._rental_credit = rental_credit
        self._car_moving_cost = car_moving_cost
        self._lambda_rental_0 = rental_rate_0
        self._lambda_rental_1 = rental_rate_1
        self._lambda_return_0 = return_rate_0
        self._lambda_return_1 = return_rate_1
        self._available_cars = [self._max_num_cars_per_loc] * 2
        self._action_space = Tuple((Discrete(2), Box(low=0, high=MAX_NUM_CARS_PER_LOC, shape=(1,), dtype=np.int_)))
        self._reward = 0.

    def start(self, seed_: int):
        """Start Jack's car rental simulation.

        Args:
            seed_: Seed for pseudo random number generator used by simulation.
        """
        t = 0
        np.random.seed(seed_)

        while t < self._num_steps:
            msg = None
            is_lost = False

            # New rental requests arrive at two locations for car rental. The arrival rates follow Poisson distribution.
            n_req_0, n_req_1 = self._get_new_rental_requests()

            # Identifies the number of possible rentals and reward.
            n_rentals = min(self._available_cars[0], n_req_0) + min(self._available_cars[1], n_req_1)
            self._reward = n_rentals * self._rental_credit

            # Checks if there is out of car for the two locations
            if self._available_cars[0] < n_req_0:
                msg = 'The business is lost because no available car at location 0'
                is_lost = False
            if self._available_cars[1] < n_req_1:
                if self._available_cars[0] < n_req_0:
                    msg += ' and location 1'
                else:
                    msg = 'The business is lost because no available car at location 1'
                    is_lost = False
            if is_lost:
                msg += '.'

            self._rent_cars(n_req_0, n_req_1)   # Rents cars for requests at each location.
            obs = self._make_obs()  # Observation consists of the numbers of available cars at the two locations.

            # Checks if the business is lost.
            if is_lost:
                continue

            # Action consists of source location, from which cars should be moved, and number of cars to be moved.
            action = self._policy(obs, self._reward)
            src = action[0]
            dst = 1 - src
            n_moving = action[1].item()

            step_log_str = get_step_log_str_for_standalone(t, obs, self._reward, action)
            logger.info(step_log_str)

            # Some rented cars are returned. The return rates follow Poisson distribution.
            # Note that the number of available cars at each location should not be exceeded _max_num_cars_per_loc.
            n_return_0, n_return_1 = self._return_cars()

            # Moves cars from the source location to the destination location.
            # Note that the number of available cars at each location should not be exceeded _max_num_cars_per_loc.
            self._move_cars(src, dst, n_moving)

            t += 1

    def _get_new_rental_requests(self) -> (int, int):
        n_req_0 = np.random.poisson(self._lambda_rental_0)
        n_req_1 = np.random.poisson(self._lambda_rental_1)
        return n_req_0, n_req_1

    def _rent_cars(self, n_req_0, n_req_1):
        self._available_cars[0] = max(self._available_cars[0] - n_req_0, 0)
        self._available_cars[1] = max(self._available_cars[1] - n_req_1, 0)

    def _return_cars(self) -> (int, int):
        n_return_0 = np.random.poisson(self._lambda_return_0)
        n_return_1 = np.random.poisson(self._lambda_return_1)
        self._available_cars[0] = min(self._available_cars[0] + n_return_0, self._max_num_cars_per_loc)
        self._available_cars[1] = min(self._available_cars[1] + n_return_1, self._max_num_cars_per_loc)
        return n_return_0, n_return_1

    def _move_cars(self, src, dst, n_moving):
        self._available_cars[src] = max(self._available_cars[src] - n_moving, 0)
        self._available_cars[dst] = min(self._available_cars[dst] + n_moving, self._max_num_cars_per_loc)

    def _make_obs(self) -> object:
        return np.array(self._available_cars, dtype=np.int32)

    def _policy(self, obs, reward):
        return self._action_space.sample()


# Environment configuration parameters.
NUM_STEPS = 100
MAX_NUM_CARS_PER_LOC = 20
RENTAL_CREDIT = 10.
CAR_MOVING_COST = 2.
RENTAL_RATE_0 = 4
RENTAL_RATE_1 = 3
RETURN_RATE_0 = 3
RETURN_RATE_1 = 2

NUM_EPISODES = 1
SEED = 2024

def main():
    simulator = CarRentalSimulator(NUM_STEPS,
                                   MAX_NUM_CARS_PER_LOC,
                                   RENTAL_CREDIT,
                                   CAR_MOVING_COST,
                                   RENTAL_RATE_0,
                                   RENTAL_RATE_1,
                                   RETURN_RATE_0,
                                   RETURN_RATE_1)
    simulator.start(SEED)

if __name__ == "__main__":
    main()
