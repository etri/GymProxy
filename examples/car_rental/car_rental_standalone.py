# Author: Sae Hyong Park <labry@etri.re.kr>
# Author: Seungjae Shin <sjshin0505@{etri.re.kr, gmail.com}>

"""Testing script for CarRental environment implemented based on the following reference:
R. S. Sutton and A. G. Barto, Reinforcement Learning - An Introduction, 2nd ed., 2018 (Example 4.2: Jack's car rental).
"""

import logging
import numpy as np
from gymnasium.spaces import Box, Discrete, Tuple

from examples.car_rental import *

# Setting logger
FORMAT = "[%(asctime)s|%(levelname)s|%(name)s] %(message)s"
DATE_FMT = "%H:%M:%S %Y-%m-%d"
log_level = logging.INFO
logging.basicConfig(format=FORMAT, datefmt=DATE_FMT, level=log_level)
logger = logging.getLogger('main')


class CarRentalSimulator:

    def __init__(self, num_steps,max_num_cars_per_loc,rental_credit,car_moving_cost,
                 rental_rate_0,rental_rate_1,return_rate_0,return_rate_1 ):
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
        self._num_steps = num_steps
        self._max_num_cars_per_loc = max_num_cars_per_loc
        self._rental_credit = rental_credit
        self._car_moving_cost = car_moving_cost
        self._lambda_rental_0 = rental_rate_0
        self._lambda_rental_1 = rental_rate_1
        self._lambda_return_0 = return_rate_0
        self._lambda_return_1 = return_rate_1
        self._available_cars = [self._max_num_cars_per_loc] * 2
        self._reward = 0.
        self._t = 0

    def start(self, seed_:int):
        """Main routine of testing CarRental gym-type environment.
        """
        """Runs Jack's car rental environment.

        :param kwargs: Dictionary of keyword arguments.
        """
        obs = None
        terminated = False
        truncated = False
        info = {}
        np.random.seed(seed_)
        while self._t < self._num_steps and not terminated:
            msg = None

            # New rental requests arrive at two locations for car rental. The arrival rates follow Poisson
            # distribution.
            n_req_0 = np.random.poisson(self._lambda_rental_0)
            n_req_1 = np.random.poisson(self._lambda_rental_1)

            # Identifies the number of possible rentals and reward.
            n_rentals = min(self._available_cars[0], n_req_0) + min(self._available_cars[1], n_req_1)
            self._reward = n_rentals * self._rental_credit

            # Checks if there is out of car for the two locations
            if self._available_cars[0] < n_req_0:
                msg = 'The business is lost because no available car at location 0'
                terminated = True
                truncated = True
            if self._available_cars[1] < n_req_1:
                if terminated:
                    msg += ' and location 1'
                else:
                    msg = 'The business is lost because no available car at location 1'
                    terminated = True
                    truncated = True
            if terminated:
                msg += '.'


            logger.info("***** obs: {}".format(obs))
            logger.info("..... self._available_cars0: {} self._available_cars1 {}".format(self._available_cars[0], self._available_cars[1]))
            # Rents cars for requests at each location.
            self._available_cars[0] = max(self._available_cars[0] - n_req_0, 0)
            self._available_cars[1] = max(self._available_cars[1] - n_req_1, 0)

            # Observation consists of the numbers of available cars at the two locations.

            obs = np.array(self._available_cars, dtype=np.int32)

            logger.info("----- obs: {}".format(obs))

            # A number of notes in the information dictionary.
            info['rental_requests'] = [n_req_0, n_req_1]
            if msg:
                info['msg'] = msg
            elif 'msg' in info.keys():
                del info['msg']

            # Checks if the business is lost.
            if terminated:
                continue

            # Action consists of source location, from which cars should be moved, and number of cars to be moved.
            # action = CarRentalActualEnv.get_action(obs, self._reward, terminated, truncated, info)
            ready = Tuple((Discrete(2), Box(low=0, high=MAX_NUM_CARS_PER_LOC, shape=(1,), dtype=np.int_)))
            action = ready.sample()
            print("action:", action)
            src = action[0]
            dst = 1 - src
            n_moving = action[1].item()
            # n_moving = action[1]

            # Some rented cars are returned. The return rates follow Poisson distribution. Note that the number of
            # available cars at each location should not be exceed _max_num_cars_per_loc.
            n_return_0 = np.random.poisson(self._lambda_return_0)
            n_return_1 = np.random.poisson(self._lambda_return_1)
            self._available_cars[0] = min(self._available_cars[0] + n_return_0, self._max_num_cars_per_loc)
            self._available_cars[1] = min(self._available_cars[1] + n_return_1, self._max_num_cars_per_loc)

            # Moves cars from the source location to the destination location. Note that the number of available
            # cars at each location should not be exceed _max_num_cars_per_loc.
            self._available_cars[src] = max(self._available_cars[src] - n_moving, 0)
            self._available_cars[dst] = min(self._available_cars[dst] + n_moving, self._max_num_cars_per_loc)

            logger.info("returned n_return_0={}, n_return_1={}".format(n_return_0, n_return_1))
            logger.info("action {} available".format(action))

            info['returns'] = [n_return_0, n_return_1]  # Note returns in the information dictionary.

            self._t += 1

        # Arrives to the end of the episode (terminal state).
        terminated = True
        truncated = True
        # CarRentalActualEnv.set_obs_and_reward(obs, self._reward, terminated, truncated, info)


def log_step(episode: int, step: int, obs: np.ndarray, reward: float, done: bool, info: dict, action: tuple):
    """Utility function for printing logs.

    :param episode: Episode number.
    :param step: Time-step.
    :param obs: Observation of the current environment.
    :param reward: Reward from the current environment.
    :param done: Indicates whether the episode ends or not.
    :param info: Contains auxiliary diagnostic information (helpful for debugging, and sometimes learning).
    :param action: An action provided by the agent.
    """
    available_cars_0 = obs[0].item()
    available_cars_1 = obs[1].item()
    source_loc = action[0]
    n_moving = action[1].item()
    step_str = '{}-th step in {}-th episode / '.format(step, episode)
    obs_str = 'obs: {} / '.format((available_cars_0, available_cars_1))
    reward_str = 'reward: {} / '.format(reward)
    done_str = 'done: {} / '.format(done)
    info_str = 'info: {} / '.format(info)
    action_str = 'action: {}'.format((source_loc, n_moving))
    result_str = step_str + obs_str + reward_str + done_str + info_str + action_str
    logger.info(result_str)


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
SEED = 147

def main():
    simulator = CarRentalSimulator(NUM_STEPS, MAX_NUM_CARS_PER_LOC, RENTAL_CREDIT,
                                   CAR_MOVING_COST, RENTAL_RATE_0, RENTAL_RATE_1, RETURN_RATE_0, RETURN_RATE_1)
    simulator.start(SEED)

if __name__ == "__main__":
    main()