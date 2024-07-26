#!/usr/bin/env python

import argparse
import time
import numpy as np
from ray.rllib.env.policy_client import PolicyClient
import logging

# Setting logger
FORMAT = "[%(asctime)s|%(levelname)s|%(name)s] %(message)s"
DATE_FMT = "%H:%M:%S %Y-%m-%d"
log_level = logging.INFO
logging.basicConfig(format=FORMAT, datefmt=DATE_FMT, level=log_level)
logger = logging.getLogger('car_rental_simulator')


class CarRentalSimulator:
    """External environment class that simulates car rental operations."""

    def __init__(self, num_steps, max_num_cars_per_loc, rental_credit, car_moving_cost,
                 rental_rate_0, rental_rate_1, return_rate_0, return_rate_1):
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
        self._seed = 0

    def reset(self, seed=None):
        """Resets the environment."""
        if seed is not None:
            np.random.seed(seed)
        self._available_cars = [self._max_num_cars_per_loc] * 2
        self._reward = 0.
        self._t = 0
        self._seed = seed
        return np.array(self._available_cars, dtype=np.float32)

    def step(self, action):
        """Performs one step in the environment."""
        obs = None
        terminated = False
        truncated = False
        info = {}
        msg = None
        np.random.seed(self._seed)
        n_req_0 = np.random.poisson(self._lambda_rental_0)
        n_req_1 = np.random.poisson(self._lambda_rental_1)
        n_rentals = min(self._available_cars[0], n_req_0) + min(self._available_cars[1], n_req_1)
        self._reward = n_rentals * self._rental_credit * REWARD_SCALE

        if self._available_cars[0] < n_req_0 or self._available_cars[1] < n_req_1:
            terminated = True
            self._t = 0
        else:
            terminated = False

        self._available_cars[0] = max(self._available_cars[0] - n_req_0, 0)
        self._available_cars[1] = max(self._available_cars[1] - n_req_1, 0)

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

        src, n_moving = action
        dst = 1 - src
        n_moving = min(n_moving.item(), self._available_cars[src])
        #print(n_moving)
        self._available_cars[src] = max(self._available_cars[src] - n_moving, 0)
        self._available_cars[dst] = min(self._available_cars[dst] + n_moving, self._max_num_cars_per_loc)

        info['rental_requests'] = [n_req_0, n_req_1]
        if msg:
            info['msg'] = msg
        elif 'msg' in info.keys():
            del info['msg']

        n_return_0 = np.random.poisson(self._lambda_return_0)
        n_return_1 = np.random.poisson(self._lambda_return_1)
        self._available_cars[0] = min(self._available_cars[0] + n_return_0, self._max_num_cars_per_loc)
        self._available_cars[1] = min(self._available_cars[1] + n_return_1, self._max_num_cars_per_loc)

        self._t += 1
        self._reward -= n_moving * self._car_moving_cost * REWARD_SCALE

        if self._t >= self._num_steps:
            terminated = True

        return np.array(self._available_cars, dtype=np.int32), self._reward, terminated, truncated, info


REWARD_SCALE = 0.005
NUM_STEPS = 100
MAX_NUM_CARS_PER_LOC = 20
RENTAL_CREDIT = 10.
CAR_MOVING_COST = 2.
RENTAL_RATE_0 = 4
RENTAL_RATE_1 = 3
RETURN_RATE_0 = 3
RETURN_RATE_1 = 2

parser = argparse.ArgumentParser()
parser.add_argument("--no-train", action="store_true", help="Whether to disable training.")
parser.add_argument("--inference-mode", type=str, default="local", choices=["local", "remote"])
parser.add_argument("--off-policy", default=False, action="store_true", help="Whether to compute random actions instead of on-policy (Policy-computed) ones.")
parser.add_argument("--stop-reward", type=float, default=99999, help="Stop once the specified reward is reached.")
parser.add_argument("--port", type=int, default=9900, help="The port to use (on localhost).")

if __name__ == "__main__":
    args = parser.parse_args()

    env = CarRentalSimulator(
        num_steps=NUM_STEPS,
        max_num_cars_per_loc=MAX_NUM_CARS_PER_LOC,
        rental_credit=RENTAL_CREDIT,
        car_moving_cost=CAR_MOVING_COST,
        rental_rate_0=RENTAL_RATE_0,
        rental_rate_1=RENTAL_RATE_1,
        return_rate_0=RETURN_RATE_0,
        return_rate_1=RETURN_RATE_1
    )

    client = PolicyClient(f"http://localhost:{args.port}", inference_mode=args.inference_mode)
    obs = env.reset(seed=126)
    eid = client.start_episode(training_enabled=not args.no_train)

    rewards = 0.0
    while True:
        if args.off_policy:
            action = (np.random.choice([0, 1]), np.random.randint(0, 6))
            client.log_action(eid, obs, action)
        else:
            action = client.get_action(eid, obs)
            #action = tuple(action)  # Ensure action is a tuple

        #print(action)
        obs, reward, terminated, truncated, info = env.step(action)
        rewards += reward
        client.log_returns(eid, reward, info=info)

        if terminated:
            logger.info("Total reward: {}".format(rewards))
            if rewards >= args.stop_reward:
                logger.info("Target reward achieved, exiting")
                exit(0)

            rewards = 0.0
            client.end_episode(eid, obs)
            obs = env.reset(seed=126)
            eid = client.start_episode(training_enabled=not args.no_train)
