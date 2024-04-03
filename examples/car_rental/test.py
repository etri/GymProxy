# Author: Sae Hyong Park <labry@etri.re.kr>
# Author: Seungjae Shin <sjshin0505@{etri.re.kr, gmail.com}>

"""Testing script for CarRental environment implemented based on the following reference:
R. S. Sutton and A. G. Barto, Reinforcement Learning - An Introduction, 2nd ed., 2018 (Example 4.2: Jack's car rental).
"""

import logging
import numpy as np

from examples.car_rental import *

# Setting logger
FORMAT = "[%(asctime)s|%(levelname)s|%(name)s] %(message)s"
DATE_FMT = "%H:%M:%S %Y-%m-%d"
log_level = logging.INFO
logging.basicConfig(format=FORMAT, datefmt=DATE_FMT, level=log_level)
logger = logging.getLogger('main')

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


def main():
    """Main routine of testing CarRental gym-type environment.
    """
    config = {'num_steps': NUM_STEPS,
              'max_num_cars_per_loc': MAX_NUM_CARS_PER_LOC,
              'rental_credit': RENTAL_CREDIT,
              'car_moving_cost': CAR_MOVING_COST,
              'rental_rate_0': RENTAL_RATE_0,
              'rental_rate_1': RENTAL_RATE_1,
              'return_rate_0': RETURN_RATE_0,
              'return_rate_1': RETURN_RATE_1}

    env = gym.make(id='CarRental-v0', config=config)
    print("env:", env)
    for i in range(0, NUM_EPISODES):
        j = 0
        print("before obs, info")
        obs, info = env.reset(seed=147, options={})
        print("obs, info:", obs,info)
        logger.info(str(obs)+ str(info))
        while True:
            env.render()
            action = env.action_space.sample()  # Means random agent.
            obs, reward, terminated, truncated, info = env.step(action)
            log_step(i, j, obs, reward, terminated, info, action)
            j = j + 1
            if terminated:
                break

    env.close()


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


if __name__ == "__main__":
    main()
