# Author: Sae Hyong Park <labry@etri.re.kr>
# Author: Seungjae Shin <sjshin0505@{etri.re.kr, gmail.com}>

"""Testing script for AccessControlQueue environment implemented based on the following reference:
R. S. Sutton and A. G. Barto, Reinforcement Learning - An Introduction, 2nd ed., 2018 (Example 10.2: An Access-Control
Queuing Task).
"""

import logging
import numpy as np

from examples.access_control_q import *

# Setting logger
FORMAT = "[%(asctime)s|%(levelname)s|%(name)s] %(message)s"
DATE_FMT = "%H:%M:%S %Y-%m-%d"
log_level = logging.INFO
logging.basicConfig(format=FORMAT, datefmt=DATE_FMT, level=log_level)
logger = logging.getLogger('main')

# Environment configuration parameters.
NUM_STEPS = 100
NUM_SERVERS = 10
SERVER_FREE_PROB = 0.06
PRIORITIES = [1., 2., 4., 8.]

NUM_EPISODES = 100


def main():
    """Main routine of testing AccessControlQueue gym-type environment.
    """
    config = {'num_steps': NUM_STEPS,
              'num_servers': NUM_SERVERS,
              'server_free_probability': SERVER_FREE_PROB,
              'priorities': PRIORITIES}

    env = gym.make(id='AccessControlQueue-v0', config=config)
    accumulated_result = 0
    for i in range(0, NUM_EPISODES):
        j = 0
        obs, info = env.reset(seed=126, options={})
        logger.info(str(obs)+str(info))
        while True:
            env.render()
            action = env.action_space.sample()  # Means random agent.
            obs, reward, terminated, truncated, info = env.step(action)
            accumulated_result += reward

            log_step(i, j, obs, reward, terminated, info, action)
            j = j + 1
            if terminated:
                break

    logger.info("Accumulated result: " + str(accumulated_result))
    env.close()


def log_step(episode: int, step: int, obs: np.ndarray, reward: float, done: bool, info: dict, action: int):
    """Utility function for printing logs.

    :param episode: Episode number.
    :param step: Time-step.
    :param obs: Observation of the current environment.
    :param reward: Reward from the current environment.
    :param done: Indicates whether the episode ends or not.
    :param info: Contains auxiliary diagnostic information (helpful for debugging, and sometimes learning).
    :param action: An action provided by the agent.
    """
    priority = obs[0].item()
    num_free_servers = obs[1].item()
    step_str = '{}-th step in {}-th episode / '.format(step, episode)
    obs_str = 'obs: {} / '.format((priority, num_free_servers))
    reward_str = 'reward: {} / '.format(reward)
    done_str = 'done: {} / '.format(done)
    info_str = 'info: {} / '.format(info)
    action_str = 'action: {}'.format(True if action else False)
    result_str = step_str + obs_str + reward_str + done_str + info_str + action_str
    logger.info(result_str)


if __name__ == "__main__":
    main()
