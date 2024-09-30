# Author: Sae Hyong Park <labry@etri.re.kr>, Seungjae Shin <sjshin0505@{etri.re.kr, gmail.com}>

"""Testing script for AccessControlQueue environment implemented based on the following reference:
R. S. Sutton and A. G. Barto, Reinforcement Learning - An Introduction, 2nd ed., 2018 (Example 10.2: An Access-Control Queuing Task).
"""

import logging

from examples.utilities import get_step_log_str
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

    for i in range(0, NUM_EPISODES):
        j = 0
        obs, info = env.reset(seed=2024, options={})
        while True:
            action = env.action_space.sample()  # Means random agent.
            obs, reward, terminated, truncated, info = env.step(action)
            step_log_str = get_step_log_str(i, j, obs, reward, terminated, truncated, info, action)
            logger.info(step_log_str)
            j = j + 1
            if terminated or truncated:
                break

    env.close()


if __name__ == "__main__":
    main()
