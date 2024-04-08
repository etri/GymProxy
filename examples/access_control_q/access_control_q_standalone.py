# Author: Sae Hyong Park <labry@etri.re.kr>
# Author: Seungjae Shin <sjshin0505@{etri.re.kr, gmail.com}>

"""External environment for actually simulating gambler's problem example. It is implemented based on the following
reference:
R. S. Sutton and A. G. Barto, Reinforcement Learning - An Introduction, 2nd ed., 2018 (Example 4.3: Gambler's Problem).
"""

import logging
import numpy as np
import random

# Setting logger
FORMAT = "[%(asctime)s|%(levelname)s|%(name)s] %(message)s"
DATE_FMT = "%H:%M:%S %Y-%m-%d"
log_level = logging.INFO
logging.basicConfig(format=FORMAT, datefmt=DATE_FMT, level=log_level)
logger = logging.getLogger('gamblers_problem_simulator')


class AccessControlSimulator:
    """External environment class that actually simulates gambler's problem.
    """

    def __init__(self, num_steps, num_servers, server_free_prob, priorities):
        """Constructor.

        :param kwargs: Dictionary of keyword arguments. It should have 'config' argument that is a dictionary for
        setting configuration parameters. kwargs['config'] should define following keys:
            num_steps (int): Number of time-steps.
            prob_head (float): Probability that a coin flip becomes head.
            initial_capital (float): Initial capital.
            winning_capital (float): Capital for winning the game.
        """
        self._num_steps = num_steps
        self._server_states = ['free'] * num_servers
        self._server_free_probability = server_free_prob
        self._priorities = priorities
        self._reward = 0.
        self._t = 0
        self._accumulated_reward = 0

    def start(self, seed_:int):
        """Runs the access-control queuing task environment.
        """
        np.random.seed(seed_)
        while self._t < self._num_steps:

            # Assumes that a new customer arrives. Chooses new customer's priority from the list of candidate
            # priorities.
            priority = self._priorities[np.random.randint(0, len(self._priorities))]

            # Identifies free servers.
            free_servers = list(filter(lambda i_:
                                       self._server_states[i_] == 'free', range(0, len(self._server_states))))

            # Observation consists of customer's priority and number of free servers.
            obs = np.array([priority, len(free_servers)], dtype=np.int32)

            terminated = False
            truncated = False
            info = {}

            # Action can be 0 or 1: 0 means rejection of customer. 1 means acceptance of customer.
            # action = AccessControlQueueActualEnv.get_action(obs, self._reward, terminated, truncated, info)
            action = np.random.choice([False,True])

            if len(free_servers) > 0:
                if action:  # Means acceptance.

                    # Randomly chooses a server for the customer among free ones.
                    i = free_servers[np.random.randint(0, len(free_servers))]
                    self._server_states[i] = 'busy'     # Selected server becomes busy.

                    self._reward = priority     # Reward is the priority of accepted customer.
                else:   # Means rejection.
                    self._reward = 0.
            else:   # Rejects the customer if the number of free servers is 0.
                self._reward = 0.
            busy_servers = list(filter(lambda i_:
                                       self._server_states[i_] == 'busy', range(0, len(self._server_states))))

            # Busy servers become free with _server_free_probability.
            for i in busy_servers:
                if np.random.rand() < self._server_free_probability:
                    self._server_states[i] = 'free'

            self._t += 1
            self._accumulated_reward += self._reward

            # Arrives to the end of the episode (terminal state).
            free_servers = list(filter(lambda i_:
                                       self._server_states[i_] == 'free', range(0, len(self._server_states))))
            obs = np.array([0, len(free_servers)], dtype=np.int32)
            terminated = True
            truncated = True
            info = {}
            step_str = '{}-th step / '.format(self._t)
            obs_str = 'obs: {} / '.format(obs)
            reward_str = 'reward: {} '.format(self._reward)
            info_str = 'info: {} / '.format(info)
            action_str = 'action: {} '.format(action)
            result_str = step_str + obs_str + reward_str + info_str + action_str
            logger.info(result_str)

        logger.info(self._accumulated_reward)


NUM_STEPS = 100
NUM_SERVERS = 10
SERVER_FREE_PROB = 0.06
PRIORITIES = [1., 2., 4., 8.]
SEED = 126


def main():
    simulator = AccessControlSimulator(NUM_STEPS, NUM_SERVERS, SERVER_FREE_PROB, PRIORITIES)
    simulator.start(SEED)

if __name__ == "__main__":
    main()