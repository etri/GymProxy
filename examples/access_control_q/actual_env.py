# Author: Seungjae Shin <sjshin0505@{etri.re.kr, gmail.com}>

"""External environment for actually simulating an access control example in a simple queuing system. It is implemented
based on the following reference:
R. S. Sutton and A. G. Barto, Reinforcement Learning - An Introduction, 2nd ed., 2018 (Example 10.2: An Access-Control
Queuing Task).
"""

import logging
import numpy as np
import random

from gymproxy import BaseActualEnv, TerminateGymProxy

logger = logging.getLogger('access_control_q_actual_env')


class AccessControlQueueActualEnv(BaseActualEnv):
    """External environment class that actually simulates the access-control queuing task.
    """

    def __init__(self, **kwargs):
        """Constructor.

        :param kwargs: Dictionary of keyword arguments. It should have 'config' argument that is a dictionary for
        setting configuration parameters. kwargs['config'] should define following keys:
            num_steps (int): Number of time-steps.
            num_servers (int): Number of servers.
            server_free_probability (float): Probability that a server completes its task.
            priorities (list of float): Priorities that a customer can has.
        """
        env_proxy = kwargs['env_proxy']
        BaseActualEnv.__init__(self, env_proxy)
        config = kwargs['config']
        self._num_steps = config['num_steps']
        self._server_states = ['free'] * config['num_servers']
        self._server_free_probability = config['server_free_probability']
        self._priorities = config['priorities']
        self._reward = 0.
        self._t = 0

    def run(self, **kwargs):
        """Runs the access-control queuing task environment.

        :param kwargs: Dictionary of keyword arguments.
        """
        try:
            while self._t < self._num_steps:

                # Assumes that a new customer arrives. Chooses new customer's priority from the list of candidate
                # priorities.
                priority = self._priorities[random.randrange(0, len(self._priorities))]

                # Identifies free servers.
                free_servers = list(filter(lambda i_:
                                           self._server_states[i_] == 'free', range(0, len(self._server_states))))

                # Observation consists of customer's priority and number of free servers.
                obs = np.array([priority, len(free_servers)], dtype=np.int)

                done = False
                info = {}

                # Action can be 0 or 1: 0 means rejection of customer. 1 means acceptance of customer.
                action = AccessControlQueueActualEnv.get_action(obs, self._reward, done, info)

                if len(free_servers) > 0:
                    if action:  # Means acceptance.

                        # Randomly chooses a server for the customer among free ones.
                        i = free_servers[random.randrange(0, len(free_servers))]
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
                    if random.random() < self._server_free_probability:
                        self._server_states[i] = 'free'

                self._t += 1

            # Arrives to the end of the episode (terminal state).
            free_servers = list(filter(lambda i_:
                                       self._server_states[i_] == 'free', range(0, len(self._server_states))))
            obs = np.array([0, len(free_servers)], dtype=np.int)
            done = True
            info = {}
            AccessControlQueueActualEnv.set_obs_and_reward(obs, self._reward, done, info)

        # Exception handling block.
        except TerminateGymProxy:
            # Means termination signal triggered by the agent.
            logger.info('Terminating AccessControlQueue environment.')
            BaseActualEnv.env_proxy.release_lock()
            BaseActualEnv.env_proxy.set_gym_env_event()
            exit(1)
        except Exception as e:
            logger.exception(e)
            BaseActualEnv.env_proxy.release_lock()
            BaseActualEnv.env_proxy.set_gym_env_event()
            exit(1)

    def finish(self, **kwargs):
        """Finishes access-control queuing task environment.

        :param kwargs: Dictionary of keyword arguments.
        """
        return
