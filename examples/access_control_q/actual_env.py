# Author: Seungjae Shin <sjshin0505@{etri.re.kr, gmail.com}>

"""External environment for actually simulating an access control example in a simple queuing system. It is implemented
based on the following reference:
R. S. Sutton and A. G. Barto, Reinforcement Learning - An Introduction, 2nd ed., 2018 (Example 10.2: An Access-Control
Queuing Task).
"""

import logging
import numpy as np
import random
import examples.access_control_q

from typing import Optional
from gymproxy.actual_env import ActualEnv, TerminateGymProxy

logger = logging.getLogger('access_control_q_actual_env')

class AccessControlQueueActualEnv(ActualEnv):
    """External environment class that actually simulates the access-control queuing task.
    """

    def __init__(self, kwargs: Optional[dict] = None):
        """Constructor.

        Args:
            kwargs: Dictionary of keyword arguments. 
            It should have 'config' argument that is a dictionary for setting configuration parameters. 
            kwargs['config'] should define following keys:
                num_steps (int): Number of time-steps.
                num_servers (int): Number of servers.
                server_free_probability (float): Probability that a server completes its task.
                priorities (list of float): Priorities that a customer can has.
        """
        env_proxy = kwargs['env_proxy']
        ActualEnv.__init__(self, env_proxy)
        config = kwargs.get('config')
        self._num_steps = kwargs['num_steps']
        self._server_states = ['free'] * kwargs['num_servers']
        self._server_free_probability = kwargs['server_free_probability']
        self._priorities = kwargs['priorities']
        self._reward = 0.
        self._t = 0

    def run(self, seed_:int, kwargs: Optional[dict] = None):
        """Runs the access-control queuing task environment.

        Args:
            kwargs: Dictionary of keyword arguments.
        """
        try:
            obs = None
            terminated = False
            truncated = False
            info = {}
            np.random.seed(seed_)
            self._reward = 0.
            self._t = 0
            while self._t < self._num_steps:
                # Assumes that a new customer arrives. Chooses new customer's priority from the list of candidate priorities.
                priority = get_new_customer(self._priorities)

                free_servers = get_free_servers(self._server_states)    # Identifies free servers.
                obs = make_obs(priority, free_servers)    # Observation consists of customer's priority and number of free servers.
                terminated = False
                truncated = False
                info = {}

                # Action can be 0 or 1: 0 means rejection of customer. 1 means acceptance of customer.
                action = AccessControlQueueActualEnv.get_action(obs, self._reward, terminated, truncated, info)

                if len(free_servers) > 0:
                    if action:  # Means acceptance.
                        i = choose_free_server(free_servers)    # Randomly chooses a server for the customer among free ones.
                        self._server_states[i] = 'busy'     # Selected server becomes busy.
                        self._reward = priority * 0.005     # Reward is the priority of accepted customer.
                    else:   # Means rejection.
                        self._reward = 0.
                else:   # Rejects the customer if the number of free servers is 0.
                    self._reward = 0.
                busy_servers = get_busy_servers(self._server_states)
                
                # Busy servers become free with _server_free_probability.
                for i in busy_servers:
                    if random.random() < self._server_free_probability:
                        self._server_states[i] = 'free'

                self._t += 1

            # Arrives to the end of the episode (terminal state).
            free_servers = get_free_servers(self._server_states)
            obs = make_obs(0, free_servers)
            terminated = True
            truncated = True
            #info = {}
            AccessControlQueueActualEnv.set_obs_and_reward(obs, self._reward, terminated, truncated, info)
            # logger.info("self._cnt: {}".format(self._cnt))
            # print("self._cnt: {}".format(self._cnt))
        # Exception handling block.
        except TerminateGymProxy:
            # Means termination signal triggered by the agent.
            logger.info('Terminating AccessControlQueue environment.')
            ActualEnv.env_proxy.terminate_sync()
            exit(1)
        except Exception as e:
            logger.exception(e)
            ActualEnv.env_proxy.terminate_sync()
            exit(1)

    def finish(self, kwargs: Optional[dict] = None):
        """Finishes access-control queuing task environment.

        :param kwargs: Dictionary of keyword arguments.
        """
        logger.info("finish")
        return
