#!/usr/bin/env python

import argparse
import logging

import numpy as np
from ray.rllib.env.policy_client import PolicyClient

# Setting logger
FORMAT = "[%(asctime)s|%(levelname)s|%(name)s] %(message)s"
DATE_FMT = "%H:%M:%S %Y-%m-%d"
log_level = logging.INFO
logging.basicConfig(format=FORMAT, datefmt=DATE_FMT, level=log_level)
logger = logging.getLogger('access_control_simulator')


class AccessControlSimulator:
    """External environment class that actually simulates gambler's problem.
    """

    def __init__(self, eid, client, num_steps, num_servers, server_free_prob, priorities):
        """Constructor.

        :param kwargs: Dictionary of keyword arguments. It should have 'config' argument that is a dictionary for
        setting configuration parameters. kwargs['config'] should define following keys:
            num_steps (int): Number of time-steps.
            prob_head (float): Probability that a coin flip becomes head.
            initial_capital (float): Initial capital.
            winning_capital (float): Capital for winning the game.
        """
        self.eid = eid
        self.client = client
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

            if self._t == 0:
                self.obs = obs

            terminated = False
            truncated = False
            info = {}

            # Action can be 0 or 1: 0 means rejection of customer. 1 means acceptance of customer.
            # action = AccessControlQueueActualEnv.get_action(obs, self._reward, terminated, truncated, info)
            # action = np.random.choice([False,True])
            action = self.client.get_action(self.eid, self.obs)

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
            if self._t >= self._num_steps :
                print("end")
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
            self.client.log_returns(self.eid, self._reward, info=info)

        logger.info(self._accumulated_reward)
        self.client.end_episode(self.eid, self.obs)



parser = argparse.ArgumentParser()
parser.add_argument(
    "--no-train", action="store_true", help="Whether to disable training."
)
parser.add_argument(
    "--inference-mode", type=str, default="local", choices=["local", "remote"]
)
parser.add_argument(
    "--off-policy",
    default=False,
    action="store_true",
    help="Whether to compute random actions instead of on-policy (Policy-computed) ones.",
)
parser.add_argument(
    "--stop-reward",
    type=float,
    default=9999,
    help="Stop once the specified reward is reached.",
)
parser.add_argument(
    "--port", type=int, default=9900, help="The port to use (on localhost)."
)

if __name__ == "__main__":
    # start = time.time()
    args = parser.parse_args()
    client = PolicyClient(
        f"http://localhost:{args.port}", inference_mode=args.inference_mode
    )

    rewards = 0.0
    while True:
        eid = client.start_episode(training_enabled=not args.no_train)
        i = np.random.randint(1000)
        env = AccessControlSimulator(
            eid,
            client,
            num_steps=100,
            num_servers=10,
            server_free_prob=0.06,
            priorities=[1., 2., 4., 8.]
        )
        env.start(i)