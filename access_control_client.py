#!/usr/bin/env python

import argparse
import numpy as np
from ray.rllib.env.policy_client import PolicyClient
import logging

# Setting logger
FORMAT = "[%(asctime)s|%(levelname)s|%(name)s] %(message)s"
DATE_FMT = "%H:%M:%S %Y-%m-%d"
log_level = logging.INFO
logging.basicConfig(format=FORMAT, datefmt=DATE_FMT, level=log_level)
logger = logging.getLogger('access_control_simulator')


class AccessControlSimulator:
    """External environment class that actually simulates access control queue."""

    def __init__(self, num_steps, num_servers, server_free_prob, priorities):
        self._num_steps = num_steps
        self._server_states = ['free'] * num_servers
        self._server_free_probability = server_free_prob
        self._priorities = priorities
        self._reward = 0.
        self._t = 0
        self._accumulated_reward = 0

    def reset(self, seed=None):
        """Resets the environment."""
        if seed is not None:
            np.random.seed(seed)
        self._server_states = ['free'] * len(self._server_states)
        self._reward = 0.
        self._t = 0
        self._accumulated_reward = 0
        priority = self._priorities[np.random.randint(0, len(self._priorities))]
        free_servers = list(filter(lambda i_: self._server_states[i_] == 'free', range(len(self._server_states))))
        return np.array([priority, len(free_servers)], dtype=np.int32)

    def step(self, action):
        """Performs one step in the environment."""
        priority = self._priorities[np.random.randint(0, len(self._priorities))]
        free_servers = list(filter(lambda i_: self._server_states[i_] == 'free', range(len(self._server_states))))
        obs = np.array([priority, len(free_servers)], dtype=np.int32)

        if len(free_servers) > 0:
            if action:  # Acceptance
                i = free_servers[np.random.randint(0, len(free_servers))]
                self._server_states[i] = 'busy'
                self._reward = priority * 0.005
            else:  # Rejection
                self._reward = 0.
        else:  # No free servers, reject customer
            self._reward = 0.

        busy_servers = list(filter(lambda i_: self._server_states[i_] == 'busy', range(len(self._server_states))))
        for i in busy_servers:
            if np.random.rand() < self._server_free_probability:
                self._server_states[i] = 'free'

        self._t += 1
        self._accumulated_reward += self._reward
        terminated = self._t >= self._num_steps

        if terminated:
            free_servers = list(filter(lambda i_: self._server_states[i_] == 'free', range(len(self._server_states))))
            obs = np.array([0, len(free_servers)], dtype=np.int32)

        return obs, self._reward, terminated, {}


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
    args = parser.parse_args()

    # Initialize the Access Control Queue simulator
    env = AccessControlSimulator(
        num_steps=100,
        num_servers=10,
        server_free_prob=0.06,
        priorities=[1., 2., 4., 8.]
    )

    # Connect to the policy server
    client = PolicyClient(
        f"http://localhost:{args.port}", inference_mode=args.inference_mode
    )

    # Start a new episode
    obs = env.reset(seed=126)
    eid = client.start_episode(training_enabled=not args.no_train)

    rewards = 0.0
    while True:
        # Compute an action randomly (off-policy) and log it.
        if args.off_policy:
            action = np.random.choice([False, True])
            client.log_action(eid, obs, action)
        # Compute an action locally or remotely (on server).
        # No need to log it here as the action
        else:
            action = client.get_action(eid, obs)

        # Perform a step in the external simulator (env).
        obs, reward, terminated, info = env.step(action)
        rewards += reward

        # Log next-obs, rewards, and infos.
        client.log_returns(eid, reward, info=info)

        # Reset the episode if done.
        if terminated:
            logger.info("Total reward: {}".format(rewards))
            if rewards >= args.stop_reward:
                logger.info("Target reward achieved, exiting")
                exit(0)

            rewards = 0.0

            # End the old episode.
            client.end_episode(eid, obs)

            # Start a new episode.
            obs = env.reset(seed=126)
            eid = client.start_episode(training_enabled=not args.no_train)
