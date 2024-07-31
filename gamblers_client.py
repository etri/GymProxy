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
logger = logging.getLogger('gamblers_problem_simulator')


class GamblersProblemSimulator:
    """External environment class that actually simulates gambler's problem."""

    def __init__(self, num_steps, prob_head, initial_capital, winning_capital):
        self._num_steps = num_steps
        self._p_h = prob_head
        self._s = initial_capital
        self._s_win = winning_capital
        self._reward = 0.
        self._t = 0

    def reset(self, seed=None):
        """Resets the environment."""
        if seed is not None:
            np.random.seed(seed)
        self._s = INITIAL_CAPITAL
        self._reward = 0.
        self._t = 0
        return np.array([self._s], dtype=np.int32)

    def step(self, bet):
        """Performs one step in the environment."""
        print("bet: {}".format(bet))
        terminated = False
        flip_result = None

        raw_action = np.array(max(np.round(bet.item() * self._s), 1))
        bet = min(raw_action, self._s, self._s_win - self._s)

        r = np.random.rand()
        if r < self._p_h:
            self._s += bet
            flip_result = 'head'
        else:
            self._s -= bet
            flip_result = 'tail'

        if self._s >= self._s_win:
            self._reward = 2.
            terminated = True
        elif self._s <= 0:
            self._reward = -0.5
            terminated = True
        else:
            self._reward = 0.

        self._t += 1
        if self._t >= self._num_steps:
            terminated = True

        obs = np.array([self._s], dtype=np.int32)
        return obs, self._reward, terminated, {"flip_result": flip_result, "bet": bet}


NUM_STEPS = 100
PROB_HEAD = 0.6
INITIAL_CAPITAL = 10
WINNING_CAPITAL = 100
SEED = 1

parser = argparse.ArgumentParser()
parser.add_argument("--no-train", action="store_true", help="Whether to disable training.")
parser.add_argument("--inference-mode", type=str, default="local", choices=["local", "remote"])
parser.add_argument("--off-policy", default=False, action="store_true", help="Whether to compute random actions instead of on-policy (Policy-computed) ones.")
parser.add_argument("--stop-reward", type=float, default=200000.0, help="Stop once the specified reward is reached.")
parser.add_argument("--port", type=int, default=9999, help="The port to use (on localhost).")

if __name__ == "__main__":
    args = parser.parse_args()

    env = GamblersProblemSimulator(
        num_steps=NUM_STEPS,
        prob_head=PROB_HEAD,
        initial_capital=INITIAL_CAPITAL,
        winning_capital=WINNING_CAPITAL
    )

    client = PolicyClient(f"http://localhost:{args.port}", inference_mode=args.inference_mode)
    obs = env.reset(seed=SEED)
    eid = client.start_episode(training_enabled=not args.no_train)

    rewards = 0.0
    while True:
        if args.off_policy:
            bet = min(obs.item(), WINNING_CAPITAL - obs.item())
            client.log_action(eid, obs, bet)
        else:
            bet = client.get_action(eid, obs)

        #print("bet : {}".format(bet))
        obs, reward, terminated, info = env.step(bet)
        rewards += reward

        client.log_returns(eid, reward, info=info)

        if terminated:
            logger.info("Total reward: {}".format(rewards))
            if rewards >= args.stop_reward:
                logger.info("Target reward achieved, exiting")
                exit(0)

            rewards = 0.0
            client.end_episode(eid, obs)
            obs = env.reset(seed=SEED)
            eid = client.start_episode(training_enabled=not args.no_train)
