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
logger = logging.getLogger('main')
REWARD_SCALE = 0.005 # between 0.005 and 1

# Environment configuration parameters.
NUM_STEPS = 100
PROB_HEAD = 0.6
INITIAL_CAPITAL = 10
WINNING_CAPITAL = 100

NUM_EPISODES = 1
SEED = 147
class GamblersProblemSimulator:
    """External environment class that actually simulates gambler's problem.
    """

    def __init__(self, eid, client, num_steps, prob_head, initial_capital, winning_capital):
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
        self._p_h = prob_head
        self._s = initial_capital
        self._s_win = winning_capital
        self._reward = 0.
        self._t = 0

    def start(self, seed_:int):
        """Runs gambler's problem environment.

        :param kwargs: Dictionary of keyword arguments.
        """
        info = {}
        terminated = False
        np.random.seed(seed_)
        while self._t < self._num_steps:
            obs = np.array([self._s], dtype=np.int32)     # Observation is current capital.
            # bet = max(np.random.randint(0, self._s), 1)
            # logger.info(obs)
            # bet = min(obs.item(), self._s_win - self._s)
            if self._t != 0:
                bet = client.get_action(eid, obs)
            else:
                bet = np.array([0.])

            # print("bet: {}".format(bet))
            terminated = False
            info = {}

            raw_action = np.array(max(np.round(bet.item() * self._s), 1))
            bet = min(raw_action, self._s, self._s_win - self._s)

            r = np.random.rand()
            if r < self._p_h:
                self._s += bet
                info['flip_result'] = 'head'
            else:
                self._s -= bet
                info['flip_result'] = 'tail'

            if self._s >= self._s_win:
                info['msg'] = 'Wins the game because the capital becomes over {} dollars.'.format(self._s)
                self._reward = 2.
                terminated = True
            elif self._s <= 0:
                info['msg'] = 'Loses the game due to out of money.'
                self._reward = -0.5
                terminated = True
            else:
                self._reward = 0.

            self._t += 1
            if self._t >= self._num_steps:
                terminated = True

            obs = np.array([self._s], dtype=np.int32)
            self.client.log_returns(episode_id=self.eid, reward=self._reward, info=info)

            if terminated:
                if self._reward == 0:
                    self._reward = -1.
                logger.info(info['msg'])
                break

            self._t += 1

        obs = np.array([self._s], dtype=np.int_)
        obs = obs.flatten()
        terminated = True
        truncated = True
        self.client.end_episode(episode_id=self.eid, observation=obs)




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
    default=99999,
    help="Stop once the specified reward is reached.",
)
parser.add_argument(
    "--port", type=int, default=9999, help="The port to use (on localhost)."
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
        env = GamblersProblemSimulator(
            eid,
            client,
            num_steps = NUM_STEPS,
            prob_head = PROB_HEAD,
            initial_capital = INITIAL_CAPITAL,
            winning_capital = WINNING_CAPITAL,
        )
        env.start(i)