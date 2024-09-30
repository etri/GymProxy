# Author: Sae Hyong Park <labry@etri.re.kr>, Seungjae Shin <sjshin0505@{etri.re.kr, gmail.com}>

"""A standalone simulator for simulating the following reference:
R. S. Sutton and A. G. Barto, Reinforcement Learning - An Introduction, 2nd ed., 2018 (Example 4.3: Gambler's Problem).
"""

import logging
import numpy as np
import random

from typing import Any
from examples.utilities import get_step_log_str_for_standalone

# Setting logger
FORMAT = "[%(asctime)s|%(levelname)s|%(name)s] %(message)s"
DATE_FMT = "%H:%M:%S %Y-%m-%d"
log_level = logging.INFO
logging.basicConfig(format=FORMAT, datefmt=DATE_FMT, level=log_level)
logger = logging.getLogger('gamblers_problem_simulator')


class GamblersProblemSimulator:
    """Standalone Simulator for Gambler's problem.
    """

    def __init__(self, num_steps, prob_head, initial_capital, winning_captial):
        """Constructor.

        Args:
            num_steps (int): Number of time-steps.
            prob_head (float): Probability that a coin flip becomes head.
            initial_capital (int): Initial capital.
            winning_capital (int): Capital for winning the game.
        """
        self._num_steps = num_steps
        self._p_h = prob_head
        self._s = initial_capital
        self._s_win = winning_captial
        self._reward = 0.

    def start(self, seed_: int):
        """Start Gambler's problem simulation.

        Args:
            seed_: Seed for pseudo random number generator used by simulation.
        """
        terminated = False
        np.random.seed(seed_)
        t = 0
        while t < self._num_steps:
            obs = self._make_obs()  # Observation is current capital.
            action = self._policy(obs, self._reward)    # Action is the capital for betting.
            flip_result = self._flip_coin(action) # Flips the coin.

            # Checks if the gambler wins or not.
            if self._s >= self._s_win:
                msg = 'Wins the game because the capital becomes over {} dollars. action is {}'.format(self._s, bet)
                self._reward = 2.
                terminated = True
            elif self._s <= 0:
                msg = 'Loses the game due to out of money.'
                terminated = True
                self._reward = -0.5

            get_step_log_str_for_standalone(t, obs, self._reward, action)
            logger.info(result_str)

            if terminated:
                if self._reward == 0:
                    self._reward = -1.
                logger.info(msg)
                break

            t += 1

    def _flip_coin(self, bet) -> str:
        r = np.random.rand()

        if r < self._p_h:
            self._s += bet
            flip_result = 'head'
        else:
            self._s -= bet
            flip_result = 'tail'

        return flip_result

    def _make_obs(self) -> np.ndarray:
        return np.array([self._s], dtype=np.int_)

    def _policy(self, obs, reward) -> Any:
        return np.random.randint(1, min(self._s, self._s_win))


NUM_STEPS = 100
PROB_HEAD = 0.6
INITIAL_CAPITAL = 10
WINNING_CAPITAL = 100
SEED = 1


def main():
    simulator = GamblersProblemSimulator(NUM_STEPS, PROB_HEAD, INITIAL_CAPITAL, WINNING_CAPITAL)
    simulator.start(SEED)

if __name__ == "__main__":
    main()
