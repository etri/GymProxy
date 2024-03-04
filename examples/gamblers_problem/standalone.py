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


class GamblersProblemSimulator:
    """External environment class that actually simulates gambler's problem.
    """

    def __init__(self, num_steps, prob_head, initial_capital, winning_captial):
        """Constructor.

        :param kwargs: Dictionary of keyword arguments. It should have 'config' argument that is a dictionary for
        setting configuration parameters. kwargs['config'] should define following keys:
            num_steps (int): Number of time-steps.
            prob_head (float): Probability that a coin flip becomes head.
            initial_capital (float): Initial capital.
            winning_capital (float): Capital for winning the game.
        """
        self._num_steps = num_steps
        self._p_h = prob_head
        self._s = initial_capital
        self._s_win = winning_captial
        self._reward = 0.
        self._t = 0

    def start(self, seed_:int):
        """Runs gambler's problem environment.

        :param kwargs: Dictionary of keyword arguments.
        """
        terminated = False
        np.random.seed(seed_)
        while self._t < self._num_steps:
            obs = np.array([self._s], dtype=np.int64)     # Observation is current capital.
            bet = max(np.random.randint(0, self._s), 1)

            # Flips the coin
            if np.random.rand() < self._p_h:
                self._s += bet
                flip_result = 'head'
            else:
                self._s -= bet
                flip_result = 'tail'

            # Checks if the gambler wins or not.
            if self._s >= self._s_win:
                msg = 'Wins the game because the capital becomes {} dollars.'.format(self._s)
                self._reward = 1.
                terminated = True
            elif self._s <= 0:
                msg = 'Loses the game due to out of money.'
                terminated = True

            step_str = '{}-th step / '.format(self._t)
            obs_str = 'obs: {} / '.format(self._s)
            reward_str = 'reward: {} '.format(self._reward)
            result_str = step_str + obs_str + reward_str
            logger.info(result_str)

            if terminated:
                logger.info(msg)
                break

            self._t += 1


NUM_STEPS = 100
PROB_HEAD = 0.5
INITIAL_CAPITAL = 10
WINNING_CAPITAL = 100
SEED = 123


def main():
    simulator = GamblersProblemSimulator(NUM_STEPS, PROB_HEAD, INITIAL_CAPITAL, WINNING_CAPITAL)
    simulator.start(SEED)

if __name__ == "__main__":
    main()