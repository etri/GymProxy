# Author: Seungjae Shin <sjshin0505@{etri.re.kr, gmail.com}>

"""External environment for actually simulating gambler's problem example. It is implemented based on the following
reference:
R. S. Sutton and A. G. Barto, Reinforcement Learning - An Introduction, 2nd ed., 2018 (Example 4.3: Gambler's Problem).
"""

import logging
import numpy as np
import random

from gymproxy import BaseActualEnv, TerminateGymProxy

logger = logging.getLogger('gamblers_problem_actual_env')


class GamblersProblemActualEnv(BaseActualEnv):
    """External environment class that actually simulates gambler's problem.
    """

    def __init__(self, **kwargs):
        """Constructor.

        :param kwargs: Dictionary of keyword arguments. It should have 'config' argument that is a dictionary for
        setting configuration parameters. kwargs['config'] should define following keys:
            num_steps (int): Number of time-steps.
            prob_head (float): Probability that a coin flip becomes head.
            initial_capital (float): Initial capital.
            winning_capital (float): Capital for winning the game.
        """
        env_proxy = kwargs['env_proxy']
        BaseActualEnv.__init__(self, env_proxy)
        config = kwargs['config']
        self._num_steps = config['num_steps']
        self._p_h = config['prob_head']
        self._s = config['initial_capital']
        self._s_win = config['winning_capital']
        self._reward = 0.
        self._t = 0

    def run(self, seed_:int, **kwargs):
        """Runs gambler's problem environment.

        :param kwargs: Dictionary of keyword arguments.
        """
        try:
            done = False
            truncated = False
            info = {}
            while self._t < self._num_steps and not done:
                obs = np.array([self._s], dtype=np.int64)     # Observation is current capital.
                action = GamblersProblemActualEnv.get_action(obs, self._reward, done, truncated, info)
                info = {}

                # Amount of betting should be less than difference between the winning and current capitals.
                bet = min(action.item(), self._s_win - self._s)

                random.seed(seed_) # added seed
                # Flips the coin
                if random.random() < self._p_h:
                    self._s += bet
                    info['flip_result'] = 'head'
                else:
                    self._s -= bet
                    info['flip_result'] = 'tail'

                # Checks if the gambler wins or not.
                if self._s >= self._s_win:
                    info['msg'] = 'Wins the game because the capital becomes {} dollars.'.format(self._s)
                    done = True
                    truncated = True
                    self._reward = 1.
                elif self._s <= 0.:
                    info['msg'] = 'Loses the game due to out of money.'
                    done = True
                    truncated = True

                self._t += 1

            # Arrives to the end of the episode (terminal state).
            obs = np.array([self._s], dtype=np.int64)
            done = True
            truncated = True
            GamblersProblemActualEnv.set_obs_and_reward(obs, self._reward, done, truncated, info)

        # Exception handling block.
        except TerminateGymProxy:
            # Means termination signal triggered by the agent.
            logger.info('Terminating CarRental environment.')
            BaseActualEnv.env_proxy.release_lock()
            BaseActualEnv.env_proxy.set_gym_env_event()
            exit(1)
        except Exception as e:
            logger.exception(e)
            BaseActualEnv.env_proxy.release_lock()
            BaseActualEnv.env_proxy.set_gym_env_event()
            exit(1)

    def finish(self, **kwargs):
        """Finishes Gambler's problem environment.

        :param kwargs: Dictionary of keyword arguments.
        """
        return
