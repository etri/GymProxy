# Author: Seungjae Shin <sjshin0505@{etri.re.kr, gmail.com}>

"""External environment for actually simulating gambler's problem example. It is implemented based on the following
reference:
R. S. Sutton and A. G. Barto, Reinforcement Learning - An Introduction, 2nd ed., 2018 (Example 4.3: Gambler's Problem).
"""

import logging
import numpy as np
import random
from typing import Optional

from gymproxy import BaseActualEnv, TerminateGymProxy

logger = logging.getLogger('gamblers_problem_actual_env')


class GamblersProblemActualEnv(BaseActualEnv):
    """External environment class that actually simulates gambler's problem.
    """

    def __init__(self, kwargs: Optional[dict] = None):
        """Constructor.

        :param kwargs: Dictionary of keyword arguments. It should have 'config' argument that is a dictionary for
        setting configuration parameters. kwargs['config'] should define following keys:
            num_steps (int): Number of time-steps.
            prob_head (float): Probability that a coin flip becomes head.
            initial_capital (float): Initial capital.
            winning_capital (float): Capital for winning the game.
        """
        NUM_STEPS = 100
        PROB_HEAD = 0.5
        INITIAL_CAPITAL = 10
        WINNING_CAPITAL = 100

        env_proxy = kwargs.get('env_proxy')
        BaseActualEnv.__init__(self, env_proxy)
        config = kwargs.get('config')
        # self._num_steps = config['num_steps']
        self._num_steps = NUM_STEPS
        # self._p_h = config['prob_head']
        self._p_h = PROB_HEAD
        # self._s = config['initial_capital']
        self._s = INITIAL_CAPITAL
        # self._s_win = config['winning_capital']
        self._s_win = WINNING_CAPITAL
        self._reward = 0.
        self._t = 0

    def run(self, seed_:int, kwargs: Optional[dict] = None):
        """Runs gambler's problem environment.

        :param kwargs: Dictionary of keyword arguments.
        """

        try:

            done = False
            truncated = False
            info = {}
            logger.debug(seed_)
            np.random.seed(seed_)  # added seed
            while self._t < self._num_steps and not done:
                obs = np.array([self._s], dtype=np.int_)     # Observation is current capital.
                action = GamblersProblemActualEnv.get_action(obs, self._reward, done, truncated, info)
                info = {}
                #print(obs, action)

                # Amount of betting should be less than difference between the winning and current capitals.

                if action is None:
                    BaseActualEnv.env_proxy.release_lock()
                    BaseActualEnv.env_proxy.set_gym_env_event()
                    exit(1)
                else:
                    tmp = action
                    #print("tmp {}, self._s_win {} self._s {}".format(tmp, self._s_win, self._s))
                    bet = min(action, self._s_win - self._s)

                # Flips the coin
                r = np.random.rand()
                logger.debug(self._p_h)
                logger.debug(r)

                if r < self._p_h:
                    self._s += bet
                    info['flip_result'] = 'head'
                else:
                    self._s -= bet
                    info['flip_result'] = 'tail'

                # Checks if the gambler wins or not.
                if self._s >= self._s_win:
                    info['msg'] = 'Wins the game because the capital becomes over {} dollars.'.format(self._s)
                    done = True
                    truncated = True
                    self._reward = 1.
                elif self._s <= 0.:
                    info['msg'] = 'Loses the game due to out of money.'
                    done = True
                    truncated = True

                self._t += 1

            # Arrives to the end of the episode (terminal state).
            obs = np.array([self._s], dtype=np.int_)
            done = True
            truncated = True
            GamblersProblemActualEnv.set_obs_and_reward(obs, self._reward, done, truncated, info)

        # Exception handling block.
        except TerminateGymProxy:
            # Means termination signal triggered by the agent.
            print("hello")
            logger.info('Terminating gamblers_problem environment.')
            BaseActualEnv.env_proxy.release_lock()
            BaseActualEnv.env_proxy.set_gym_env_event()
            exit(1)

        except Exception as e:
            print("hello")
            logger.exception(e)
            BaseActualEnv.env_proxy.release_lock()
            BaseActualEnv.env_proxy.set_gym_env_event()
            exit(1)

    def finish(self, kwargs: Optional[dict] = None):
        """Finishes Gambler's problem environment.

        :param kwargs: Dictionary of keyword arguments.
        """
        logger.debug("gambler finish")
        return
