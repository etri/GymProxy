# Author: Seungjae Shin <sjshin0505@{etri.re.kr, gmail.com}>

"""External environment for actually simulating gambler's problem example. It is implemented based on the following
reference:
R. S. Sutton and A. G. Barto, Reinforcement Learning - An Introduction, 2nd ed., 2018 (Example 4.3: Gambler's Problem).
"""

import copy
import logging
from typing import Optional

import numpy as np

from gymproxy.actual_env import ActualEnv, TerminateGymProxy

logger = logging.getLogger('gamblers_problem_actual_env')

# NUM_STEPS = 100
# PROB_HEAD = 0.6
# INITIAL_CAPITAL = 30
# WINNING_CAPITAL = 100

class GamblersProblemActualEnv(ActualEnv):
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
        env_proxy = kwargs.get('env_proxy')
        ActualEnv.__init__(self, env_proxy)
        config = kwargs.get('config')
        # self._num_steps = config['num_steps']
        self._num_steps = kwargs['num_steps']
        # self._p_h = config['prob_head']
        self._p_h = kwargs['prob_head']
        # self._s = config['initial_capital']
        self._ic = np.array([copy.deepcopy(kwargs['initial_capital'])])
        # Create a copy of the _ic array to ensure _s is not just a reference
        self._s = np.copy(self._ic)
        # self._s_win = config['winning_capital']
        self._s_win = kwargs['winning_capital']
        self._reward = 0.
        self._t = 0
        # GamblersProblem.update_action_space(self, obs=self._s[0])

    def run(self, seed_:int, kwargs: Optional[dict] = None):
        """Runs gambler's problem environment.

        :param kwargs: Dictionary of keyword arguments.
        """

        try:
            # if kwargs is None:
            #     kwargs = {"num_steps" :100,
            #                 "prob_head":0.6,
            #                 "initial_capital":30,
            #                 "winning_capital":100,
            #     }
            #
            # self._num_steps = kwargs['num_steps']
            # # self._p_h = config['prob_head']
            # self._p_h = kwargs['prob_head']
            self._s = np.copy(self._ic)
            # print("self._s {} self._ic {}".format(self._s, self._ic))
            # self._s = np.array([kwargs['initial_capital']])
            # # self._s_win = config['winning_capital']
            # self._s_win = kwargs['winning_capital']
            self._reward = 0.
            self._t = 0
            # if seed_ >= 100:
            #     print("")
            # logger.info("Starting gambler's self._s=" + str(self._s))
            info = {}
            terminated = False
            truncated = False
            # logger.debug(seed_)
            np.random.seed(seed_)  # added seed
            while self._t < self._num_steps and not terminated:

                obs = np.array([self._s], dtype=np.float32)     # Observation is current capital.
                # obs = obs.flatten()
                # if obs[0] < 0 or obs[0] > 100:
                #     logger.info("obs in run: ", obs)
                #     print("obs in run: ", obs)

                # logger.info("current obs: {}".format(obs))
                raw_action = GamblersProblemActualEnv.get_action(obs, self._reward, terminated, truncated, info)
                # logger.info("current action {} and obs: {} action*obs {}".format(raw_action, obs, raw_action[0]*obs[0]))
                raw_action = np.array(max(np.round(raw_action[0] * obs[0]), 1))
                action = min(raw_action, self._s, self._s_win - self._s)
                # logger.info("action {}".format(action))
                info = {}
                #print(obs, action)

                # Amount of betting should be less than difference between the winning and current capitals.

                if action is None:
                    ActualEnv.env_proxy.terminate_sync()
                    exit(1)
                else:
                    tmp = action
                    #print("tmp {}, self._s_win {} self._s {}".format(tmp, self._s_win, self._s))
                    bet = np.int32(action)
                    # logger.info("bet: {} action {} s_win - s {}".format(bet, action, self._s_win-self._s))

                # Flips the coin
                r = np.random.rand()
                # logger.info("r {}".format(r))
                # logger.debug(self._p_h)
                # logger.debug(r)

                if r < self._p_h:
                    # logger.info("heads")
                    bet = bet.flatten()
                    # print(type(self._s), type(bet))
                    # print("_s {} bet {}", self._s, bet)
                    self._s += bet
                    # info['flip_result'] = 'head'
                else:
                    # logger.info("tails")
                    bet = bet.flatten()
                    # print("flip_result ", self._s, bet)
                    self._s -= bet
                    # info['flip_result'] = 'tail'

                # Checks if the gambler wins or not.
                if self._s >= self._s_win:
                    # info['msg'] = 'Wins the game because the capital becomes over {} dollars.'.format(self._s)
                    terminated = True
                    truncated = True
                    self._reward = 2.
                    # self._s = self._ic
                elif self._s <= 0.:
                    # info['msg'] = 'Loses the game due to out of money.'
                    terminated = True
                    truncated = True
                    # self._s = self._ic
                    self._reward = -0.5

                self._t += 1
                # GamblersProblem.update_action_space(self, obs=self._s)

            # Arrives to the end of the episode (terminal state).
            if self._t >= self._num_steps and self._reward == 0:
                self._reward = -1.
            obs = np.array([self._s], dtype=np.float32)
            # obs = obs.flatten()
            terminated = True
            truncated = True
            GamblersProblemActualEnv.set_obs_and_reward(obs, self._reward, terminated, truncated, info)

        # Exception handling block.
        except TerminateGymProxy:
            # Means termination signal triggered by the agent.
            logger.info('Terminating gamblers_problem environment.')
            ActualEnv.env_proxy.terminate_sync()
            exit(1)
        except Exception as e:
            print("Exception e")
            logger.exception(e)
            ActualEnv.env_proxy.terminate_sync()
            exit(1)

    def finish(self, kwargs: Optional[dict] = None):
        """Finishes Gambler's problem environment.

        :param kwargs: Dictionary of keyword arguments.
        """
        logger.debug("gambler finish")
        return
