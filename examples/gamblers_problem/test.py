# Author: Seungjae Shin <sjshin0505@{etri.re.kr, gmail.com}>

"""Testing script for GamblersProblem environment implemented based on the following reference:
R. S. Sutton and A. G. Barto, Reinforcement Learning - An Introduction, 2nd ed., 2018 (Example 4.3: Gambler's problem).
"""

import logging
import numpy as np

from examples.gamblers_problem import *

# Setting logger
FORMAT = "[%(asctime)s|%(levelname)s|%(name)s] %(message)s"
DATE_FMT = "%H:%M:%S %Y-%m-%d"
log_level = logging.INFO
logging.basicConfig(format=FORMAT, datefmt=DATE_FMT, level=log_level)
logger = logging.getLogger('main')

# Environment configuration parameters.
NUM_STEPS = 100
PROB_HEAD = 0.5
INITIAL_CAPITAL = 10
WINNING_CAPITAL = 100

NUM_EPISODES = 1


def main():
    """Main routine of testing GamblersProblem gym-type environment.
    """
    config = {'num_steps': NUM_STEPS,
              'prob_head': PROB_HEAD,
              'initial_capital': INITIAL_CAPITAL,
              'winning_capital': WINNING_CAPITAL}
    env = gym.make(id='GamblersProblem-v0', config=config)
    for i in range(0, NUM_EPISODES):
        j = 0
        obs = env.reset()
        while True:
            env.render()
            action = env.action_space.sample()  # Means random agent

            # Amount of betting should be less than current capital.
            action[0] = min(action[0].item(), obs[0].item())

            obs, reward, done, info = env.step(action)
            log_step(i, j, obs, reward, done, info, action)
            j = j + 1
            if done:
                break
    env.close()


def log_step(episode: int, step: int, obs: np.ndarray, reward: float, done: bool, info: dict, action: np.ndarray):
    """Utility function for printing logs.

    :param episode: Episode number.
    :param step: Time-step.
    :param obs: Observation of the current environment.
    :param reward: Reward from the current environment.
    :param done: Indicates whether the episode ends or not.
    :param info: Contains auxiliary diagnostic information (helpful for debugging, and sometimes learning).
    :param action: An action provided by the agent.
    """
    capital = obs[0].item()
    bet = action[0].item()
    step_str = '{}-th step in {}-th episode / '.format(step, episode)
    obs_str = 'obs: {} / '.format(capital)
    reward_str = 'reward: {} / '.format(reward)
    done_str = 'done: {} / '.format(done)
    info_str = 'info: {} / '.format(info)
    action_str = 'action: {}'.format(bet)
    result_str = step_str + obs_str + reward_str + done_str + info_str + action_str
    logger.info(result_str)


if __name__ == "__main__":
    main()
