# Author: Sae Hyong Park <labry@etri.re.kr>
# Author: Seungjae Shin <sjshin0505@{etri.re.kr, gmail.com}>

"""Testing script for GamblersProblem environment implemented based on the following reference:
R. S. Sutton and A. G. Barto, Reinforcement Learning - An Introduction, 2nd ed., 2018 (Example 4.3: Gambler's problem).
"""

import logging

from examples.gamblers_problem import *
import numpy as np

# from examples.gamblers_problem import *

# Setting logger
FORMAT = "[%(asctime)s|%(levelname)s|%(name)s] %(message)s"
DATE_FMT = "%H:%M:%S %Y-%m-%d"
log_level = logging.INFO
logging.basicConfig(format=FORMAT, datefmt=DATE_FMT, level=log_level)
logger = logging.getLogger('main')

# Environment configuration parameters.
NUM_STEPS = 100
PROB_HEAD = 0.6
INITIAL_CAPITAL = 10
WINNING_CAPITAL = 100

NUM_EPISODES = 10000


def main():
    """Main routine of testing GamblersProblem gym-type environment.
    """
    config = {'num_steps': NUM_STEPS,
              'prob_head': PROB_HEAD,
              'initial_capital': INITIAL_CAPITAL,
              'winning_capital': WINNING_CAPITAL}

    # metadata_ = {"render_modes": ["human", "rgb_array"], "render_fps": 4}


    env = gym.make(id='GamblersProblem-v0', config=config)
    total_reward = 0
    for i in range(0, NUM_EPISODES):

        logger.info("seed {}".format(i))
        obs, info = env.reset(seed=i, options=config)
        log_step(i, 0, obs, 0.0, False, False, info, {})
        # print(obs)
        capital = INITIAL_CAPITAL
        #print("1")
        j = 0
        # obs, info = env.reset(seed=126, options={})
        #print(obs, info)
        #logger.info(str(obs))
        pre_obs = obs[0]
        while True:
            env.render()

            # Amount of betting should be less than current capital.
            action = env.action_space.sample()
            # action = np.array([0.1])

            obs, reward, terminated, truncated, info = env.step(action)
            multiplier = min(pre_obs, 100-pre_obs)
            action = max(round(action[0] * multiplier), 1)
            pre_obs = obs[0]
            if reward < 0:
                reward = 0
            total_reward += reward
            if info["flip_result"] == "head":
                # print("test", capital, action)
                capital += action
            else:
                capital -= action

            log_step(i, j, obs, reward, terminated, truncated, info, action)
            j = j + 1
            if terminated:
                env.close()
                logger.info("\n")
                break

    logger.info("total reward: {}".format(total_reward/2))



def log_step(episode: int, step: int, obs: np.ndarray, reward: float, terminated: bool, truncated:bool, info: dict, action: int):
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
    if action != 0 :
        bet = action
    else:
        bet = 0
    step_str = '{}-th step in {}-th episode / '.format(step, episode)
    obs_str = 'obs: {} / '.format(capital)
    reward_str = 'reward: {} / '.format(reward)
    done_str = 'terminated: {} / '.format(terminated)
    truncated_str = 'truncated: {} / '.format(truncated)
    info_str = 'info: {} / '.format(info)
    action_str = 'action: {}'.format(bet)
    result_str = step_str + obs_str + reward_str + done_str + truncated_str + info_str + action_str
    logger.info(result_str)


if __name__ == "__main__":
    main()
