import random
from typing import Optional

import gymnasium as gym
from gymnasium.spaces import Box
from ray.tune import register_env


def coin_flip(prob_of_head = 0.4):
    '''
    Args:
        prob_of_head - probability of getting a Head on coin flip
    Returns:
        0 for Heads or 1 for Tails
    '''
    return 0 if random.random() < prob_of_head else 1

'''
Represents a Gambler's problem Gym Environment which provides a Fully observable
MDP
'''
class GamblersEnv(gym.Env):
    '''
    GamblerEnv represents the Gym Environment for the Gambler's problem environment
    '''
    metadata = {'render.modes': ['human']}

    def __init__(self, initial_cash_in_hand = 10, p_h = 0.40, goal_cash = 100):
        '''
        Constructor for the GamblersEnv class

        Args:
            initial_cash_in_hand - represents the cash that the player has initially
            prob_head - probability of getting a heads on a coin flip
            goal_cash - maximum cash obtained before the game ends
        '''
        self.metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
        self.initial_cash_in_hand = initial_cash_in_hand
        self.cash_in_hand = np.array([initial_cash_in_hand])
        self.prob_head = p_h
        self.goal_cash = goal_cash
        self.nS = self.goal_cash
        self.nA = min(self.cash_in_hand, 100 - self.cash_in_hand) + 1

        self.action_space = Box(low=1., high=100, shape=(1,), dtype=np.int_)
        self.observation_space = Box(low=0., high=100, shape=(1,), dtype=np.int_)
        # self.reset()

    def reset(self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
        ) -> (object, dict):
        '''
        Resets the environment
        Returns:
            observations containing player's current cash in hand
        '''
        self.seed = seed
        self.cash_in_hand = self.initial_cash_in_hand
        return self.get_obs(), {}

    def get_obs(self):
        '''
        Returns the player's cash in hand as the observation of the environment
        '''
        return (self.cash_in_hand)

    def render(self, mode='human'):
        '''
        Renders the environment
        '''
        # print("Current capital: {}".format(self.cash_in_hand))

    def sample_action(self):
        my_action = random.randint(1, self.cash_in_hand)
        # logger.info("my_action: {}".format(my_action))
        return my_action


    def step(self, action):
        '''
        Performs the given action
        Args:
            action : action from the action_space to be taking in the environment
        Returns:
            observation - returns current cash in hand of the player
            reward - reward obtained after taking the given action
            done - True if the episode is complete else False
        '''
        info = {}
        truncated = False
        terminated = False
        action = int(action)
        # print("action and cash_in_hand :", action, self.cash_in_hand)
        if action > self.cash_in_hand:
            action = self.cash_in_hand

        coinflip_result = coin_flip(self.prob_head)

        if coinflip_result:
            self.cash_in_hand = min(self.goal_cash, self.cash_in_hand + action)
            info = {'flip_result': 'heads'}
        else:
            self.cash_in_hand = max(0.0, self.cash_in_hand - action)
            info= {'flip_result': 'tails'}

        self.nA = self.cash_in_hand + 1

        if self.cash_in_hand >= self.goal_cash:
            terminated = True
            reward = 1
            info['msg'] = 'Wins the game because the capital becomes over {} dollars.'.format(self.cash_in_hand)
        else:
            terminated = False
            reward = 0

        if self.cash_in_hand <= 0:
            terminated = True
            reward = 0
            info['msg'] = 'Loses the game due to out of money.'

        return self.get_obs(), reward, terminated, truncated, info

import logging
import numpy as np

# Setting logger
FORMAT = "[%(asctime)s|%(levelname)s|%(name)s] %(message)s"
DATE_FMT = "%H:%M:%S %Y-%m-%d"
log_level = logging.INFO
logging.basicConfig(format=FORMAT, datefmt=DATE_FMT, level=log_level)
logger = logging.getLogger('main')

# Environment configuration parameters.
NUM_STEPS = 10
PROB_HEAD = 0.5
INITIAL_CAPITAL = 10
WINNING_CAPITAL = 100

NUM_EPISODES = 1000

def main():
    """Main routine of testing GamblersProblem gym-type environment.
    """
    gym.register(id='GamblersEnv-v0', entry_point='my_test5:GamblersEnv')
    register_env("GamblersEnv-v0", GamblersEnv)

    config = {'num_steps': NUM_STEPS,
              'prob_head': PROB_HEAD,
              'initial_capital': INITIAL_CAPITAL,
              'winning_capital': WINNING_CAPITAL}

    # metadata_ = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    env = gym.make(id='GamblersEnv-v0')
    # print("env", env)

    total_reward = 0

    for i in range(0, NUM_EPISODES):
        obs, info = env.reset(seed=i, options={})
        log_step(0, 0, [obs], 0.0, False, False, info, {})
        # print(obs)
        capital = INITIAL_CAPITAL
        j = 0
        # obs, info = env.reset(seed=126, options={})
        #print(obs, info)
        #logger.info(str(obs))
        while True:
            env.render()
            # action = env.action_space.sample()  # Means random agent
            action = env.sample_action()
            # Amount of betting should be less than current capital.
            action = min(action, WINNING_CAPITAL - capital)
            #print(action, obs[0].item(), WINNING_CAPITAL-capital)

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            # if info["flip_result"] == "head":
            #     capital += action
            # else:
            #     capital -= action

            log_step(i, j, obs, reward, terminated, truncated, info, action)
            j = j + 1
            if terminated:
                logger.info("\n")
                break
    env.close()
    logger.info("Total reward: {}".format(total_reward))

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
    capital = obs
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
