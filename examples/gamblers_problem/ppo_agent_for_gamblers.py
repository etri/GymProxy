from ray.rllib.algorithms.ppo.ppo import PPOConfig
from ray.tune import register_env

from examples.gamblers_problem import GamblersProblem

NUM_STEPS = 100
PROB_HEAD = 0.6
INITIAL_CAPITAL = 10
WINNING_CAPITAL = 100

config = {'num_steps': NUM_STEPS,
          'prob_head': PROB_HEAD,
          'initial_capital': INITIAL_CAPITAL,
          'winning_capital': WINNING_CAPITAL}
# gymnasium.register(id='GamblersProblem-v0', entry_point='examples.gamblers_problem:GamblersProblem')
register_env("GamblersProblem-v0", GamblersProblem)

# Step 1: Create a PPOConfig object
config = PPOConfig().environment("GamblersProblem-v0", env_config=config)

# Step 2: Build the PPOTrainer from the config
agent = config.build()

# Step 3: Restore the trainer from a checkpoint
# checkpoint_path ="C:/Users/labry/ray_results/PPO_2024-06-12_17-16-10/PPO_GamblersProblem-v0_067f9_00000_0_2024-06-12_17-16-10/checkpoint_000001"
checkpoint_path = "C:/ray_results/PPO_2024-06-18_21-38-20/PPO_a4f9b_00000/checkpoint_000008"

agent.load_checkpoint(checkpoint_path)
print("agent:", agent)

import gymnasium as gym
import logging
import numpy as np

# Setting logger
FORMAT = "[%(asctime)s|%(levelname)s|%(name)s] %(message)s"
DATE_FMT = "%H:%M:%S %Y-%m-%d"
log_level = logging.INFO
logging.basicConfig(format=FORMAT, datefmt=DATE_FMT, level=log_level)
logger = logging.getLogger('main')




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
    # print("log_step")
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


env = gym.make("GamblersProblem-v0")

total_reward = 0
NUM_EPISODES = 10000

for i in range(0, NUM_EPISODES):
    obs, info = env.reset(seed=i)
    pre_obs = obs[0]
    logger.info("obs {} and info: {}".format(obs, info))
    while True:
        action = agent.compute_single_action(observation=obs, info=info)
        obs, reward, terminated, truncated, info = env.step(action)
        action = max(round(action[0] * pre_obs), 1)
        pre_obs = obs[0]
        # logger.info("action {} and info: {}".format(action, info))
        if reward < 0:
            reward = 0
        total_reward += reward
        log_step(i, 1, obs, reward, terminated, truncated, info, action)
        env.render()
        if terminated:
            break
    env.close()

logger.info("total reward: {}".format(total_reward/2))