from ray.rllib.algorithms.ppo.ppo import PPOConfig
from ray.tune import register_env

from examples.gamblers_problem import GamblersProblem

# gymnasium.register(id='GamblersProblem-v0', entry_point='examples.gamblers_problem:GamblersProblem')
register_env("GamblersProblem-v0", GamblersProblem)

# Step 1: Create a PPOConfig object
config = PPOConfig().environment("GamblersProblem-v0")

# Step 2: Build the PPOTrainer from the config
agent = config.build()

# Step 3: Restore the trainer from a checkpoint
# checkpoint_path = "C://Users//ADMIN//ray_results//PPO_2024-06-05_20-45-08//PPO_GamblersProblem-v0_0f582_00000_0_2024-06-05_20-45-09//checkpoint_000003//"
# checkpoint_path = "C://Users//ADMIN//ray_results//PPO_2024-06-09_18-58-23//PPO_GamblersProblem-v0_cf047_00000_0_2024-06-09_18-58-23//checkpoint_000013//"
# checkpoint_path = "C://Users//ADMIN//ray_results//PPO_2024-06-09_21-40-05//PPO_GamblersProblem-v0_65d28_00000_0_2024-06-09_21-40-05//checkpoint_000013"
# checkpoint_path ="C://Users//ADMIN//ray_results//PPO_2024-06-10_23-55-43//PPO_GamblersProblem-v0_831dc_00000_0_2024-06-10_23-55-43//checkpoint_000008//"
# checkpoint_path = "C:/Users/labry/ray_results/PPO_2024-06-12_10-56-24/PPO_GamblersProblem-v0_f96f8_00000_0_2024-06-12_10-56-24/checkpoint_000013/"
# trainer.restore(checkpoint_path)
# checkpoint_path ="C:/Users/labry/ray_results/PPO_2024-06-12_17-16-10/PPO_GamblersProblem-v0_067f9_00000_0_2024-06-12_17-16-10/checkpoint_000001"
checkpoint_path = "C:/Users/ADMIN/ray_results/PPO_2024-06-15_20-45-40/PPO_GamblersProblem-v0_ca369_00000_0_2024-06-15_20-45-40/checkpoint_000009/"

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
NUM_EPISODES = 1000

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

logger.info("total reward: {}".format(total_reward))