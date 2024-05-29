from ray.rllib.algorithms.ppo.ppo import PPOConfig

# Step 1: Create a PPOConfig object
config = PPOConfig().environment("CartPole-v1")

# Step 2: Build the PPOTrainer from the config
agent = config.build()

# Step 3: Restore the trainer from a checkpoint
checkpoint_path = "C://Users//labry//ray_results//PPO_2024-05-28_17-35-49//PPO_CartPole-v1_49590_00000_0_2024-05-28_17-35-49//checkpoint_000003//"  # Adjust the path to your checkpoint
# trainer.restore(checkpoint_path)
agent.load_checkpoint(checkpoint_path)
print(agent)

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

env = gym.make("CartPole-v1")
obs, info = env.reset()
while True:
    action = agent.compute_single_action(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    log_step(1, 1, obs, reward, terminated, truncated, info, action)
    env.render()
    if terminated:
        break
env.close()

