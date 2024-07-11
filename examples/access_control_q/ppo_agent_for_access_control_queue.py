from ray.rllib.algorithms.ppo.ppo import PPOConfig
from ray.tune import register_env

from examples.access_control_q import AccessControlQueue

NUM_STEPS = 100
NUM_SERVERS = 10
SERVER_FREE_PROB = 0.06
PRIORITIES = [1., 2., 4., 8.]
config = {'num_steps': NUM_STEPS,
          'num_servers': NUM_SERVERS,
          'server_free_probability': SERVER_FREE_PROB,
          'priorities': PRIORITIES}
# gymnasium.register(id='GamblersProblem-v0', entry_point='examples.gamblers_problem:GamblersProblem')
register_env("AccessControlQueue-v0", AccessControlQueue)

# Step 1: Create a PPOConfig object
config = PPOConfig().environment("AccessControlQueue-v0", env_config=config)

# Step 2: Build the PPOTrainer from the config
agent = config.build()

# Step 3: Restore the trainer from a checkpoint
# checkpoint_path ="C:/Users/labry/ray_results/PPO_2024-06-12_17-16-10/PPO_GamblersProblem-v0_067f9_00000_0_2024-06-12_17-16-10/checkpoint_000001"
# checkpoint_path = "C:/ray_results/PPO_2024-07-04_13-40-38/PPO_8f7d0_00000/checkpoint_000009"
checkpoint_path = "C:/Users/ADMIN/ray_results/PPO_2024-07-05_20-25-04/PPO_None_39cc3_00000_0_2024-07-05_20-25-04/checkpoint_000000"

agent.load_checkpoint(checkpoint_path)
# print("agent:", agent)

import gymnasium as gym
import logging
import numpy as np

# Setting logger
FORMAT = "[%(asctime)s|%(levelname)s|%(name)s] %(message)s"
DATE_FMT = "%H:%M:%S %Y-%m-%d"
log_level = logging.INFO
logging.basicConfig(format=FORMAT, datefmt=DATE_FMT, level=log_level)
logger = logging.getLogger('main')




def log_step(episode: int, step: int, obs: np.ndarray, reward: float, done: bool, info: dict, action: int):
    """Utility function for printing logs.

    :param episode: Episode number.
    :param step: Time-step.
    :param obs: Observation of the current environment.
    :param reward: Reward from the current environment.
    :param done: Indicates whether the episode ends or not.
    :param info: Contains auxiliary diagnostic information (helpful for debugging, and sometimes learning).
    :param action: An action provided by the agent.
    """
    priority = obs[0].item()
    num_free_servers = obs[1].item()
    step_str = '{}-th step in {}-th episode / '.format(step, episode)
    obs_str = 'obs: {} / '.format((priority, num_free_servers))
    reward_str = 'reward: {} / '.format(reward)
    done_str = 'done: {} / '.format(done)
    info_str = 'info: {} / '.format(info)
    action_str = 'action: {}'.format(True if action else False)
    result_str = step_str + obs_str + reward_str + done_str + info_str + action_str
    logger.info(result_str)


env = gym.make("AccessControlQueue-v0")

total_reward = 0
NUM_EPISODES = 100

for i in range(0, NUM_EPISODES):
    obs, info = env.reset(seed=i)
    pre_obs = obs[0]
    logger.info("obs {} and info: {}".format(obs, info))
    while True:
        action = agent.compute_single_action(observation=obs, info=info)
        # logger.info("action: {}".format(action))
        obs, reward, terminated, truncated, info = env.step(action)
        # action = max(round(action[0] * pre_obs), 1)
        pre_obs = obs[0]
        # logger.info("action {} and info: {}".format(action, info))
        # if reward < 0:
        #     reward = 0
        total_reward += reward
        log_step(i, 1, obs, reward, terminated, info, action)
        env.render()
        if terminated:
            break
    env.close()

logger.info("total reward: {}".format(total_reward))