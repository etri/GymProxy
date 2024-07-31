import time

from ray.rllib.algorithms.ppo.ppo import PPOConfig
from ray.tune import register_env

from examples.access_control_q import AccessControlQueue
import gymnasium as gym
import logging
import numpy as np

# Constants and Configuration
NUM_STEPS = 100
NUM_SERVERS = 10
SERVER_FREE_PROB = 0.06
PRIORITIES = [1., 2., 4., 8.]

# NUM_EPISODES = 100

env_config = {'num_steps': NUM_STEPS,
          'num_servers': NUM_SERVERS,
          'server_free_probability': SERVER_FREE_PROB,
          'priorities': PRIORITIES}

# Register the environment
register_env("AccessControlQueue-v0", lambda config: AccessControlQueue(config))

# Step 1: Create a PPOConfig object
config = PPOConfig().environment("AccessControlQueue-v0", env_config=env_config).rollouts(num_rollout_workers=1)

config.update_from_dict(
    {
        "callback_verbose": False,
        # "rollout_fragment_length": 1000,
        "train_batch_size": 4000,
        "model": {"use_lstm": False},
    }
)

# Step 2: Build the PPOTrainer from the config
agent = config.build()

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

# Training loop
NUM_TRAINING_ITERATIONS = 25
SAVE_INTERVAL = 100

for iteration in range(NUM_TRAINING_ITERATIONS):
    start_time = time.time()
    result = agent.train()
    end_time = time.time()
    duration = end_time - start_time
    logger.info("(Duration: {:.2f} seconds Training iteration {} )".format(duration, iteration))

    if (iteration + 1) % SAVE_INTERVAL == 0:
        checkpoint_dir = agent.save("checkpoints")
        logger.info(f"Checkpoint saved at iteration {iteration + 1} to {checkpoint_dir}")

# Run the trained agent in the environment
# total_reward = 0
# NUM_EPISODES = 100
#
# env = gym.make("CarRental-v0", env_config=env_config)
#
# for i in range(NUM_EPISODES):
#     obs, info = env.reset(seed=i)
#     step = 0
#     logger.info("obs {} and info: {}".format(obs, info))
#     while True:
#         action = agent.compute_single_action(observation=obs)
#         obs, reward, terminated, truncated, info = env.step(action)
#         total_reward += reward
#         step += 1
#         log_step(i, step, obs, reward, terminated, info, action)
#         env.render()
#         if terminated:
#             break
#     env.close()
#
# logger.info("Total reward: {}".format(total_reward))
