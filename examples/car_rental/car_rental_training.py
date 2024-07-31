import logging
import time

import numpy as np
from ray.rllib.algorithms.ppo.ppo import PPOConfig
from ray.tune import register_env

from examples.car_rental import CarRental

# Constants and Configuration
NUM_STEPS = 100
MAX_NUM_CARS_PER_LOC = 20
RENTAL_CREDIT = 10.
CAR_MOVING_COST = 2.
RENTAL_RATE_0 = 4
RENTAL_RATE_1 = 3
RETURN_RATE_0 = 3
RETURN_RATE_1 = 2

# NUM_EPISODES = 100

env_config = {'num_steps': NUM_STEPS,
              'max_num_cars_per_loc': MAX_NUM_CARS_PER_LOC,
              'rental_credit': RENTAL_CREDIT,
              'car_moving_cost': CAR_MOVING_COST,
              'rental_rate_0': RENTAL_RATE_0,
              'rental_rate_1': RENTAL_RATE_1,
              'return_rate_0': RETURN_RATE_0,
              'return_rate_1': RETURN_RATE_1}

# Register the environment
register_env("CarRental-v0", lambda config: CarRental(config))

# Step 1: Create a PPOConfig object
config = PPOConfig().environment("CarRental-v0", env_config=env_config).rollouts(num_rollout_workers=1)

config.update_from_dict(
    {
        "callback_verbose": False,
        # "rollout_fragment_length": 1000,
        "train_batch_size": 4000,
        "model": {"use_lstm": False},
        "seed": 126,
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

def log_step(episode: int, step: int, obs: np.ndarray, reward: float, done: bool, info: dict, action: tuple):
    """Utility function for printing logs.

    :param episode: Episode number.
    :param step: Time-step.
    :param obs: Observation of the current environment.
    :param reward: Reward from the current environment.
    :param done: Indicates whether the episode ends or not.
    :param info: Contains auxiliary diagnostic information (helpful for debugging, and sometimes learning).
    :param action: An action provided by the agent.
    """
    available_cars = [MAX_NUM_CARS_PER_LOC] * 2

    available_cars[0] = obs[0].item()
    available_cars[1] = obs[1].item()

    src = action[0]
    dst = 1 - src
    n_moving = action[1].item()

    available_cars[src] = max(available_cars[src] - n_moving, 0)
    available_cars[dst] = min(available_cars[dst] + n_moving, MAX_NUM_CARS_PER_LOC)


    source_loc = action[0]
    n_moving = action[1].item()
    step_str = '{}-th step in {}-th episode / '.format(step, episode)
    obs_str = 'obs: {} / '.format((available_cars[0], available_cars[1]))
    reward_str = 'reward: {} / '.format(reward)
    done_str = 'done: {} / '.format(done)
    info_str = 'info: {} / '.format(info)
    action_str = 'action: {}'.format((source_loc, n_moving))
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
