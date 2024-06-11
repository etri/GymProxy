import time
from threading import Thread

import gymnasium as gym
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.utils.metrics import (
    ENV_RUNNER_RESULTS,
)
from ray.tune import register_env

EPISODE_REWARD_MEAN = "episode_reward_mean"

NUM_STEPS = 100
PROB_HEAD = 0.5
INITIAL_CAPITAL = 10
WINNING_CAPITAL = 100

config = {'num_steps': NUM_STEPS,
          'prob_head': PROB_HEAD,
          'initial_capital': INITIAL_CAPITAL,
          'winning_capital': WINNING_CAPITAL}

from examples.gamblers_problem import *
from examples.gamblers_problem.gym_env import GamblersProblem

def my_class(config=None):
    env = GamblersProblem()
    return env

# gym.register(id='GamblersProblem-v0', entry_point='examples.gamblers_problem:GamblersProblem')
register_env("GamblersProblem-v0", my_class)

# Create an RLlib Algorithm instance from a PPOConfig to learn how to
# act in the above environment.
config = (
    PPOConfig().environment(
        # Env class to use (here: our gym.Env sub-class from above).
        env="GamblersProblem-v0",
        # Config dict to be passed to our custom env's constructor.
        env_config=config,
    )
    # .evaluation(
    #     evaluation_interval=0,
    #     evaluation_num_env_runners=0,
    # )
    # Parallelize environment rollouts.
    .env_runners(num_env_runners=0, num_rollout_workers=0, num_envs_per_env_runner=1)
)
algo = config.build()

# Train for n iterations and report results (mean episode rewards).
# Since we have to guess 10 times and the optimal reward is 0.0
# (exact match between observation and action value),
# we can expect to reach an optimal episode reward of 0.0.
for i in range(10):
    results = algo.train()
    print(f"Iter: {i}; avg. reward={results[ENV_RUNNER_RESULTS][EPISODE_REWARD_MEAN]}")

# Perform inference (action computations) based on given env observations.
# Note that we are using a slightly simpler env here (-3.0 to 3.0, instead
# of -5.0 to 5.0!), however, this should still work as the agent has
# (hopefully) learned to "just always repeat the observation!".
env = GamblersProblem()
# Get the initial observation (some value between -10.0 and 10.0).
obs, info = env.reset()
done = False
total_reward = 0.0
# Play one episode.
while not done:
    # Compute a single action, given the current observation
    # from the environment.
    action = algo.compute_single_action(obs)
    # Apply the computed action in the environment.
    obs, reward, done, truncated, info = env.step(action)
    # Sum up rewards for reporting purposes.
    total_reward += reward
# Report results.
print(f"Played 1 episode; total-reward={total_reward}")