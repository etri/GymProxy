from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune import register_env
from ray import air
from ray import tune

from examples.gamblers_problem.gym_env import GamblersProblem
from examples.gamblers_problem import *

NUM_STEPS = 100
PROB_HEAD = 0.5
INITIAL_CAPITAL = 10
WINNING_CAPITAL = 100

config = PPOConfig()
# Print out some default values.

# Set the config object's env.
config = config.environment(GamblersProblem)
tune.Tuner(
    "DQN",
    run_config=air.RunConfig(stop={"training_iteration": 1}),
    param_space=config.to_dict(),
).fit()