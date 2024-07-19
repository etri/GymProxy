import gymnasium
from gymnasium.spaces import Box
from ray.tune import register_env

from examples.gamblers_problem import GamblersProblem

from gymnasium.wrappers import RescaleAction
import numpy as np

# gymnasium.register(id='GamblersProblem-v0', entry_point='examples.gamblers_problem:GamblersProblem')
register_env("GamblersProblem-v0", GamblersProblem)
# env = gymnasium.make(id='InventoryEnv-v0')
# print("env", env)
# obs, info = env.reset(seed=126, options={})

from ray import tune

NUM_STEPS = 100
PROB_HEAD = 0.6
INITIAL_CAPITAL = 10
WINNING_CAPITAL = 100

config = {'num_steps': NUM_STEPS,
          'prob_head': PROB_HEAD,
          'initial_capital': INITIAL_CAPITAL,
          'winning_capital': WINNING_CAPITAL}

options = [
    {"num_steps": 100, "prob_head": 0.6, "initial_capital": 10, "winning_capital": 100},
    {"num_steps": 100, "prob_head": 0.6, "initial_capital": 10, "winning_capital": 100},
]

def trial_name_creator(trial):
    return f"{trial.trainable_name}_{trial.trial_id}"

tune.run("PPO",
         config={"env": "GamblersProblem-v0",    # Instead of strings e.g. "CartPole-v1", we pass the custom env class
                 "env_config": {'num_steps': NUM_STEPS,
                                  'prob_head': PROB_HEAD,
                                  'initial_capital': INITIAL_CAPITAL,
                                  'winning_capital': WINNING_CAPITAL,
                                },
                 # "env_reset": {
                 #     "options": tune.grid_search(options)  # Add reset options dict
                 # },
                 "num_workers":0,
                 "evaluation_interval": 20,
                 # Each episode uses different shop params. Need lots of samples to gauge agent's performance
                 "evaluation_duration_unit": 1000,
                 "normalize_actions": False,
                 # "seed": 42,
                 #"clip_actions": True,
                 },
         # keep_checkpoints_num=100,
         checkpoint_freq=20,
         local_dir="C:/ray_results",
         trial_dirname_creator=trial_name_creator,
         )
