import gymnasium
from ray.tune import register_env

from examples.gamblers_problem import GamblersProblem

# gymnasium.register(id='GamblersProblem-v0', entry_point='examples.gamblers_problem:GamblersProblem')
register_env("GamblersProblem-v0", GamblersProblem)
# env = gymnasium.make(id='InventoryEnv-v0')
# print("env", env)
# obs, info = env.reset(seed=126, options={})

from ray import tune

NUM_STEPS = 100
PROB_HEAD = 0.5
INITIAL_CAPITAL = 10
WINNING_CAPITAL = 100

config = {'num_steps': NUM_STEPS,
          'prob_head': PROB_HEAD,
          'initial_capital': INITIAL_CAPITAL,
          'winning_capital': WINNING_CAPITAL}

tune.run("PPO",
         config={"env": "GamblersProblem-v0",    # Instead of strings e.g. "CartPole-v1", we pass the custom env class
                 "env_config": {"config":config},
                 "num_workers":0,
                 "evaluation_interval": 100,
                 # Each episode uses different shop params. Need lots of samples to gauge agent's performance
                 "evaluation_duration_unit": 1000,
                 #"clip_actions": True,
                 },
         # keep_checkpoints_num=100,
         checkpoint_freq=10,
         )