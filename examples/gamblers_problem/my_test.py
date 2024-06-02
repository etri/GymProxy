from ray import tune

NUM_STEPS = 100
PROB_HEAD = 0.5
INITIAL_CAPITAL = 10
WINNING_CAPITAL = 100

config = {'num_steps': NUM_STEPS,
          'prob_head': PROB_HEAD,
          'initial_capital': INITIAL_CAPITAL,
          'winning_capital': WINNING_CAPITAL}

from gym_env import GamblersProblem

gamblers_problem = GamblersProblem(config)

tune.run("PPO",
         config={"env": gamblers_problem,    # Instead of strings e.g. "CartPole-v1", we pass the custom env class
                 "env_config": {"config":config},
                 "evaluation_interval": 1000,
                 # Each episode uses different shop params. Need lots of samples to gauge agent's performance
                 "evaluation_duration_unit": 10000,
                 },
         checkpoint_freq=10,
         )