from ray.tune import register_env

from examples.access_control_q import AccessControlQueue

register_env("AccessControlQueue-v0", AccessControlQueue)

from ray import tune

NUM_STEPS = 100
NUM_SERVERS = 10
SERVER_FREE_PROB = 0.06
PRIORITIES = [1., 2., 4., 8.]

config = {'num_steps': NUM_STEPS,
          'num_servers': NUM_SERVERS,
          'server_free_probability': SERVER_FREE_PROB,
          'priorities': PRIORITIES}

def trial_name_creator(trial):
    return f"{trial.trainable_name}_{trial.trial_id}"

tune.run("PPO",
         config={"env": "AccessControlQueue-v0",    # Instead of strings e.g. "CartPole-v1", we pass the custom env class
                 "env_config": {'num_steps': NUM_STEPS,
                  'num_servers': NUM_SERVERS,
                  'server_free_probability': SERVER_FREE_PROB,
                  'priorities': PRIORITIES
                    },
                 "num_workers":0,
                 "evaluation_interval": 100,
                 # Each episode uses different shop params. Need lots of samples to gauge agent's performance
                 "evaluation_duration_unit": 1000,
                 "normalize_actions": False,
                 # "seed": 42,
                 #"clip_actions": True,
                 },
         # keep_checkpoints_num=100,
         checkpoint_freq=10,
         local_dir="C:/ray_results",
         trial_dirname_creator=trial_name_creator,
         )