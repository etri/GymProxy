from ray.tune import register_env

from examples.car_rental import CarRental

register_env("CarRental-v0", CarRental)

from ray import tune

NUM_STEPS = 100
MAX_NUM_CARS_PER_LOC = 20
RENTAL_CREDIT = 10.
CAR_MOVING_COST = 2.
RENTAL_RATE_0 = 4
RENTAL_RATE_1 = 3
RETURN_RATE_0 = 3
RETURN_RATE_1 = 2

config = {'num_steps': NUM_STEPS,
          'max_num_cars_per_loc': MAX_NUM_CARS_PER_LOC,
          'rental_credit': RENTAL_CREDIT,
          'car_moving_cost': CAR_MOVING_COST,
          'rental_rate_0': RENTAL_RATE_0,
          'rental_rate_1': RENTAL_RATE_1,
          'return_rate_0': RETURN_RATE_0,
          'return_rate_1': RETURN_RATE_1}

def trial_name_creator(trial):
    return f"{trial.trainable_name}_{trial.trial_id}"

tune.run("PPO",
         config={"env": "CarRental-v0",    # Instead of strings e.g. "CartPole-v1", we pass the custom env class
                 "env_config": {'num_steps': NUM_STEPS,
                  'max_num_cars_per_loc': MAX_NUM_CARS_PER_LOC,
                  'rental_credit': RENTAL_CREDIT,
                  'car_moving_cost': CAR_MOVING_COST,
                  'rental_rate_0': RENTAL_RATE_0,
                  'rental_rate_1': RENTAL_RATE_1,
                  'return_rate_0': RETURN_RATE_0,
                  'return_rate_1': RETURN_RATE_1
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
         checkpoint_freq=20,
         local_dir="C:/ray_results",
         trial_dirname_creator=trial_name_creator,
         )