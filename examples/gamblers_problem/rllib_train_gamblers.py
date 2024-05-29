import gymnasium as gym
from ray import tune

tune.run("PPO",
         config={"env": "GamblersProblem-v0",
                 # other configurations go here, if none provided, then default configurations will be use
                 },
         # evaluation_interval= 2,
         # evaluation_duration_unit= 10,
         checkpoint_freq= 5,
         storage_path = "C:/Users/labry/git/GymProxy/gamblers",
         )
