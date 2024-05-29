from ray import tune

tune.run("PPO",
         config={"env": "CartPole-v1",
                 # other configurations go here, if none provided, then default configurations will be used
                 "evaluation_interval": 2,
                 "evaluation_duration_unit": 10,
                 "storage_path": "C://Users//labry//git//GymProxy//cartpole",
                 },
         checkpoint_freq=5,
         )