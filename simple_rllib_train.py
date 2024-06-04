import ray
from ray.rllib.algorithms.ppo import PPOConfig

# Initialize Ray
ray.init(ignore_reinit_error=True)

# Create a PPOConfig object and set the environment
config = PPOConfig().environment(env="SimpleEnv-v0")

# Build the PPOTrainer from the config
trainer = config.build()

# Train the PPO agent
for i in range(100):  # Number of training iterations
    result = trainer.train()
    print(f"Iteration {i}: {result['episode_reward_mean']}")

    # Optionally, save checkpoints
    if i % 10 == 0:
        checkpoint_path = trainer.save()
        print(f"Checkpoint saved at {checkpoint_path}")

# Optionally, restore the trainer from a checkpoint
# trainer.restore(checkpoint_path)

# Shutdown Ray
ray.shutdown()