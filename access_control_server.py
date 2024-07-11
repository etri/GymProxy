#!/usr/bin/env python

import argparse
import os

import gymnasium
import numpy as np
import ray
from ray import air, tune
from ray.air.constants import TRAINING_ITERATION
from ray.rllib.env.policy_server_input import PolicyServerInput
from ray.rllib.utils.metrics import (
    NUM_ENV_STEPS_SAMPLED,
    NUM_AGENT_STEPS_SAMPLED,
)
from ray.tune.logger import pretty_print
from ray.tune.registry import get_trainable_cls

SERVER_ADDRESS = "localhost"
SERVER_BASE_PORT = 9900  # + worker-idx - 1

CHECKPOINT_FILE = "last_checkpoint_{}.out"

def get_cli_args():
    """Create CLI parser and return parsed arguments"""
    parser = argparse.ArgumentParser()

    # parser.add_argument(
    #     "--stop-iters", type=int, default=10, help="Number of iterations to train."
    # )
    # Example-specific args.
    parser.add_argument(
        "--port",
        type=int,
        default=SERVER_BASE_PORT,
        help="The base-port to use (on localhost). " f"Default is {SERVER_BASE_PORT}.",
    )
    parser.add_argument(
        "--callbacks-verbose",
        action="store_true",
        help="Activates info-messages for different events on "
        "server/client (episode steps, postprocessing, etc..).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="The number of workers to use. Each worker will create "
        "its own listening socket for incoming experiences.",
    )
    parser.add_argument(
        "--no-restore",
        action="store_true",
        help="Do not restore from a previously saved checkpoint (location of "
        "which is saved in `last_checkpoint_[algo-name].out`).",
    )

    # General args.
    parser.add_argument(
        "--run",
        default="PPO",
        choices=["PPO"],
        help="The RLlib-registered algorithm to use.",
    )
    parser.add_argument("--num-cpus", type=int, default=3)
    parser.add_argument(
        "--framework",
        choices=["torch"],
        default="torch",
        help="The DL framework specifier.",
    )
    parser.add_argument(
        "--use-lstm",
        action="store_true",
        help="Whether to auto-wrap the model with an LSTM. Only valid option for "
        "--run=[IMPALA|PPO|R2D2]",
    )
    parser.add_argument(
        "--stop-iters", type=int, default=1, help="Number of iterations to train."
    )
    parser.add_argument(
        "--stop-timesteps",
        type=int,
        default=500000,
        help="Number of timesteps to train.",
    )
    parser.add_argument(
        "--stop-reward",
        type=float,
        default=80.0,
        help="Reward at which we stop training.",
    )
    parser.add_argument(
        "--as-test",
        action="store_true",
        help="Whether this script should be run as a test: --stop-reward must "
        "be achieved within --stop-timesteps AND --stop-iters.",
    )
    parser.add_argument(
        "--no-tune",
        action="store_true",
        help="Run without Tune using a manual train loop instead. Here,"
        "there is no TensorBoard support.",
    )
    parser.add_argument(
        "--local-mode",
        action="store_true",
        help="Init Ray in local mode for easier debugging.",
    )

    args = parser.parse_args()
    print(f"Running with following CLI args: {args}")
    return args

if __name__ == "__main__":
    args = get_cli_args()
    ray.init(local_mode=args.local_mode)

    # `InputReader` generator (returns None if no input reader is needed on
    # the respective worker).
    def _input(ioctx):
        # We are remote worker or we are local worker with num_env_runners=0:
        # Create a PolicyServerInput.
        if ioctx.worker_index > 0 or ioctx.worker.num_workers == 0:
            return PolicyServerInput(
                ioctx,
                SERVER_ADDRESS,
                args.port + ioctx.worker_index - (1 if ioctx.worker_index > 0 else 0),
            )
        # No InputReader (PolicyServerInput) needed.
        else:
            return None

    # Algorithm config. Note that this config is sent to the client only in case
    # the client needs to create its own policy copy for local inference.
    config = (
        get_trainable_cls(args.run).get_default_config()
        # Indicate that the Algorithm we setup here doesn't need an actual env.
        # Allow spaces to be determined by user (see below).
        .environment(
            env=None,
            # Define observation and action spaces for the Access Control Queue
            observation_space=gymnasium.spaces.Box(0, np.inf, shape=(2,), dtype=np.int32),
            action_space=gymnasium.spaces.Discrete(2),  # Accept (1) or Reject (0) customer
        )
        # DL framework to use.
        .framework(args.framework)
        # Create a "chatty" client/server or not.
        # .callbacks(MyCallbacks if args.callbacks_verbose else None)
        # Use the `PolicyServerInput` to generate experiences.
        .offline_data(input_=_input)
        # Use n worker processes to listen on different ports.
        .env_runners(
            num_env_runners=args.num_workers,
            # Connectors are not compatible with the external env.
            enable_connectors=False,
        )
        # Disable OPE, since the rollouts are coming from online clients.
        .evaluation(off_policy_estimation_methods={})
        # Set to INFO so we'll see the server's actual address:port.
        .debugging(log_level="INFO")
    )
    # Disable RLModules because they need connectors
    # TODO (Sven): Deprecate ExternalEnv (via EnvRunner path) and reenable connectors
    #  and RL Modules here.
    # config.api_stack(enable_rl_module_and_learner=False)

    # PPO.
    if args.run == "PPO":
        # Example of using PPO (does NOT support off-policy actions).
        config.update_from_dict(
            {
                "rollout_fragment_length": 2000,
                "train_batch_size": 4000,
                "model": {"use_lstm": args.use_lstm},
            }
        )

    checkpoint_path = CHECKPOINT_FILE.format(args.run)
    # Attempt to restore from checkpoint, if possible.
    if not args.no_restore and os.path.exists(checkpoint_path):
        checkpoint_path = open(checkpoint_path).read()
    else:
        checkpoint_path = None

    # Manual training loop (no Ray tune).
    if args.no_tune:
        algo = config.build()

        if checkpoint_path:
            print("Restoring from checkpoint path", checkpoint_path)
            algo.restore(checkpoint_path)

        # Serving and training loop.
        ts = 0
        print("stop_iters =", args.stop_iters)
        for _ in range(args.stop_iters):
            results = algo.train()
            print(pretty_print(results))
            checkpoint = algo.save().checkpoint
            print("Last checkpoint", checkpoint)
            with open(checkpoint_path, "w") as f:
                f.write(checkpoint.path)
            if (
                results["episode_reward_mean"] >= args.stop_reward
                or ts >= args.stop_timesteps
            ):
                break
            ts += results[NUM_ENV_STEPS_SAMPLED]

        algo.stop()

    # Run with Tune for auto env and algo creation and TensorBoard.
    else:
        print("Ignoring restore even if previous checkpoint is provided...")
        print("args.stop_iters =", args.stop_iters)
        stop = {
            TRAINING_ITERATION: args.stop_iters,
            NUM_ENV_STEPS_SAMPLED: args.stop_timesteps,
            "episode_reward_mean": args.stop_reward,
        }

        tune.Tuner(
            args.run, param_space=config, run_config=air.RunConfig(stop=stop, verbose=2)
        ).fit()
