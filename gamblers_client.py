import argparse
import numpy as np
from ray.rllib.env.policy_client import PolicyClient


class GamblersProblemSimulator:
    """External environment class that actually simulates gambler's problem.
    """

    def __init__(self, num_steps, prob_head, initial_capital, winning_capital):
        """Constructor.

        :param num_steps: Number of time-steps.
        :param prob_head: Probability that a coin flip becomes head.
        :param initial_capital: Initial capital.
        :param winning_capital: Capital for winning the game.
        """
        self._num_steps = num_steps
        self._p_h = prob_head
        self._s = initial_capital
        self._s_win = winning_capital
        self._reward = 0.
        self._t = 0

    def reset(self, seed=None):
        """Resets the environment."""
        if seed is not None:
            np.random.seed(seed)
        self._s = 10
        self._reward = 0.
        self._t = 0
        return np.array([self._s], dtype=np.int_)

    def step(self, action):
        """Performs one step in the environment."""
        terminated = False
        bet = min(action, self._s_win - self._s)

        r = np.random.rand()
        if r < self._p_h:
            self._s += bet
        else:
            self._s -= bet

        if self._s >= self._s_win:
            self._reward = 1.
            terminated = True
        elif self._s <= 0:
            self._reward = -1.
            terminated = True

        self._t += 1
        terminated = terminated or (self._t >= self._num_steps)
        return np.array([self._s], dtype=np.int_), self._reward, terminated, {}


parser = argparse.ArgumentParser()
parser.add_argument(
    "--no-train", action="store_true", help="Whether to disable training."
)
parser.add_argument(
    "--inference-mode", type=str, default="local", choices=["local", "remote"]
)
parser.add_argument(
    "--off-policy",
    action="store_true",
    help="Whether to compute random actions instead of on-policy "
         "(Policy-computed) ones.",
)
parser.add_argument(
    "--stop-reward",
    type=float,
    default=9999,
    help="Stop once the specified reward is reached.",
)
parser.add_argument(
    "--port", type=int, default=9900, help="The port to use (on localhost)."
)

if __name__ == "__main__":
    args = parser.parse_args()

    # Initialize the Gambler's Problem simulator
    env = GamblersProblemSimulator(
        num_steps=100,
        prob_head=0.5,
        initial_capital=10,
        winning_capital=100
    )

    # Connect to the policy server
    client = PolicyClient(
        f"http://localhost:{args.port}", inference_mode=args.inference_mode
    )

    # Start a new episode
    obs = env.reset(seed=126)
    eid = client.start_episode(training_enabled=not args.no_train)

    rewards = 0.0
    while True:
        # Compute an action randomly (off-policy) and log it.
        if args.off_policy:
            action = np.random.randint(1, env._s + 1)
            client.log_action(eid, obs, action)
        # Compute an action locally or remotely (on server).
        # No need to log it here as the action
        else:
            action = client.get_action(eid, obs)

        # Perform a step in the external simulator (env).
        obs, reward, terminated, info = env.step(action)
        rewards += reward

        # Log next-obs, rewards, and infos.
        client.log_returns(eid, reward, info=info)

        # Reset the episode if done.
        if terminated:
            print("Total reward:", rewards)
            if rewards >= args.stop_reward:
                print("Target reward achieved, exiting")
                exit(0)

            rewards = 0.0

            # End the old episode.
            client.end_episode(eid, obs)

            # Start a new episode.
            obs = env.reset(seed=126)
            eid = client.start_episode(training_enabled=not args.no_train)
