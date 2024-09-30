# Author: Sae Hyong Park <labry@etri.re.kr>, Seungjae Shin <sjshin0505@{etri.re.kr, gmail.com}>

"""Module for a number of utility routines.
"""

from typing import SupportsFloat, Any


def get_step_log_str(episode: int,
                     step: int,
                     obs: Any,
                     reward: SupportsFloat,
                     terminated: bool,
                     truncated: bool,
                     info: dict,
                     action: Any):
    """Utility function for printing logs.

    Args:
        episode: Episode number.
        step: Time-step.
        obs: Observation of the current environment.
        reward: Reward from the current environment.
        terminated: Indicates whether the episode ends or not.
        truncated: Indicates whether the episode ends or not.
        info: Contains auxiliary diagnostic information (helpful for debugging, and sometimes learning).
        action: An action provided by the agent.
    """
    priority = obs[0].item()
    num_free_servers = obs[1].item()
    step_str = '{}-th step in {}-th episode / '.format(step, episode)
    obs_str = 'obs: {} / '.format((priority, num_free_servers))
    reward_str = 'reward: {} / '.format(reward)
    terminated_str = 'terminated: {} / '.format(terminated)
    truncated_str = 'truncated: {} / '.format(truncated)
    info_str = 'info: {} / '.format(info)
    action_str = 'action: {}'.format(action)
    step_log_str = step_str + obs_str + reward_str + terminated_str + truncated_str + info_str + action_str
    return step_log_str


def get_step_log_str_for_standalone(step: int,
                                    obs: Any,
                                    reward: SupportsFloat,
                                    action: Any):
    """Utility function for printing logs for standalone simulator case.

    Args:
        step: Time-step.
        obs: Observation of the current environment.
        reward: Reward from the current environment.
        action: An action provided by the policy.
    """
    priority = obs[0].item()
    num_free_servers = obs[1].item()
    step_str = '{}-th step / '.format(step)
    obs_str = 'obs: {} / '.format((priority, num_free_servers))
    reward_str = 'reward: {} / '.format(reward)
    action_str = 'action: {}'.format(action)
    step_log_str = step_str + obs_str + reward_str + action_str
    return step_log_str
