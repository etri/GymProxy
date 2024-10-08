# Author: Sae Hyong Park <labry@etri.re.kr>, Seungjae Shin <sjshin0505@{etri.re.kr, gmail.com}>

"""Module including TerminateGymProxy and ActualEnv classes.
"""

from abc import *
from gymproxy.env_proxy import EnvProxy
from typing import Optional

class TerminateGymProxy(Exception):
    """Utility class for terminating GymProxy as an exception.
    """
    def __init__(self):
        """Constructor."""
        super().__init__('TerminateGymProxy')


class ActualEnv(ABC):
    """Base class of external environment that actually implements a target simulation.
    """
    env_proxy = None    # Class variable for holding the reference of the environment proxy object.

    def __init__(self, env_proxy: EnvProxy):
        """Constructor.

        Args:
            env_proxy: Environment proxy object.
        """
        ActualEnv.env_proxy = env_proxy

    @abstractmethod
    def run(self, seed: int, kwargs: Optional[dict] = None):
        """Should define the main control loop of the actual environment here.

        Args:
            kwargs: Dictionary of keyword arguments for beginning the actual environment.
            seed: TBD
        """
        raise NotImplementedError

    @abstractmethod
    def finish(self, kwargs: Optional[dict] = None):
        """Should define the procedure required for finishing actual environment here.

        Args:
            kwargs: Dictionary of keyword arguments for finishing the actual environment.
        """
        raise NotImplementedError

    @staticmethod
    def get_action(obs: object, reward: float, terminated: bool, truncated:bool, info: dict) -> any:
        """Gets action from the environment proxy. This method should be called in the scope of the actual environment.

        Args:
            obs: Observation to be given to the agent.
            reward: Reward to be given to the agent.
            terminated: Indicates whether the episode finishes or not.
            truncated: Indicates whether the episode is truncated or not.
            info: Information that is additionally to be given to the agent.

        Returns:
            action: Action from the agent.
        """
        action, closing = ActualEnv.env_proxy.get_action(obs, reward, terminated, truncated, info)
        
        if closing:
            raise TerminateGymProxy()
        return action

    @staticmethod
    def set_obs_and_reward(obs: object, reward: float, terminated: bool, truncated:bool, info: dict):
        """Sends observation, reward, done, and information to the agent, but do not receive action. 
        This method should be called in the scope of environment when the begin or end of an episode.

        Args:
            obs: Observation to be given to the agent.
            reward: Reward to be given to the agent.
            terminated: Indicates whether the episode finishes or not.
            truncated: Indicates whether the episode is truncated or not.
            info: Information that is additionally to be given to the agent.
        """
        ActualEnv.env_proxy.set_obs_and_reward(obs, reward, terminated, truncated, info)
