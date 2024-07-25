# Author: Seungjae Shin <sjshin0505@{etri.re.kr, gmail.com}>

"""Module including TerminateGymProxy and BaseActualEnv classes.
"""

from abc import ABC, abstractmethod
from gymproxy.env_proxy_cy import EnvProxy
from typing import Optional

class TerminateGymProxy(Exception):
    """Utility class for terminating GymProxy as an exception."""
    def __init__(self):
        super().__init__('TerminateGymProxy')

class BaseActualEnv(ABC):
    """Base class of external environment that actually implements a target simulation."""
    env_proxy = None  # Class variable for holding the reference of the environment proxy object.

    def __init__(self, env_proxy: EnvProxy):
        """Constructor.

        :param env_proxy: Environment proxy object.
        """
        BaseActualEnv.env_proxy = env_proxy

    @abstractmethod
    def run(self, seed: int, kwargs: Optional[dict] = None):
        """Should define the main control loop of the actual environment here.

        :param seed: Seed for random number generation.
        :param kwargs: Dictionary of keyword arguments for beginning the actual environment.
        """
        raise NotImplementedError

    @abstractmethod
    def finish(self, kwargs: Optional[dict] = None):
        """Should define the procedure required for finishing the actual environment here.

        :param kwargs: Dictionary of keyword arguments for finishing the actual environment.
        """
        raise NotImplementedError

    @staticmethod
    def get_action(obs: object, reward: float, done: bool, truncated: bool, info: dict) -> any:
        """Gets action from the environment proxy. This method should be called in the scope of the actual environment.

        :param obs: Observation to be given to the agent.
        :param reward: Reward to be given to the agent.
        :param done: Indicates whether the actual environment is finished or not.
        :param truncated: Indicates whether the episode was truncated.
        :param info: Information that is additionally to be given to the agent.
        :return: action: Action from the agent.
        """
        action, closing = BaseActualEnv.env_proxy.get_action(obs, reward, done, truncated, info)
        if closing:
            raise TerminateGymProxy()
        return action

    @staticmethod
    def set_obs_and_reward(obs: object, reward: float, done: bool, truncated: bool, info: dict):
        """Sends observation, reward, done, and information to the agent, but does not receive action. This method should
        be called in the scope of the environment when the beginning or end of an episode.

        :param obs: Observation to be given to the agent.
        :param reward: Reward to be given to the agent.
        :param done: Indicates whether the actual environment is finished or not.
        :param truncated: Indicates whether the episode was truncated.
        :param info: Information that is additionally to be given to the agent.
        """
        BaseActualEnv.env_proxy.set_obs_and_reward(obs, reward, done, truncated, info)
