# Author: Seungjae Shin <sjshin0505@{etri.re.kr, gmail.com}>

"""Module including BaseEnv class.
"""
from abc import *
from typing import TypeVar, Optional

import gymnasium

from gymproxy.base_actual_env_cy import BaseActualEnv
from gymproxy.env_proxy_cy import EnvProxy

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")
RenderFrame = TypeVar("RenderFrame")

class BaseEnv(gymnasium.Env, metaclass=ABCMeta):
    """Base class of gym-type environment.
    """
    metadata = {'render_modes': ['human']}
    actual_env_class = None

    def __init__(self, kwargs: Optional[dict] = None):
        """Constructor.

        :param kwargs: Dictionary of keyword arguments for beginning the actual environment. It should include a
        dictionary object indexed by 'config' keyword.
        """
        config = kwargs
        self.metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

        # Builds obs and action spaces.
        obs_space = self.build_obs_space(config)
        action_space = self.build_action_space(config)

        super().__init__()
        # Initializes the environment proxy object.
        self._env_proxy = EnvProxy(self.init_actual_env, self.reset_actual_env, self.close_actual_env, **config)

        self.observation_space = obs_space
        self.action_space = action_space

    def reset(self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
        ) -> (object, dict):
        """Resets the environment to an initial state and returns an initial observation. See gym.core.Env.reset() for
        detailed description.

        :return: observation: Agent's observation of the current environment.
        """
        if seed is None:
            import numpy as np
            seed = np.random.randint(999)
        # print("seed {}, options {}".format(seed, options))
        self._env_proxy.reset_actual_env(seed, options)  # Resets the actual environment.
        return self._env_proxy.get_obs_and_info()    # Gets observation object from the environment proxy.

    def step(self, action: object) -> (object, float, bool, bool, dict):
        """Run one time-step of the environment's dynamics. See gym.core.Env.step() for detailed description.

        :param action: An action provided by the agent.
        :return: Tuple of (observation, reward, done, info).
            observation: Agent's observation of the current environment.
            reward: Amount of reward returned after previous action.
            done: Whether the episode has ended, in which case further step() calls will return undefined results.
            info: Contains auxiliary diagnostic information (helpful for debugging, and sometimes learning).
        """
        #print("base_env step", action)
        self._env_proxy.set_action(action)  # Sends action to the environment proxy.

        # Gets tuple of (obs, reward, done, info) from the environment proxy.
        return self._env_proxy.get_obs_and_reward()

    def render(self, mode: str = 'human'):
        """Renders the environment. See gym.core.Env.render() for detailed description.

        :param mode: The mode to render with.
        """
        return

    def close(self):
        """Override close in your subclass to perform any necessary cleanup. See gym.core.Env.close() for detailed
        description.
        """
        self._env_proxy.close_actual_env()

    # @staticmethod
    # @abstractmethod
    # def update_action_space(self, obs: int):
    #     """Builds action space.
    #
    #     :param kwargs: Dictionary of keyword arguments for building action space.
    #     """
    #     raise NotImplementedError

    @staticmethod
    def init_actual_env(kwargs: Optional[dict] = None) -> object:
        """Initializes the actual environment.

        :param kwargs: Dictionary of keyword arguments for initializing the actual environment.
        :return: Result of constructing the actual environment object.
        """
        return eval('BaseEnv.actual_env_class(kwargs)')

    @staticmethod
    def reset_actual_env(actual_env: BaseActualEnv, seed:int, kwargs: Optional[dict] = None):
        """Resets the actual environment.

        :param actual_env: Reference of the actual environment.
        :param kwargs: Dictionary of keyword arguments for resetting the actual environment.
        """
        actual_env.run(seed, kwargs)

    @staticmethod
    def close_actual_env(actual_env: BaseActualEnv, kwargs: Optional[dict] = None):
        """Closes the actual environment.

        :param actual_env: Reference of the actual environment.
        :param kwargs: Dictionary of keyword arguments for closing the actual environment.
        """
        actual_env.finish(kwargs)

    @staticmethod
    @abstractmethod
    def build_obs_space(kwargs: Optional[dict] = None):
        """Builds observation space.

        :param kwargs: Dictionary of keyword arguments for building observation space.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def build_action_space(kwargs: Optional[dict] = None):
        """Builds action space.

        :param kwargs: Dictionary of keyword arguments for building action space.
        """
        raise NotImplementedError
