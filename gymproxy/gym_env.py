# Author: Seungjae Shin <sjshin0505@{etri.re.kr, gmail.com}>
#         Sae Hyong Park <labry@etri.re.kr>

"""Module including GymEnv class.
"""

import gymnasium as gym

from abc import *
from typing import TypeVar, Optional
from gymproxy.actual_env import ActualEnv
from gymproxy.env_proxy import EnvProxy

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")
RenderFrame = TypeVar("RenderFrame")

class GymEnv(gym.Env, metaclass=ABCMeta):
    """Base class of gym-type environment.
    """
    metadata = {'render_modes': ['human']}
    actual_env_class = None

    def __init__(self, kwargs: Optional[dict] = None):
        """Constructor.

        Args:
            kwargs (optional dict): Dictionary of keyword arguments for beginning the actual environment. 
                It should include a dictionary object indexed by 'config' keyword.
        """
        config = kwargs
        self.metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

        # Builds obs and action spaces.
        obs_space = self.build_obs_space(config)
        action_space = self.build_action_space(config)

        super().__init__()
        
        # Initializes the environment proxy object.
        self._env_proxy = EnvProxy(GymEnv.actual_env_class, **config)

        self.observation_space = obs_space
        self.action_space = action_space

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None, ) -> tuple[ObsType, dict[str, Any]]:
        """Resets the environment to an initial internal state, returning an initial observation and info. 
            See gym.core.Env.reset() for detailed description.

        Args:
            seed (optional int): The seed that is used to initialize the environment's PRNG (`np_random`) and
                the read-only attribute `np_random_seed`. See gym.core.Env.reset() for detailed description.
            options (optional dict): Additional information to specify how the environment is reset. 
                See gym.core.Env.reset() for detailed description.

        Returns:
            observation (ObsType): Observation of the initial state. See gym.core.Env.reset() for detailed description.
            info (dictionary): This dictionary contains auxiliary information complementing ``observation``. 
                See gym.core.Env.reset() for detailed description.
        """
        self._env_proxy.reset_actual_env(seed, options)  # Resets the actual environment.
        return self._env_proxy.get_obs_and_info()        # Gets observation object from the environment proxy.

    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Run one timestep of the environment's dynamics using the agent actions. See gym.core.Env.step() for detailed description.

        Args:
            action (ActType): an action provided by the agent to update the environment state.

        Returns:
            observation (ObsType): An element of the environment's :attr:`observation_space` as the next observation due to the agent actions.
                See gym.core.Env.step() for detailed description.
            reward (SupportsFloat): The reward as a result of taking the action.
            terminated (bool): Whether the agent reaches the terminal state (as defined under the MDP of the task)
                which can be positive or negative. See gym.core.Env.step() for detailed description.
            truncated (bool): Whether the truncation condition outside the scope of the MDP is satisfied.
                See gym.core.Env.step() for detailed description.
            info (dict): Contains auxiliary diagnostic information (helpful for debugging, learning, and logging).
                See gym.core.Env.step() for detailed description.
        """
        self._env_proxy.set_action(action)  # Sends action to the environment proxy.

        # Gets tuple of (obs, reward, done, info) from the environment proxy.
        return self._env_proxy.get_obs_and_reward()
    
    def close(self):
        """After the user has finished using the environment, close contains the code necessary to "clean up" the environment.
        See gym.core.Env.close() for detailed description.
        """
        self._env_proxy.close_actual_env()

    @staticmethod
    @abstractmethod
    def build_obs_space(kwargs: Optional[dict] = None):
        """Builds observation space.

        Args:
            kwargs: Dictionary of keyword arguments for building observation space.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def build_action_space(kwargs: Optional[dict] = None):
        """Builds action space.

        Args:
            kwargs: Dictionary of keyword arguments for building action space.
        """
        raise NotImplementedError
