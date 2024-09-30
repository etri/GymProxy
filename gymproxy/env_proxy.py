# Author: Sae Hyong Park <labry@etri.re.kr>, Seungjae Shin <sjshin0505@{etri.re.kr, gmail.com}>

"""Module including EnvProxy class.
"""

from abc import *
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from typing import Optional

def singleton(class_):
    instances = {}
    def getinstance(*args, **kwargs):
        if class_ not in instances:
            instances[class_] = class_(*args, kwargs)
        return instances[class_]
    return getinstance

@singleton
class EnvProxy(ABC):
    """Environment proxy that plays a role of interface between a gym-type and external actual environment.
    """

    def __init__(self, actual_env_class, kwargs: Optional[dict] = None):
        """Constructor.

        Args:
            actual_env_class: TBD. 
            kwargs: Dictionary of keyword arguments. It should include a dictionary indexed by 'config' keyword.
        """
        # Prepares a thread pool and synchronization variables.
        self._pool = ThreadPoolExecutor(max_workers=1)
        self._future = None
        self._actual_env_event = Event()    # Event for signaling the actual environment to enter the critical section.
        self._gym_env_event = Event()       # Event for signaling gym-type environment to enter the critical section.
        self._actual_env_event.clear()
        self._gym_env_event.clear()

        self._config = kwargs
        kwargs['env_proxy'] = self      # Adds self-reference to kwargs

        # Launches the actual environment.
        self._actual_env = eval('actual_env_class(kwargs)')

        # Critical section variables
        self._closing = False
        self._obs = None
        self._action = None
        self._reward = None
        self._terminated = None
        self._truncated = None
        self._info = None

    def reset_actual_env(self, seed:int, options: Optional[dict] = None):
        """Resets the actual environment.

        Args:
            seed: TBD. 
            options: Dictionary of optional arguments.
        """
        def reset_actual_env_(seed:int, options: Optional[dict] = None):
            """Nested function for adding synchronization mechanism to _reset_actual_env() function. 
            This function is executed in a thread provided by _pool. 
            This implies that the actual environment is executed in a separate thread provided by _pool.
            """
            self._actual_env_event.wait()   # Waits for calling get_obs() method from the gym-type environment.
            self._actual_env.run(seed, options)    # Actually resets the actual environment.

        # Begins the thread for executing the actual environment.
        self._future = self._pool.submit(reset_actual_env_, seed, options)

    def close_actual_env(self, kwargs: Optional[dict] = None):
        """Closes the actual environment.
        """
        self._actual_env.finish(kwargs)    # Actually closes the actual environment.

        # Drive the thread for actual environment finished.
        if self._future and self._future.running():
            self._closing = True
            self._sync_gym_env()   # Resumes the actual environment and stops gym-type environment.

    def get_obs_and_reward(self) -> (object, float, bool, bool, dict):
        """Gets observation and reward from the actual environment. Called by gym-type environment.

        Returns: 
            observation (ObsType): Agent's observation of the actual environment.
            reward (float): Amount of reward returned after previous action.
            terminated (bool): Whether the episode has ended, in which case further step() calls will return undefined results.
            info (dictionary): Contains auxiliary diagnostic information (helpful for debugging, and sometimes learning).    
        """
        self._sync_gym_env()   # Resumes the actual environment, and then, waits until the actual environment stops.

        # Obtains a tuple of (observation, reward, terminated, info) after the actual environment stops.
        return self._obs, self._reward, self._terminated, self._truncated, self._info

    def get_obs_and_info(self) -> (object, dict):
        """Gets observation and information from the actual environment. Called by gym-type environment.

        Returns: 
            observation (ObsType): Agent's observation of the actual environment.
            info (dictionary): Contains auxiliary diagnostic information (helpful for debugging, and sometimes learning).
        """
        self._sync_gym_env()    # Resumes the actual environment, and then, waits until the actual environment stops.
        return self._obs, self._info         # Obtains observation after the actual environment stops.

    def get_action(self, obs: object, reward: float, terminated: bool, truncated:bool, info: dict) -> (object, bool):
        """Gets action from the gym-type environment. Called by the actual environment.

        Args:
            obs: Agent's observation of the actual environment.
            reward: Amount of reward returned after previous action.
            terminated: Whether the episode has ended, in which case further step() calls will return undefined results.
            info: Contains auxiliary diagnostic information (helpful for debugging, and sometimes learning).
        
        Returns: 
            action: An action provided by gym-type environment.
            closing: Whether the actual environment should be closed or not.
        """
        self.set_obs_and_reward(obs, reward, terminated, truncated, info)

        # Obtains observation after the gym-type environment stops.
        return self._action, self._closing

    def set_obs_and_reward(self, obs: object, reward: float, terminated: bool, truncated: bool, info: dict):
        """Sets tuple of (observation, reward, terminated, info) from the actual environment.

        Args:
            obs: Agent's observation of the actual environment.
            reward: Amount of reward returned after previous action.
            terminated: Whether the episode has ended, in which case further step() calls will return undefined results.
            info: Contains auxiliary diagnostic information (helpful for debugging, and sometimes learning).
        """
        self._obs = obs.flatten()
        self._reward = reward
        self._terminated = terminated
        self._truncated = truncated
        self._info = info

        # Resumes the gym-type environment, and then, waits until the gym-type environment stops.
        self._sync_actual_env(terminated)

    def set_action(self, action: object):
        """Sets action from the gym-type environment.

        Args:
            action: An action provided by the gym-type environment.
        """
        self._action = action

    def terminate_sync(self):
        """Sets _gym_env_event. Utility method required for closing safely the actual environment.
        """
        self._gym_env_event.set()

    def _sync_actual_env(self, terminated):
        """Resumes the gym-type environment and wait for its next stop. Called by the actual environment.

        Args:
            terminated: Whether the episode should be ended.
        """
        # Resumes the gym-type environment.
        self._actual_env_event.clear()
        self._gym_env_event.set()

        # Wait during the gym-type environment is active.
        if not terminated:
            self._actual_env_event.wait()

    def _sync_gym_env(self):
        """Resumes the actual environment and wait for its next stop. Called by the gym-type environment.
        """
        # Resumes the actual environment.
        self._gym_env_event.clear()
        self._actual_env_event.set()

        # Waits during the actual environment is active.
        self._gym_env_event.wait()
