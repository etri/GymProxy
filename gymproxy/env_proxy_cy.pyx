# Author: Seungjae Shin <sjshin0505@{etri.re.kr, gmail.com}>

"""Module including EnvProxy class.
"""

from abc import ABC
from concurrent.futures import ThreadPoolExecutor
from threading import Event, Lock
from typing import Optional

def singleton(class_):
    instances = {}
    def getinstance(*args, **kwargs):
        if class_ not in instances:
            instances[class_] = class_(*args, **kwargs)
        return instances[class_]
    return getinstance

@singleton
class EnvProxy(ABC):
    """Environment proxy that plays a role of interface between a gym-type and external actual environment.
    """

    _instance = None
    _lock = Lock()

    @classmethod
    def get_instance(cls, init_actual_env=None, reset_actual_env=None, close_actual_env=None, kwargs: Optional[dict] = None):
        with cls._lock:
            if cls._instance is None:
                if kwargs is None:
                    kwargs = {}
                cls._instance = cls(init_actual_env, reset_actual_env, close_actual_env, kwargs)
            return cls._instance

    def __init__(self, init_actual_env: callable, reset_actual_env: callable, close_actual_env: callable, **kwargs):
        if self._instance is not None:
            raise RuntimeError("This class is a singleton! Use get_instance() method to get the single instance.")

        self._pool = ThreadPoolExecutor(max_workers=1)
        self._future = None
        self._lock = Lock()                 # For locking a critical section.
        self._actual_env_event = Event()    # Event for signaling the actual environment to enter the critical section.
        self._gym_env_event = Event()       # Event for signaling gym-type environment to enter the critical section.
        self._actual_env_event.clear()
        self._gym_env_event.clear()

        self._config = kwargs or {}
        self._config['env_proxy'] = self      # Adds self-reference to kwargs

        self._actual_env = init_actual_env(self._config)
        self._reset_actual_env = reset_actual_env
        self._close_actual_env = close_actual_env

        # Critical section variables
        self._closing = False
        self._obs = None
        self._action = None
        self._reward = None
        self._terminated = None
        self._truncated = None
        self._info = None

    def reset_actual_env(self, seed: int, options: Optional[dict] = None):
        """Resets the actual environment.
        """
        def reset_actual_env_(seed: int, options: Optional[dict] = None):
            """Nested function for adding synchronization mechanism to _reset_actual_env() function. This function is
            executed in a thread provided by _pool. This implies that the actual environment is executed in a separate
            thread provided by _pool.
            """
            self._lock.acquire()
            self._actual_env_event.wait()   # Waits for calling get_obs() method from the gym-type environment.
            self._reset_actual_env(self._actual_env, seed, options)   # Actually resets the actual environment.

        if not self._lock.locked():
            self._lock.acquire()
            self._lock.release()

        self._closing = False
        self._future = self._pool.submit(reset_actual_env_, seed, options)

    def close_actual_env(self):
        """Closes the actual environment.
        """
        self._close_actual_env(actual_env=self._actual_env)  # Actually closes the actual environment.

        if self._future and self._future.running():
            self._closing = True
            self._sync_gym_env()  # Resumes the actual environment and stops gym-type environment.

    def get_obs_and_reward(self) -> (object, float, bool, bool, dict):
        """Gets observation and reward from the actual environment. Called by gym-type environment.

        :return: Tuple of (observation, reward, terminated, truncated, info).
        """
        self._sync_gym_env()  # Resumes the actual environment, and then, waits until the actual environment stops.
        return self._obs, self._reward, self._terminated, self._truncated, self._info

    def get_obs(self) -> object:
        """Gets observation from the actual environment. Called by gym-type environment.

        :return: observation: Agent's observation of the current environment.
        """
        self._sync_gym_env()  # Resumes the actual environment, and then, waits until the actual environment stops.
        return self._obs  # Obtains observation after the actual environment stops.

    def get_obs_and_info(self) -> (object, dict):
        """Gets observation from the actual environment. Called by gym-type environment.

        :return: observation, info: Agent's observation and additional information of the current environment.
        """
        self._sync_gym_env()  # Resumes the actual environment, and then, waits until the actual environment stops.
        return self._obs, self._info  # Obtains observation and info after the actual environment stops.

    def get_action(self, obs: object, reward: float, terminated: bool, truncated: bool, info: dict) -> (object, bool):
        """Gets action from the gym-type environment. Called by the actual environment.

        :param obs: Agent's observation of the actual environment.
        :param reward: Amount of reward returned after previous action.
        :param terminated: Whether the episode has ended, in which case further step() calls will return undefined results.
        :param truncated: Whether the episode was truncated before its natural end.
        :param info: Contains auxiliary diagnostic information (helpful for debugging, and sometimes learning).
        :return: Tuple of (action, closing).
        """
        self.set_obs_and_reward(obs, reward, terminated, truncated, info)
        return self._action, self._closing

    def set_obs_and_reward(self, obs: object, reward: float, terminated: bool, truncated: bool, info: dict):
        """Sets tuple of (observation, reward, terminated, truncated, info) from the actual environment.

        :param obs: Agent's observation of the actual environment.
        :param reward: Amount of reward returned after previous action.
        :param terminated: Whether the episode has ended, in which case further step() calls will return undefined results.
        :param truncated: Whether the episode was truncated before its natural end.
        :param info: Contains auxiliary diagnostic information (helpful for debugging, and sometimes learning).
        """
        self._obs = obs.flatten()
        self._reward = reward
        self._terminated = terminated
        self._truncated = truncated
        self._info = info
        self._sync_actual_env(terminated)

    def set_action(self, action: object):
        """Sets action from the gym-type environment.

        :param action: An action provided by the gym-type environment.
        """
        self._action = action

    def release_lock(self):
        """Releases _lock. Utility method required for closing safely the actual environment.
        """
        self._lock.release()

    def set_gym_env_event(self):
        """Sets _gym_env_event. Utility method required for closing safely the actual environment.
        """
        self._gym_env_event.set()

    def _sync_actual_env(self, terminated):
        """Resumes the gym-type environment and wait for its next stop. Called by the actual environment.

        :param terminated: Whether the episode should be ended.
        """
        self._actual_env_event.clear()
        self._gym_env_event.set()
        if terminated:
            self._lock.release()
        if not terminated:
            self._actual_env_event.wait()

    def _sync_gym_env(self):
        """Resumes the actual environment and wait for its next stop. Called by the gym-type environment.
        """
        self._gym_env_event.clear()
        self._actual_env_event.set()
        self._gym_env_event.wait()
