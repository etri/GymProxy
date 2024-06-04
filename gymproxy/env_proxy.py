# Author: Seungjae Shin <sjshin0505@{etri.re.kr, gmail.com}>

"""Module including EnvProxy class.
"""

from abc import *
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

    def __init__(self, init_actual_env: callable, reset_actual_env: callable, close_actual_env: callable, kwargs: Optional[dict] = None):
        """Constructor.

        :param init_actual_env: Function for initializing the actual environment.
        :param reset_actual_env: Function for resetting the actual environment.
        :param close_actual_env: Function for closing the actual environment.
        :param kwargs: Dictionary of keyword arguments. It should include a dictionary indexed by 'config' keyword.
        """
        # Prepares a thread pool and synchronization variables.
        self._pool = ThreadPoolExecutor(max_workers=1)
        self._future = None
        print("EnvProxy __init__ called self._lock = Lock()")
        self._lock = Lock()                 # For locking a critical section.
        self._actual_env_event = Event()    # Event for signaling the actual environment to enter the critical section.
        self._gym_env_event = Event()       # Event for signaling gym-type environment to enter the critical section.
        self._actual_env_event.clear()
        self._gym_env_event.clear()

        self._config = kwargs
        kwargs['env_proxy'] = self      # Adds self-reference to kwargs

        # Prepares the function objects for handling the actual environment.
        self._actual_env = init_actual_env(kwargs)
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

    def reset_actual_env(self, seed:int):
        """Resets the actual environment.
        """
        print("reset_acutal_env")
        def reset_actual_env_(seed:int):
            """Nested function for adding synchronization mechanism to _reset_actual_env() function. This function is
            executed in a thread provided by _pool. This implies that the actual environment is executed in a  separate
            thread provided by _pool.
            """
            # Enters the critical section if it is unlocked. Otherwise, wait until the critical section is unlocked as
            # a result of calling get_obs() method from a gym-type environment.

            self._lock.acquire()

            self._actual_env_event.wait()   # Waits for calling get_obs() method from the gym-type environment.
            self._reset_actual_env(self._actual_env, seed)   # Actually resets the actual environment.

        # Locks the critical section for making reset_actual_env_() stopped before resetting the actual environment. It
        # will be unlocked by get_obs() method calling from the gym-type environment.

        if not self._lock.locked():
            self._lock.acquire()

        self._closing = False

        # Begins the thread for executing the actual environment.
        self._future = self._pool.submit(reset_actual_env_, seed)


    def close_actual_env(self):
        """Closes the actual environment.
        """
        print("close_actual_env")
        self._close_actual_env(actual_env=self._actual_env)     # Actually closes the actual environment.

        # Drive the thread for actual environment finished.
        if self._future and self._future.running():
            self._closing = True
            self._sync_gym_env()   # Resumes the actual environment and stops gym-type environment.

    def get_obs_and_reward(self) -> (object, float, bool, bool, dict):
        """Gets observation and reward from the actual environment. Called by gym-type environment.

        :return: Tuple of (observation, reward, terminated, info).
            observation: Agent's observation of the actual environment.
            reward: Amount of reward returned after previous action.
            terminated: Whether the episode has ended, in which case further step() calls will return undefined results.
            info: Contains auxiliary diagnostic information (helpful for debugging, and sometimes learning).
        """
        print("get_obs_and_reward")
        self._sync_gym_env()   # Resumes the actual environment, and then, waits until the actual environment stops.

        # Obtains a tuple of (observation, reward, terminated, info) after the actual environment stops.
        return self._obs, self._reward, self._terminated, self._truncated, self._info

    def get_obs(self) -> object:
        """Gets observation from the actual environment. Called by gym-type environment.

        :return: observation: Agent's observation of the current environment.
        """
        print("get_obs")
        self._sync_gym_env()    # Resumes the actual environment, and then, waits until the actual environment stops.
        return self._obs        # Obtains observation after the actual environment stops.

    def get_obs_and_info(self) -> (object, dict):
        """Gets observation from the actual environment. Called by gym-type environment.

        :return: observation: Agent's observation of the current environment.
        """
        print("get_obs_and_info")
        self._sync_gym_env()    # Resumes the actual environment, and then, waits until the actual environment stops.
        return self._obs, self._info         # Obtains observation after the actual environment stops.

    def get_action(self, obs: object, reward: float, terminated: bool, truncated:bool, info: dict) -> (object, bool):
        """Gets action from the gym-type environment. Called by the actual environment.

        :param obs: Agent's observation of the actual environment.
        :param reward: Amount of reward returned after previous action.
        :param terminated: Whether the episode has ended, in which case further step() calls will return undefined results.
        :param info: Contains auxiliary diagnostic information (helpful for debugging, and sometimes learning).
        :return: Tuple of (action, closing).
            action: An action provided by gym-type environment.
            closing: Whether the actual environment should be closed or not.
        """
        print("env_proxy get_action")
        self.set_obs_and_reward(obs, reward, terminated, truncated, info)

        # Obtains observation after the gym-type environment stops.
        return self._action, self._closing

    def set_obs_and_reward(self, obs: object, reward: float, terminated: bool, truncated: bool, info: dict):
        """Sets tuple of (observation, reward, terminated, info) from the actual environment.

        :param obs: Agent's observation of the actual environment.
        :param reward: Amount of reward returned after previous action.
        :param terminated: Whether the episode has ended, in which case further step() calls will return undefined results.
        :param info: Contains auxiliary diagnostic information (helpful for debugging, and sometimes learning).
        """
        print("set_obs_and_reward", obs, reward, terminated)
        self._obs = obs
        self._reward = reward
        self._terminated = terminated
        self._truncated = truncated
        self._info = info

        # Resumes the gym-type environment, and then, waits until the gym-type environment stops.
        self._sync_actual_env(terminated)

    def set_action(self, action: object):
        """Sets action from the gym-type environment.

        :param action: An action provided by the gym-type environment.
        """
        print("set_action")
        self._action = action

    def release_lock(self):
        """Releases _lock. Utility method required for closing safely the actual environment.
        """
        print("release_lock called")
        self._lock.release()

    def set_gym_env_event(self):
        """Sets _gym_env_event. Utility method required for closing safely the actual environment.
        """
        print("set_gym_env_event")
        self._gym_env_event.set()

    def _sync_actual_env(self, terminated):
        """Resumes the gym-type environment and wait for its next stop. Called by the actual environment.

        :param terminated: Whether the episode should be ended.
        """
        # Resumes the gym-type environment.
        print("_sync_actual_env called {}", terminated)
        self._actual_env_event.clear()
        print("_sync_actual_env called 1")
        self._gym_env_event.set()
        print("_sync_actual_env called 2")
        print("self._lock ", self._lock)
        self._lock.release()
        print("_sync_actual_env called 3")

        # Wait during the gym-type environment is active.
        if not terminated:
            print("if not terminated")
            self._actual_env_event.wait()
            self._lock.acquire()

    def _sync_gym_env(self):
        """Resumes the actual environment and wait for its next stop. Called by the gym-type environment.
        """
        # Resumes the actual environment.
        print("_sync_gem_env called")
        self._gym_env_event.clear()
        self._actual_env_event.set()
        self._lock.release()

        # Waits during the actual environment is active.
        self._gym_env_event.wait()
        self._lock.acquire()
