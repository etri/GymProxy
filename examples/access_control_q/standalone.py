# Author: Sae Hyong Park <labry@etri.re.kr>, Seungjae Shin <sjshin0505@{etri.re.kr, gmail.com}>

"""External environment for actually simulating gambler's problem example. It is implemented based on the following reference:
R. S. Sutton and A. G. Barto, Reinforcement Learning - An Introduction, 2nd ed., 2018 (Example 10.2: An Access-Control Queuing Task).
"""

import logging
import numpy as np

from examples.utilities import get_step_log_str_for_standalone

# Setting logger
FORMAT = "[%(asctime)s|%(levelname)s|%(name)s] %(message)s"
DATE_FMT = "%H:%M:%S %Y-%m-%d"
log_level = logging.INFO
logging.basicConfig(format=FORMAT, datefmt=DATE_FMT, level=log_level)
logger = logging.getLogger('access_control_q')

NUM_STEPS = 100
NUM_SERVERS = 10
SERVER_FREE_PROBABILITY = 0.06
PRIORITIES = [1., 2., 4., 8.]
SEED = 2024

def get_new_customer(priorities) -> float:
    return priorities[np.random.randint(0, len(priorities))]
    
def get_free_servers(server_states) -> list[int]:
    return list(filter(lambda i: server_states[i] == 'free', range(0, len(server_states))))
    
def get_busy_servers(server_states) -> list[int]:
    return list(filter(lambda i: server_states[i] == 'busy', range(0, len(server_states))))
    
def make_obs(priority, free_servers) -> np.ndarray:
    return np.array([priority, len(free_servers)], dtype=np.int32)
    
def choose_free_server(free_servers) -> int:
    return free_servers[np.random.randint(0, len(free_servers))]

def policy(obs, reward) -> int:
    return np.random.choice([False, True])

def main():
    server_states = ['free'] * NUM_SERVERS
    reward = 0.
    action = None
    t = 0
    np.random.seed(SEED)
    
    while t < NUM_STEPS:
        priority = get_new_customer(PRIORITIES) # Assumes that a new customer arrives.
        # Chooses new customer's priority from the list of candidate priorities.

        free_servers = get_free_servers(server_states)    # Identifies free servers.
        obs = make_obs(priority, free_servers)    # Observation consists of customer's priority and number of free servers.
        action = policy(obs, reward)    # Action can be 0 or 1: 0 means rejection of customer. 1 means acceptance of customer.
        log_step_str = get_step_log_str_for_standalone(t, obs, reward, action)
        logger.info(log_step_str)

        if len(free_servers) > 0:
            if action:  # Means acceptance.
                i = choose_free_server(free_servers)    # Randomly chooses a server for the customer among free ones.
                server_states[i] = 'busy'    # Selected server becomes busy.
                reward = priority    # Reward is the priority of accepted customer.
            else:    # Means rejection.
                reward = 0.
        else:    # Rejects the customer if the number of free servers is 0.
            reward = 0.
            
        busy_servers = get_busy_servers(server_states)
        
        for i in busy_servers:    # Busy servers become free with SERVER_FREE_PROBABILITY
            if np.random.rand() < SERVER_FREE_PROBABILITY:
                server_states[i] = 'free'

        t += 1

    # Arrives to the end of the episode (terminal state).
    free_servers = get_free_servers(server_states)
    obs = make_obs(0, free_servers)
    log_step_str = get_step_log_str_for_standalone(t, obs, reward, action)
    logger.info(log_step_str)


if __name__ == "__main__":
    main()
