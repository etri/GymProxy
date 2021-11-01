# Author: Seungjae Shin <sjshin0505@{etri.re.kr, gmail.com}>

"""Package for simulating an access control example in a simple queuing system. It is implemented based on the
following reference:
R. S. Sutton and A. G. Barto, Reinforcement Learning - An Introduction, 2nd ed., 2018 (Example 10.2: An Access-Control
Queuing Task).
"""

import gym
from examples.access_control_q.gym_env import AccessControlQueue

gym.register(id='AccessControlQueue-v0', entry_point='examples.access_control_q:AccessControlQueue')
