# Author: Seungjae Shin <sjshin0505@{etri.re.kr, gmail.com}>

"""Package for simulating gambler's problem described in the following reference:
R. S. Sutton and A. G. Barto, Reinforcement Learning - An Introduction, 2nd ed., 2018 (Example 4.3: Gambler's Problem).
"""

import gymnasium as gym
from examples.gamblers_problem.gym_env import GamblersProblem

gym.register(id='GamblersProblem-v0', entry_point='examples.gamblers_problem:GamblersProblem')
