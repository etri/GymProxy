# Author: Seungjae Shin <sjshin0505@{etri.re.kr, gmail.com}>

"""Package for simulating Jack's car rental example described in the following reference:
R. S. Sutton and A. G. Barto, Reinforcement Learning - An Introduction, 2nd ed., 2018 (Example 4.2: Jack's Car Rental).
"""

import gym
from examples.car_rental.gym_env import CarRental

gym.register(id='CarRental-v0', entry_point='examples.car_rental:CarRental')
