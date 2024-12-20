# GymProxy

GymProxy is a tiny library for porting an external python-based simulation on Gymnasium (Gym) environment. 

It is designed for users who want to apply reinforcement learning (RL) in an existing python-based simulation. 

GymProxy makes a target simulation environment inter-operate with Gym through multi-threading.   

## Installation

As pre-requisite, you should have Python 3.7+ installed on your machine. 

[NumPy](https://numpy.org) and [Gym]([https://gymnasium.farama.org/) libraries are also required.    

Clone this repository on your machine and run:    

    $ cd ~/projects/gymproxy    # We assume that the repository is cloned to this directory
    $ pip install .

If you use Anaconda, you can install GymProxy by the followings:

    $ conda activate my_env     # We assume that 'my_env' is your working environment 
    $ conda develop ~/projects/gymproxy    

## Usage Examples

We present three gym-type environments as usage examples of GymProxy: 
- CarRental
- GamblersProblem
- AccessControlQueue

Each of the above environments simulates example 4.2 (Jack's car rental), 4.3 (gambler's problem), and 10.2 (access-control queuing task), respectively, described in this book:   

R. S. Sutton and A. G. Barto, *Reinforcement Learning: An Introduction*, 2nd ed., MIT Press, 2018. 

## Acknowledgement

This work was supported in part by Institute of Information & Communications Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT) (RS-2024-00392332, Development of 6G Network Integrated Intelligence Plane Technologies & 2017-0-00045, Hyper-Connected Intelligent Infrastructure Technology Development), and part by Electronics and Telecommunications Research Institutue (ETRI) grant funded by the Korea government (No. 24ZR1100, A Study of Hyper-Connected Thinking Internet Technology by Autonomous Connecting, Controlling and Evolving Ways).
