import numpy as np

PROB_HEAD = 0.5

_p_h = PROB_HEAD
seed_ = 1

np.random.seed(seed_)

win = 0
loose = 0
EPISODES = 1000000

np.random.seed(seed_)
for i in range(EPISODES):
    r = np.random.rand()
    if r < _p_h:
        # logger.info("heads")
        win +=1
    else:
        loose +=1


print("win rate: {} loose rate: {}".format(win/EPISODES, loose/EPISODES))