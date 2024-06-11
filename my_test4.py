import numpy as np
import matplotlib.pyplot as plt

# Random policy
# Assume there is a linear grid, the chance of choosing each direction is same
# The objective is to reach the two ends

stNum = 100
epsThr = 1.0e-5
probH = 0.4

states = list(range(0,stNum+1))
valVec = np.zeros(stNum+1)
policy = [0 for ii in range(stNum+1)]
# valVec[stNum]=1.0
rewVec = np.zeros(stNum+1)
rewVec[stNum] = 1.0

# Function for value evaluation

deltaVal = 1
tmpval = 0
runTime = 1
# while (deltaVal > epsThr):
while True:
    deltaVal = 0.0
    gf=[]
    for qq in (range(1,stNum)):
        oldVal = valVec[qq]
        optValTmp = valVec[qq]
        optSol = policy[qq]
        for ee in range(min(qq+1,stNum+1-qq)):
            futVal = probH * (rewVec[qq+ee] + valVec[qq+ee]) \
                     + (1 - probH) * (rewVec[qq-ee] + valVec[qq-ee])
            if futVal > optValTmp+epsThr:
                optValTmp = futVal
                optSol = ee
                valVec[qq] = futVal
                policy[qq] = ee
        gg = abs(valVec[qq]-oldVal)
        gf.append(gg)
    if (max(gf)<epsThr) and (min(gf)<epsThr/10.0):
        break
    # deltaVal = max(deltaVal,gg)
    runTime += 1
# print(deltaVal)

print(len(states))
# print(len(policy))
print('optimal policy')
print(policy)

print(valVec)
print(runTime)
plt.plot(states, policy)
# plt.plot(valVec)
plt.show()