
s = 10

for i in range(100):
    pre_action = i
    action = min(pre_action, s, 100-s)
    s += action
    print("action {} sum of money {}".format(action, s))
    if s >= 100:
        break;