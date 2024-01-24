import numpy as np

"""
q[(s, a)] object
"""
class doubleDefaultDict:
    def __init__(self):
        self.obj = {}

    def getActions(self, state):
        # if actions have not been recorded before, return empty dict
        if state not in self.obj:
            self.obj[state] = {}
        return self.obj[state]

    def __getitem__(self, stateAction):
        state, action = stateAction

        if state not in self.obj:
            self.obj[state] = {}

        # by default if item doesn't exist, return 0
        if action not in self.obj[state]:
            self.obj[state][action] = np.float("-inf")

        return self.obj[state][action]

    def __setitem__(self, stateAction, value):
        state, action = stateAction

        if state not in self.obj:
            self.obj[state] = {}

        self.obj[state][action] = value

"""
convert state/action array to dictionary key
将给定的状态或动作数组（arr）转换为一个字符串，用于作为字典的键
"""
def array2key(arr):
    reshapeArr = arr.flatten()
    reshapeArr = [str(a) for a in reshapeArr]
    return "_".join(reshapeArr)

"""
convert dictionary key to state/action array
将由 array2key 函数生成的字符串键转换回原始的状态或动作数组
"""
def key2array(key):
    arr = key.split("_")
    arr = np.array([np.float(a) for a in arr])
    return arr

"""
find the maximum (key, value) in dictionary
在给定的字典中找到具有最大值的键值对，并返回其中一个最大值的键值对。如果有多个最大值，函数会随机选择一个。
"""
def getMaxDict(dicts):
    # if dictionary is empty
    if (len(dicts) == 0):
        return (None, np.float("-inf"))

    setKeysMax = []
    maxVal = np.float("-inf")
    for k, v in dicts.items():
        if v == maxVal:
            setKeysMax.append(k)
        elif v > maxVal:
            maxVal = v
            setKeysMax = [k]

    # choose randomly. return (maxState, maxVal)
    return (np.random.choice(setKeysMax), maxVal)