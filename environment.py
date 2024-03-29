import numpy as np

class Environment():
    def __init__(self, envParams):
        self.envParams = envParams

    """
    natural state update
    """
    def step(self, state, demand):
        # get newState
        newState = state.copy()

        # check if daysToDelivery has been fulfilled
        for i in range(3):
            # check if waiting for any delivery
            if newState[3, i] > 0:
                # subtract daysToDelivery
                newState[3, i] -= 1

            # fulfill order
            if newState[3, i] == 0:
                # reduce from supplier when inventory is 0
                newState[0, i] += newState[2, i]
                # cut supplier's inventory
                if i > 0:
                    newState[0, i - 1] -= newState[2, i]

                # set pending delivery to 0
                newState[2, i] = 0

        # check for stockout
        retailerState = newState[:, 2]
        if retailerState[0] >= demand:
            # consume inventory
            retailerState[0] -= demand
        else:
            # add to stockout
            retailerState[4] += demand - retailerState[0]
            # set inventory to zero
            retailerState[0] = 0

        # trigger for aking action: retailer's inventory < reorderPoint & not waiting for any delivery
        actionTrigger = (retailerState[0] <= retailerState[1]) & (retailerState[3] == 0)

        # get rewards
        # include state[4] accumulated number of stockouts + state[0] long-term inventory (after delivered to customer)
        # e.g. just before the next reorderPoint
        reward = 0
        reward -= newState[4, 2] * self.envParams["stockoutCost"]  # 由于缺货而引起的总成本
        reward -= (newState[0, 0] * self.envParams["inventoryCost"][0] + newState[0, 1] * self.envParams["inventoryCost"][1])  # 存储在供应链中各级的总库存成本
        reward -= newState[0, 2] * self.envParams["inventoryCost"][2]  # 零售商处未使用的安全库存的成本

        return (newState, actionTrigger, reward)

    """
    execute action if there's any
    """
    def execute(self, state, action):
        newState = state.copy()

        retailerState = newState[:, 2]
        s1State = newState[:, 1]
        s1Action = action[:, 1]
        s0State = newState[:, 0]
        s0Action = action[:, 0]

        # pendingDelivery = ordered items
        newState[2] = action[1]

        # daysToDelivery = serviceTimes + processingTimes
        s0State[3] = 0 + s0State[5]  # s0 has 0 supplier serviceTime
        s1State[3] = s0Action[0] + s1State[5]
        retailerState[3] = s1Action[0]

        # set next reorder point
        newState[1, 2] = action[2, 2]

        # reset stockout
        newState[4] = 0

        return newState