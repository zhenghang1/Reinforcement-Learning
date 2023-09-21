
import numpy as np
import time
from tabulate import tabulate

def maxValueIndex(l):
    max_value = max(l)
    index = []
    for i in range(len(l)):
        if l[i] == max_value:
            index.append(i)
    return index

class GridWorld:
    def __init__(self, width, height) -> None:
        # create grid world, where 0 means non-terminal and 1 means terminal
        self.width, self.height = width, height
        self.grid = [[0 for i in range(self.width)] for i in range(self.height)]
        self.action = ['east','south','west','north']

    def setTerminal(self,x,y):
        self.grid[x][y] = 1

    def isTerminal(self, state):
        x,y = state
        if self.grid[x][y] == 1:
            return True
        else:
            return False

    def getReward(self, state, action, next_state):
        if self.isTerminal(state):
            return 0.0
        else:
            return -1.0
        
    def getStates(self):
        return [(i,j) for i in range(self.height) for j in range(self.width)]
    
    def getPossibleActions(self,state):
        if self.isTerminal(state):
            return []
        else:
            return self.action
        
    def isAllowed(self, x, y):
        if y < 0 or y >= self.width: return False
        if x < 0 or x >= self.height: return False
        return True

    def getTransition(self):
        states = self.getStates()
        trans = dict()
        for state in states:
            trans[state] = dict()
            actions = self.getPossibleActions(state)
            if self.isTerminal(state):
                continue
            x,y = state
            northState = (x-1,y) if (self.isAllowed(x-1,y)) else state
            westState = (x,y-1) if (self.isAllowed(x,y-1)) else state
            southState = (x+1,y) if (self.isAllowed(x+1,y)) else state
            eastState = (x,y+1) if (self.isAllowed(x,y+1)) else state
            next_state = [eastState,southState,westState,northState]
            for i, action in enumerate(actions):
                trans[state][action] = [(next_state[i],1)]
        return trans
    

def PolicyEvaluation(epsilon, states, values, policy, transition, discount, gridworld):
    max_margin = epsilon
    # store the new value of each state after a new iteration
    new_value = {state: 0 for state in states}

    count = 0
    while max_margin >= epsilon:
        max_margin = 0
        for state in states:
            # terminal state
            if not policy[state]:
                continue
            # Qvalue with current policy
            new_value[state] = ComputeQValueFromValues(state, policy[state], transition, values, discount, gridworld)
        max_margin = max(map(lambda state:abs(new_value[state] - values[state]),states))
        for state in states:
            if not policy[state]:
                continue
            values[state] = new_value[state]
        count += 1
    return values, count

def ComputeQValueFromValues(state, action, trans, values, discount, gridworld):
    if not trans[state]:
        return 0
    state_prob_list = trans[state][action]
    value = sum(map(lambda state_prob: state_prob[1]*(gridworld.getReward(
        state, action, state_prob[0])+discount*values[state_prob[0]]), state_prob_list)) 
    return value

class RandomPolicyEvaluation():
    def __init__(self,grid,epsilon=1e-3,discount=1) -> None:
        self.grid = grid
        self.epsilon = epsilon
        self.discount = discount
        self.values = self.runRandomPolicyEvaluation()
        self.policy = self.extractPolicy()
        
    def runRandomPolicyEvaluation(self):
        start_time = time.time()
        states = self.grid.getStates()
        transition = self.grid.getTransition()
        available_actions = {state: self.grid.getPossibleActions(state) for state in states}
        values = {state:0.0 for state in states}

        max_margin = self.epsilon
        # store the new value of each state after a new iteration
        new_value = {state: 0 for state in states}
        self.count = 0
        while max_margin >= self.epsilon:
            max_margin = 0
            new_value = {state: 0 for state in states}
            for state in states:
                if self.grid.isTerminal(state):
                    continue
                for action in available_actions[state]:
                    new_value[state] += ComputeQValueFromValues(state, action, transition, values, self.discount, self.grid)
                if len(available_actions[state]) != 0:
                    new_value[state] /= len(available_actions[state])
            max_margin = max(map(lambda state:abs(new_value[state] - values[state]),states))
            for state in states:
                values[state] = new_value[state]
            self.count += 1
        values = {state:round(values[state],4) for state in values.keys()}
        self.time = time.time()-start_time
        return values
    
    def extractPolicy(self):
        trans = self.grid.getTransition()
        states = self.grid.getStates()
        policy = {state:[] for state in states}
        for state in states:
            actions = self.grid.getPossibleActions(state)
            values = list(map(lambda action: self.values[trans[state][action][0][0]],actions))
            if len(values)!=0:
                index = maxValueIndex(values)
                policy[state] = [actions[i] for i in index]
        return policy
        
    def printValuesAndPolicy(self):
        # plot value table
        values = list(self.values.values())
        length = int(len(values)/6)
        value_list = [values[6*i:6*(i+1)] for i in range(length)]
        print("Value of the Gridworld:")
        print(tabulate(value_list, tablefmt='fancy_grid'))
        print('')

        # plot policy table
        policy = list(self.policy.values())
        policy_list = [policy[6*i:6*(i+1)] for i in range(length)]
        print("Extracted Policy of the Gridworld:")
        print(tabulate(policy_list, tablefmt='fancy_grid'))

    def getPolicy(self):
        return self.policy,self.count, self.time

class PolicyAgent:
    def __init__(self, grid, discount=1, epsilon=0.001, iterations=100) -> None:
        self.grid = grid
        self.discount = discount
        self.epsilon = epsilon  # For examing the convergence of policy iteration
        self.iterations = iterations  # The policy iteration will run AT MOST these steps
        self.values = dict()
        self.policy = dict()
        self.transition = self.grid.getTransition()

    def initPolicy(self,policy,count,time):
        for state in policy.keys():
            if len(policy[state])==0:
                self.policy[state] = None
            else:
                self.policy[state] = policy[state][0]
        self.count = count
        self.time = time

    def runPolicyIteration(self):
        start_time = time.clock()
        states = self.grid.getStates()
        # initialize self.values
        self.values = {state:0 for state in states}
        # policy_done is a dict to store whether a state's policy has converged, if yes, policy_done[state]=1, otherwise 0
        policy_done = {state: 0 for state in states}
        # policy_done_flag is 1 when all the policies converge, otherwise 0
        policy_done_flag = 0
        # Value function updating counts
        # Start Iteration
        for i in range(self.iterations):
            # policy evaluation
            self.values,c = PolicyEvaluation(self.epsilon, states, self.values, self.policy, 
                                           self.transition, self.discount, self.grid)
            self.count += c
            # policy improvement
            for state in states:
                new_policy = self.computeActionFromValues(state)
                if new_policy != self.policy[state]:
                    # policy hasn't converge
                    self.policy[state] = new_policy
                    policy_done[state] = 0
                else:
                    # policy converges
                    policy_done[state] = 1
            # see whether all policies converge
            policy_done_flag = min(policy_done.values())
            # converge, early stop
            if policy_done_flag:
                print("In PolicyIteration, converge at iteration: {}, total value updating times: {}".format(i+1,self.count),flush=True)
                break
        end_time = time.clock()
        print("\nRunning time: %.8f s" % (end_time - start_time + self.time),flush=True)            

    def computeActionFromValues(self,state):
        bestaction = None
        if self.grid.isTerminal(state):
            return bestaction
        # get avaliable actions under current state
        available_actions = self.grid.getPossibleActions(state)
        best_value = -100
        # find the max Qvalue and its corresponding action
        for action in available_actions:
            value = ComputeQValueFromValues(state, action, self.transition, self.values, self.discount,self.grid)
            if value > best_value:
                best_value = value
                bestaction = action
        return bestaction

    def printValuesAndPolicy(self):
        # plot value table
        values = list(self.values.values())
        length = int(len(values)/6)
        value_list = [values[6*i:6*(i+1)] for i in range(length)]
        print("\nValue of the Gridworld:")
        print(tabulate(value_list, tablefmt='fancy_grid'))
        print('')

        # plot policy table
        tmp = list(self.policy.values())
        policy = []
        for p in tmp:
            if not p:
                policy.append([])
            else:
                policy.append([p])
        policy_list = [policy[6*i:6*(i+1)] for i in range(length)]
        print("Extracted Policy of the Gridworld:")
        print(tabulate(policy_list, tablefmt='fancy_grid'))


class ValueAgent():
    def __init__(self, grid, discount=1, epsilon=0.001, iterations=100) -> None:
        self.grid = grid
        self.discount = discount
        self.epsilon = epsilon  # For examing the convergence of policy iteration
        self.iterations = iterations  # The policy iteration will run AT MOST these steps
        self.transition = self.grid.getTransition()
        self.values = dict()

    def runValueIteration(self):
        start_time = time.clock()
        # get all the states
        states = self.grid.getStates()
        # get all availbale actions for each state and store in a dict
        self.values = {state:0 for state in states}
        available_actions = {
            state: self.grid.getPossibleActions(state) for state in states}
        # store the new value of each state after a new iteration
        new_value = {state: 0 for state in states}
        for i in range(self.iterations):
            max_margin = 0
            for state in states:
                # terminal state
                if self.grid.isTerminal(state):
                    continue
                # max Qvalue
                new_value[state] = max(map(lambda action: ComputeQValueFromValues(
                    state, action, self.transition, self.values, self.discount, self.grid), available_actions[state]))
            max_margin = max(map(lambda state: abs(new_value[state] - self.values[state]), states))
            for state in states:
                self.values[state] = new_value[state]
            # converge, early stop
            if max_margin < self.epsilon:
                print("In ValueIteration, converge at iteration: {}, total value updating times: {}".format(i+1,i+1))
                break
        end_time = time.clock()
        print("\nRunning time: %.8f s" % (end_time - start_time)) 

    def extractPolicy(self):
        trans = self.grid.getTransition()
        states = self.grid.getStates()
        policy = {state:[] for state in states}
        for state in states:
            actions = self.grid.getPossibleActions(state)
            values = list(map(lambda action: self.values[trans[state][action][0][0]],actions))
            if len(values)!=0:
                index = maxValueIndex(values)
                policy[state] = [actions[i] for i in index]
        return policy

    def printValuesAndPolicy(self):
        # plot value table
        values = list(self.values.values())
        length = int(len(values)/6)
        value_list = [values[6*i:6*(i+1)] for i in range(length)]
        print("\nValue of the Gridworld:")
        print(tabulate(value_list, tablefmt='fancy_grid'))
        print('')

        # Extract policy out of values
        policy = self.extractPolicy()
        
        # plot policy table
        policy = list(policy.values())
        policy_list = [policy[6*i:6*(i+1)] for i in range(length)]
        print("Extracted Policy of the Gridworld:")
        print(tabulate(policy_list, tablefmt='fancy_grid'))