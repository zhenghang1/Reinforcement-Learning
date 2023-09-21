import time
import random
import numpy as np
from tabulate import tabulate


def maxValueIndex(l):
    max_value = max(l)
    index = []
    for i in range(len(l)):
        if l[i] == max_value:
            index.append(i)
    return index


class GridWorld:
    def __init__(self, height, width) -> None:
        # create grid world, where 0 means non-terminal, 1 means terminal, 2 means start, -1 means cliff
        self.width, self.height = width, height
        self.grid = [[0 for i in range(self.width)]
                     for i in range(self.height)]
        self.action = ['east', 'south', 'west', 'north']

    def setTerminal(self, x, y):
        self.grid[x][y] = 1

    def setStart(self, x, y):
        self.grid[x][y] = 2
        self.start_state = (x, y)

    def setCliff(self, x, y):
        self.grid[x][y] = -1

    def getStart(self):
        return self.start_state

    def isTerminal(self, state):
        x, y = state
        if self.grid[x][y] == 1:
            return True
        else:
            return False

    def isCliff(self, state):
        x, y = state
        if self.grid[x][y] == -1:
            return True
        else:
            return False

    def getReward(self, state, action, next_state):
        if self.isTerminal(state):
            return 0.0
        elif self.isCliff(next_state):
            return -100.0
        else:
            return -1.0

    def getStates(self):
        return [(i, j) for i in range(self.height) for j in range(self.width)]

    def getPossibleActions(self, state):
        if self.isTerminal(state) or self.isCliff(state):
            return []
        else:
            return self.action

    def isAllowed(self, x, y):
        if y < 0 or y >= self.width:
            return False
        if x < 0 or x >= self.height:
            return False
        return True

    def getTransition(self):
        states = self.getStates()
        trans = dict()
        for state in states:
            trans[state] = dict()
            actions = self.getPossibleActions(state)
            if self.isTerminal(state):
                continue
            x, y = state
            northState = (x-1, y) if (self.isAllowed(x-1, y)) else state
            westState = (x, y-1) if (self.isAllowed(x, y-1)) else state
            southState = (x+1, y) if (self.isAllowed(x+1, y)) else state
            eastState = (x, y+1) if (self.isAllowed(x, y+1)) else state
            next_state = [eastState, southState, westState, northState]
            for i, action in enumerate(actions):
                trans[state][action] = [(next_state[i], 1)]
        return trans

    def getRandomState(self):
        while True:
            ran_x = random.randint(0, self.height-1)
            ran_y = random.randint(0, self.width-1)
            if not self.isTerminal((ran_x, ran_y)):
                return (ran_x, ran_y)


class MDP_Agent():
    def __init__(self, mdp, discount=1, epsilon=0.2, alpha=0.01) -> None:
        self.mdp = mdp
        self.discount = discount
        self.alpha = alpha
        self.epsilon = epsilon
        self.states = self.mdp.getStates()
        self.actions = self.mdp.action

    def runSarsa(self, iteration):
        self.Qvalues = {state: {action: 0 for action in self.actions}
                        for state in self.states}
        trans = self.mdp.getTransition()
        for i in range(iteration):
            state = self.mdp.getStart()
            action = self.getNextAction(state)
            while True:
                margin = 0
                if self.mdp.isTerminal(state):
                    break
                state_prob_list = trans[state][action]
                ran = random.randint(0, len(state_prob_list)-1)
                next_state, prob = state_prob_list[ran]
                reward = self.mdp.getReward(state, action, next_state)
                if self.mdp.isCliff(next_state):
                    next_state = self.mdp.getStart()
                next_action = self.getNextAction(next_state,epsilon_greedy=True)

                # Update Q values
                delta = self.alpha * (reward+self.discount*self.Qvalues[next_state][next_action]-self.Qvalues[state][action])
                self.Qvalues[state][action] = self.Qvalues[state][action] + delta

                state = next_state
                action = next_action

    def runQLearning(self, iteration):
        self.Qvalues = {state: {action: 0 for action in self.actions}
                        for state in self.states}
        trans = self.mdp.getTransition()
        for i in range(iteration):
            state = self.mdp.getStart()
            while True:
                if self.mdp.isTerminal(state):
                    break
                action = self.getNextAction(state,epsilon_greedy=True)
                state_prob_list = trans[state][action]
                ran = random.randint(0, len(state_prob_list)-1)
                next_state, prob = state_prob_list[ran]
                reward = self.mdp.getReward(state, action, next_state)
                if self.mdp.isCliff(next_state):
                    next_state = self.mdp.getStart()
                next_action = self.getNextAction(next_state,epsilon_greedy=False)

                # Update Q values
                delta = self.alpha * (reward+self.discount*self.Qvalues[next_state][next_action]-self.Qvalues[state][action])
                self.Qvalues[state][action] = self.Qvalues[state][action] + delta

                state = next_state

    def getNextAction(self, state, epsilon_greedy=True):
        m = len(self.actions)
        best_action = max(self.Qvalues[state],key=lambda x:self.Qvalues[state][x])
        if not epsilon_greedy or self.epsilon == 0:
            return best_action

        p = random.randint(0,1000000) * 1e-6
        assert p>=0 and p<=1
        if p < self.epsilon:
            while True:
                ran = random.randint(0,m-1)
                if self.actions[ran] != best_action:
                    return self.actions[ran] 
        else:
            return best_action
    
    def extractPolicy(self):
        states = self.mdp.getStates()
        policy = []
        for state in states:
            if self.mdp.isTerminal(state) or self.mdp.isCliff(state):
                policy.append(None)
            else:
                policy.append(max(self.Qvalues[state],key=lambda x:self.Qvalues[state][x]))
        return policy
    
    def extractPath(self):
        state = self.mdp.getStart()
        trans = self.mdp.getTransition()
        tmp = {}
        while not self.mdp.isTerminal(state):
            best_action = max(self.Qvalues[state],key=lambda x:self.Qvalues[state][x])
            tmp[state] = best_action
            # move to next state
            state_prob_list = trans[state][best_action]
            ran = random.randint(0, len(state_prob_list)-1)
            next_state, prob = state_prob_list[ran]
            state = next_state
        path = []
        for state in self.states:
            if state in tmp:
                path.append(tmp[state])
            else:
                path.append(None)
        return path

    def trans_Qvalues(self):
        for x in range(self.mdp.height):
            for y in range(self.mdp.width):
                state = (x,y)
                if not self.mdp.isAllowed(x+1,y):
                    self.Qvalues[state]['south']  = -1000
                if not self.mdp.isAllowed(x-1,y):
                    self.Qvalues[state]['north']  = -1000
                if not self.mdp.isAllowed(x,y+1):
                    self.Qvalues[state]['east']  = -1000
                if not self.mdp.isAllowed(x,y-1):
                    self.Qvalues[state]['west']  = -1000

    def printValuesAndPolicy(self):
        self.trans_Qvalues()

        # plot value table
        Qvalues = [list(self.Qvalues[state].values()) for state in self.states]
        values = list(map(max,Qvalues))
        length = int(len(values)/self.mdp.width)
        value_list = [values[self.mdp.width*i:self.mdp.width*(i+1)] for i in range(length)]
        print("\nValue of the Gridworld:", flush=True)
        print(tabulate(value_list, tablefmt='fancy_grid'), flush=True)
        print('', flush=True)

        # plot policy table
        policy = self.extractPolicy()
        policy_list = [policy[self.mdp.width*i:self.mdp.width*(i+1)] for i in range(length)]
        print("Extracted Policy of the Gridworld:", flush=True)
        print(tabulate(policy_list, tablefmt='fancy_grid'), flush=True) 

        # plot path table
        path = self.extractPath()
        path_list = [path[self.mdp.width*i:self.mdp.width*(i+1)] for i in range(length)]
        print("\nExtracted Path :", flush=True)
        print(tabulate(path_list, tablefmt='fancy_grid'), flush=True) 

