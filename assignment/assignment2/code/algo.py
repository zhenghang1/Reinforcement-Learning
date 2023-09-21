import time
import random
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
    
    def getRandomState(self):
        while True:
            ran_x = random.randint(0,self.height-1)
            ran_y = random.randint(0,self.width-1)
            if not self.isTerminal((ran_x,ran_y)):
                return (ran_x,ran_y)

class MDP_Agent():
    def __init__(self,mdp,path_length=1000,discount=1,epsilon=5e-4,successful_count=3) -> None:
        self.mdp = mdp
        self.path_length = path_length
        self.discount = discount
        self.epsilon = epsilon
        self.successful_count = successful_count
        self.states = self.mdp.getStates()

    def runFirstVisit_MC(self, iteration):
        start_time = time.time()
        trans = self.mdp.getTransition()
        counter = {state:0 for state in self.states}
        values = {state:0 for state in self.states}
        successful_count = 0
        iter = 0
        while iter < iteration or successful_count < self.successful_count:
            # one iteration is an episode
            while True:
                visited = set()
                state_reward_path = []
                state = self.mdp.getRandomState()
                # generate state path and record the reward 
                while not self.mdp.isTerminal(state):
                    if state not in visited:
                        visited.add(state)
                        counter[state] += 1
                    actions = self.mdp.getPossibleActions(state)
                    ran = random.randint(0,len(actions)-1)
                    action = actions[ran]
                    state_prob_list = trans[state][action]
                    ran = random.randint(0,len(state_prob_list)-1)
                    next_state, prob = state_prob_list[ran]
                    reward = self.mdp.getReward(state,action,next_state)
                    state_reward_path.append((state,reward))
                    state = next_state
                if len(state_reward_path) <= self.path_length:
                    break
            state_reward_path.reverse()
            G_t = {state:0 for state in self.states}
            total_reward = 0
            for i in range(len(state_reward_path)):
                state, reward = state_reward_path[i]
                total_reward = total_reward * self.discount + reward
                G_t[state] = total_reward
            margin = 0
            for state in self.states:
                if state in visited:
                    delta = (G_t[state]-values[state]) / counter[state]
                    values[state] = values[state] + delta
                    margin = max(margin,abs(delta))
            if margin < self.epsilon:
                successful_count += 1
            iter += 1
        end_time = time.time()
        print("Running FirstVisit_MC for {} episodes, total time spent: {}".format(iter,end_time-start_time),flush=True)
        self.values = values

    def runEveryVisit_MC(self, iteration):
        start_time = time.time()
        trans = self.mdp.getTransition()
        counter = {state:0 for state in self.states}
        values = {state:0 for state in self.states}
        successful_count = 0
        iter = 0
        while iter < iteration or successful_count < self.successful_count:
            # one iteration is an episode
            while True:
                state_reward_path = []
                state = self.mdp.getRandomState()
                # generate state path and record the reward 
                while not self.mdp.isTerminal(state):
                    counter[state] += 1
                    actions = self.mdp.getPossibleActions(state)
                    ran = random.randint(0,len(actions)-1)
                    action = actions[ran]
                    state_prob_list = trans[state][action]
                    ran = random.randint(0,len(state_prob_list)-1)
                    next_state, prob = state_prob_list[ran]
                    reward = self.mdp.getReward(state,action,next_state)
                    state_reward_path.append((state,reward))
                    state = next_state
                if len(state_reward_path) <= self.path_length:
                    break
            state_reward_path.reverse()
            total_reward = 0
            margin = 0
            for i in range(len(state_reward_path)):
                state, reward = state_reward_path[i]
                total_reward = total_reward * self.discount + reward
                G_t = total_reward
                delta = (G_t-values[state]) / counter[state]
                values[state] = values[state] + delta
                margin = max(margin,abs(delta))
            if margin < self.epsilon:
                successful_count += 1
            iter += 1
        end_time = time.time()
        print("Running EveryVisit_MC for {} episodes, total time spent: {}".format(iter,end_time-start_time),flush=True)
        self.values = values

    def setEpsilon(self,e):
        self.epsilon = e

    def runTD_learning(self, iteration, alpha):
        start_time = time.time()
        trans = self.mdp.getTransition()
        values = {state:0 for state in self.states}
        successful_count = 0
        iter = 0
        while iter < iteration or successful_count < self.successful_count:
            # one iteration is an episode
            while True:
                state_reward_path = []
                state = self.mdp.getRandomState()
                # generate state path and record the reward 
                while not self.mdp.isTerminal(state):
                    actions = self.mdp.getPossibleActions(state)
                    ran = random.randint(0,len(actions)-1)
                    action = actions[ran]
                    state_prob_list = trans[state][action]
                    ran = random.randint(0,len(state_prob_list)-1)
                    next_state, prob = state_prob_list[ran]
                    reward = self.mdp.getReward(state,action,next_state)
                    state_reward_path.append((state,reward,next_state))
                    state = next_state
                if len(state_reward_path) <= self.path_length:
                    break
            margin = 0
            for i in range(len(state_reward_path)):
                state, reward, next_state = state_reward_path[i]
                delta = alpha * (reward + self.discount * values[next_state] - values[state])
                values[state] = values[state] + delta
                margin = max(margin,abs(delta))
            if margin < self.epsilon:
                successful_count += 1
            iter += 1
        end_time = time.time()
        print("Running EveryVisit_MC for {} episodes, total time spent: {}".format(iter,end_time-start_time),flush=True)
        self.values = values

    def extractPolicy(self):
        trans = self.mdp.getTransition()
        states = self.mdp.getStates()
        policy = {state:[] for state in states}
        for state in states:
            actions = self.mdp.getPossibleActions(state)
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
        print("\nValue of the Gridworld:",flush=True)
        print(tabulate(value_list, tablefmt='fancy_grid'),flush=True)
        print('',flush=True)

        # plot policy table
        policy= self.extractPolicy()
        policy  = list(policy.values())
        policy_list = [policy[6*i:6*(i+1)] for i in range(length)]
        print("Extracted Policy of the Gridworld:",flush=True)
        print(tabulate(policy_list, tablefmt='fancy_grid'),flush=True)

