from model import GridWorld,PolicyAgent,ValueAgent,RandomPolicyEvaluation
import sys

if __name__ == "__main__":
    # redirect output
    savedStdout = sys.stdout
    output_name = './output.txt'
    f = open(output_name,'w+',encoding="utf-8")
    sys.stdout = f

    # Create a Gridworld with size 6*6
    grid = GridWorld(6,6)
    grid.setTerminal(0,1)
    grid.setTerminal(5,5)

    # Random policy evaluation
    print("--------------------------------------Random Policy Evaluation-----------------------------------------")
    agent = RandomPolicyEvaluation(grid)
    agent.printValuesAndPolicy()
    policy, count, t = agent.getPolicy()

    # Policy Iteration
    print("\n-----------------------------------------Policy Iteration--------------------------------------------")
    agent = PolicyAgent(grid)
    agent.initPolicy(policy, count, t)
    agent.runPolicyIteration()
    agent.printValuesAndPolicy()

    # Value Iteration
    print("\n-----------------------------------------Value Iteration--------------------------------------------")
    agent = ValueAgent(grid)
    agent.runValueIteration()
    agent.printValuesAndPolicy()

    # close the output file and reset stdout
    f.close()
    sys.stdout = savedStdout
