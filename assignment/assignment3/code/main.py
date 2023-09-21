from algo import GridWorld,MDP_Agent,MDP_Agent
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i','--iteration', default=100000, type=int, help='minimum iterations')
parser.add_argument('-d','--discount', default=1.0, type=float, help='discounting factor')
parser.add_argument('-e','--epsilon', default=0.2, type=float, help='epsilon in epsilon greedy selection')
parser.add_argument('-o','--output_file', default='./output/output.txt', type=str, help='output file path to save the results')
parser.add_argument('-a','--alpha', default=0.01, type=float, help='step size in TD learning')
args = parser.parse_args()


if __name__ == "__main__":
    # redirect output
    savedStdout = sys.stdout
    output_name = args.output_file
    f = open(output_name,'w+',encoding="utf-8")
    sys.stdout = f

    # Create a Gridworld with size 6*6
    grid = GridWorld(4,12)
    grid.setTerminal(3,11)
    grid.setStart(3,0)
    for i in range(1,11):
        grid.setCliff(3,i)

    agent = MDP_Agent(grid,discount=args.discount,epsilon=args.epsilon,alpha=args.alpha)

    print("--------------------------------------Sarsa-----------------------------------------",flush=True)
    agent.runSarsa(iteration=args.iteration)
    agent.printValuesAndPolicy()

    print("\n--------------------------------------Q Learning-----------------------------------------",flush=True)
    agent.runQLearning(iteration=args.iteration)
    agent.printValuesAndPolicy()

    # close the output file and reset stdout
    f.close()
    sys.stdout = savedStdout
