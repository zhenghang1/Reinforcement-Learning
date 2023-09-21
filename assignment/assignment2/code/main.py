from algo import GridWorld,MDP_Agent
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-e','--episode', default=10000, type=int, help='minimum episodes')
parser.add_argument('-d','--discount', default=1.0, type=float, help='discounting factor')
parser.add_argument('--mc_epsilon', default=1e-4, type=float, help='epsilon in MC learning')
parser.add_argument('--td_epsilon', default=0.1, type=float, help='epsilon in TD learning')
parser.add_argument('-o','--output_file', default='./output/output.txt', type=str, help='output file path to save the results')
parser.add_argument('-l','--path_length', default=1000, type=int, help='length limitation to the random generated state path')
parser.add_argument('-s','--successful_count', default=3, type=int, help='successful counts to make the algorithm more stable')
parser.add_argument('-a','--alpha', default=0.01, type=float, help='step size in TD learning')
args = parser.parse_args()


if __name__ == "__main__":
    # redirect output
    savedStdout = sys.stdout
    output_name = args.output_file
    f = open(output_name,'w+',encoding="utf-8")
    sys.stdout = f

    # Create a Gridworld with size 6*6
    grid = GridWorld(6,6)
    grid.setTerminal(0,1)
    grid.setTerminal(5,5)

    agent = MDP_Agent(grid,discount=args.discount,path_length=args.path_length)

    # First Visit
    agent.setEpsilon(args.mc_epsilon)
    print("--------------------------------------Monte-Carlo First Visit-----------------------------------------",flush=True)
    agent.runFirstVisit_MC(args.episode)
    agent.printValuesAndPolicy()

    # Every Visit
    print("\n--------------------------------------Monte-Carlo Every Visit-----------------------------------------",flush=True)
    agent.runEveryVisit_MC(args.episode)
    agent.printValuesAndPolicy()

    # TD(0)
    print("\n--------------------------------------Temporal-Difference Learning-----------------------------------------",flush=True)
    agent.setEpsilon(args.td_epsilon)
    agent.runTD_learning(args.episode,alpha=args.alpha)
    agent.printValuesAndPolicy()

    # close the output file and reset stdout
    f.close()
    sys.stdout = savedStdout
