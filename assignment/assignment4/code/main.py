from model import Agent
import sys
import argparse
import gym

parser = argparse.ArgumentParser()
parser.add_argument('-m','--model_type', default='DQN', type=str, help='Model type')
parser.add_argument('-i','--iteration', default=500, type=int, help='iterations')
parser.add_argument('-b','--buffer_capacity', default=1000, type=int, help='Buffer capacity')
parser.add_argument('--batch_size', default=128, type=int, help='Batch size')
parser.add_argument('-e','--epsilon', default=0.01, type=float, help='epsilon in epsilon greedy selection')
parser.add_argument('-o','--output_file', default='./output/output.txt', type=str, help='output file path to save the results')
parser.add_argument('-a','--alpha', default=0.9, type=float, help='step size in TD learning')
parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate')
parser.add_argument('-s','--step', default=300, type=int, help='Step size in Fixed Target Network')
parser.add_argument('-r','--reward', default=1, type=int, help='Reward type')
parser.add_argument('-n','--network', default=1, type=int, help='Network type')
args = parser.parse_args()


if __name__ == "__main__":
    # redirect output
    savedStdout = sys.stdout
    output_name = args.model_type + '_alpha'+ str(args.alpha) + '_network'+ str(args.network) + '_reward' + str(args.reward) + '.txt'
    f = open(output_name,'w+',encoding="utf-8")
    sys.stdout = f

    env = gym.make("MountainCar-v0")
    env = env.unwrapped

    agent = Agent(env=env,network_type=args.network,model_type=args.model_type,buffer_capacity=args.buffer_capacity,
                  epsilon=args.epsilon,fix_step=args.step,reward_type=args.reward)
    agent.train(args.iteration,args.batch_size,args.lr,args.alpha)
    agent.test(test_episode=5)

    agent.plot_loss()

    # close the output file and reset stdout
    f.close()
    sys.stdout = savedStdout
