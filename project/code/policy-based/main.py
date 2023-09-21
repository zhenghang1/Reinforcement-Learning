import time
from model import Agent
import sys
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--env_name', default='Ant-v2', type=str, help='Mujoco environment name')
parser.add_argument('-m','--model_type', default='PPO', type=str, help='Model type')
parser.add_argument('-e','--episodes', default=5000, type=int, help='Episodes')
parser.add_argument('--max_steps', default=5000, type=int, help='Max steps in a episode')
parser.add_argument('-b','--buffer_capacity', default=400000, type=int, help='Buffer capacity')
parser.add_argument('--hidden_dim', default=32, type=int, help='Hidden layer dim')
parser.add_argument('--batch_size', default=128, type=int, help='Batch size')
parser.add_argument('--discount_factor', default=0.99, type=float, help='step size in TD learning')
parser.add_argument('--tau', default=5e-3, type=float, help='tau')
parser.add_argument('--actor_lr', default=3e-4, type=float, help='Learning rate')
parser.add_argument('--critic_lr', default=3e-4, type=float, help='Learning rate')
parser.add_argument('--alpha_lr', default=3e-4, type=float, help='Learning rate')
parser.add_argument('--alpha_init', default=0.2, type=float, help='Alpha in SAC')
parser.add_argument('--seed', default=1, type=int, help='random seed')
parser.add_argument('--n_step', default=1, type=int, help='')
parser.add_argument('--save_model',action='store_true',default=False)
parser.add_argument('--pre_fill_buffer', default=5000, type=int, help='Pre-fill buffer size')
parser.add_argument('-c','--cuda', default=0, type=int, help='Cuda index')
parser.add_argument('-u','--update_frequency', default=100, type=int, help='update frequency')


args = parser.parse_args()


if __name__ == "__main__":
    # redirect output
    savedStdout = sys.stdout
    output = args.model_type + args.env_name
    f = open(output + '.txt','w+',encoding="utf-8")
    sys.stdout = f

    agent = Agent(env_name=args.env_name,args=args,output=output)
    agent.train(args.episodes, args.max_steps)

    # close the output file and reset stdout
    f.close()
    sys.stdout = savedStdout
