import time
from A3C import A3C_Agent
from DDPG import DDPG_Agent
import sys
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-m','--method', default='DDPG', type=str, help='Methods')
parser.add_argument('-e','--episode', default=1500, type=int, help='number of episodes')
parser.add_argument('-c','--cuda', default=0, type=int, help='cuda index')
parser.add_argument('--env_name', default='Pendulum-v1', type=str, help='gym environment name')
parser.add_argument('-b','--buffer_capacity', default=10000, type=int, help='Buffer capacity')
parser.add_argument('--batch_size', default=128, type=int, help='Batch size')
parser.add_argument('-g','--gamma', default=0.9, type=float, help='gamma in value updating')
parser.add_argument('-t','--tau', default=0.001, type=float, help='tau in DDPG network updating')
parser.add_argument('--noise_std', default=0.2, type=float, help='tau in DDPG network updating')
parser.add_argument('--entropy_beta', default=0.01, type=float, help='entropy beta in policy loss calculation')
parser.add_argument('--policy_lr', default=1e-4, type=float, help='Learning rate in policy network')
parser.add_argument('--value_lr', default=0.002, type=float, help='Learning rate in value network')
parser.add_argument('--actor_lr', default=1e-4, type=float, help='Learning rate in policy network')
parser.add_argument('--critic_lr', default=0.002, type=float, help='Learning rate in value network')
parser.add_argument('-i','--update_interval', default=5, type=int, help='Interval between updating of global networks')
parser.add_argument('-l','--max_episode_length', default=200, type=int, help='Maximum episode length')
parser.add_argument('--test_episode', default=10, type=int, help='Episodes taken when testing')
parser.add_argument('-n','--num_actors', default=4, type=int, help='Number of actors')
parser.add_argument('--idx', default=0, type=int)
args = parser.parse_args()

if __name__ == "__main__":
    s_time = time.time()
    savedStdout = sys.stdout
    os.makedirs(f'{args.method}_output',exist_ok=True)
    if args.method == 'A3C':
        agent = A3C_Agent(args=args)

        # redirect output
        output_name = f'{args.method}_output/actors{args.num_actors}_interval{args.update_interval}_gamma{args.gamma}_idx{args.idx}.txt'
        f = open(output_name,'w+',encoding="utf-8")
        sys.stdout = f
    elif args.method == 'DDPG':
        agent = DDPG_Agent(args=args)

        # redirect output
        output_name = f'{args.method}_output/tau{args.tau}_interval{args.update_interval}_gamma{args.gamma}_l{args.max_episode_length}_idx{args.idx}.txt'
        f = open(output_name,'w+',encoding="utf-8")
        sys.stdout = f
    else:
        raise ValueError("method should be in [A3C, DDPG]")

    agent.train()

    print(f"Total training time {time.time()-s_time}",flush=True)

    agent.test()

    agent.plot_reward()

    # close the output file and reset stdout
    f.close()
    sys.stdout = savedStdout
