from model import Agent
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--env', default=0, type=int, help='Arita environment index')
parser.add_argument('--env_name', default='PongNoFrameskip-v4', type=str, help='Mujoco environment name')
parser.add_argument('--conv',action='store_true',default=False)
parser.add_argument('-m','--model_type', default='DQN', type=str, help='Model type')
parser.add_argument('-e','--episodes', default=5000, type=int, help='Episodes')
parser.add_argument('-b','--buffer_capacity', default=400000, type=int, help='Buffer capacity')
parser.add_argument('--batch_size', default=32, type=int, help='Batch size')
parser.add_argument('--epsilon', default=0.01, type=float, help='epsilon in epsilon greedy selection')
parser.add_argument('--max_steps', default=50000, type=int, help='Max steps in an episode')
parser.add_argument('--discount_factor', default=0.99, type=float, help='step size in TD learning')
parser.add_argument('--tau', default=1e-3, type=float, help='tau')
parser.add_argument('--lr', default=0.00025, type=float, help='Learning rate')
parser.add_argument('--seed', default=20, type=int, help='random seed')
parser.add_argument('--n_step', default=1, type=int, help='N-step experience')
parser.add_argument('--min_epsilon', default=0.05, type=float, help='minimum epsilon')
parser.add_argument('--epsilon_frames', default=1000000, type=int, help='Max frames using epsilon greedy')
parser.add_argument('-s','--step', default=300, type=int, help='Step size in Fixed Target Network')
parser.add_argument('--save_model',action='store_true',default=False)
parser.add_argument('--per',action='store_true',default=False)
parser.add_argument('--noisy',action='store_true',default=False)
parser.add_argument('--pre_fill_buffer', default=50000, type=int, help='Pre-fill buffer size')
parser.add_argument('--atom_size', default=51, type=int, help='Atom size')
parser.add_argument('--vmax', default=10, type=int, help='')
parser.add_argument('--vmin', default=-10, type=int, help='')
parser.add_argument('-c','--cuda', default=0, type=int, help='Cuda index')
parser.add_argument('-u','--update_frequency', default=100, type=int, help='update frequency')


args = parser.parse_args()

arita_env = ['VideoPinball-ramNoFrameskip-v4','BreakoutNoFrameskip-v4','PongNoFrameskip-v4','BoxingNoFrameskip-v4']
env_name = arita_env[args.env]

if __name__ == "__main__":
    # redirect output
    savedStdout = sys.stdout
    output = env_name +'_'+ args.model_type + '_' + str(args.per+args.noisy) + '.txt'
    f = open(output,'w+',encoding="utf-8")
    sys.stdout = f

    agent = Agent(env_name=env_name,args=args,output=output)
    agent.train(args.episodes,eps_frames=args.epsilon_frames,min_eps=args.min_epsilon)

    # close the output file and reset stdout
    f.close()
    sys.stdout = savedStdout
