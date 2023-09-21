import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--env_name', default='VideoPinball-ramNoFrameskip-v4', type=str, help='Mujoco environment name')
parser.add_argument('-m','--model_type', default='DQN', type=str, help='Model type')
parser.add_argument('-c','--cuda', default=0, type=int, help='Cuda index')
parser.add_argument('--per',action='store_true',default=False)
parser.add_argument('--noisy',action='store_true',default=False)
args = parser.parse_args()

arita_env = ['VideoPinball-ramNoFrameskip-v4','BreakoutNoFrameskip-v4','PongNoFrameskip-v4','BoxingNoFrameskip-v4']
mujoco_env = ['Hopper-v2','Humanoid-v2','HalfCheetah-v2','Ant-v2']

env = args.env_name
if env in arita_env:
    command = f'python value-based/main.py --env_name {env} -m {args.model_type} -c {args.cuda}'
    if env != 'VideoPinball-ramNoFrameskip-v4':
        command += ' --conv'
    if args.per:
        command += ' --per'
    if args.noisy:
        command += ' --noisy'
    
    os.system(command)

elif env in mujoco_env:
    command = f'python policy-based/main.py --env_name {env} -m {args.model_type} -c {args.cuda}'
    os.system(command)

else:
    raise ValueError("Error: Invalid env_name!")