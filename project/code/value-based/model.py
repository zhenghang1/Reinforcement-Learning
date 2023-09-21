from collections import deque
import os
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
from matplotlib import pyplot as plt
from utils import get_device
from agents import DQNAgent,RainbowAgent


def run_random_policy(agent,env_name,size):
    env = gym.make(env_name)
    action_size = env.action_space.n
    state = env.reset()[0]
    score = 0
    episode = 1
    print(f"Running random policy to fill Replay Buffer to size: {size}",flush=True)
    for i in range(size):
        action = np.random.randint(action_size)
        next_state, reward, done, _ , _ = env.step(action)
        agent.buffer.add(state, action, reward, next_state, done)
        score += reward
        state = next_state
        if done:
            print(f"Episode {episode} Score {score}",flush=True)
            episode += 1
            score = 0
            state = env.reset()[0]


class Agent():
    def __init__(self,env_name,args,output) -> None:
        self.env = gym.make(env_name)
        self.state_dim = self.env.observation_space.shape
        self.action_dim = self.env.action_space.n
        self.model_type = args.model_type
        self.device = get_device(cuda=args.cuda)
        self.output = output
        self.save_model = args.save_model
        self.max_steps = args.max_steps
        # 模型类型
        if args.model_type == "DQN" or args.model_type == "DDQN" or args.model_type == "Dueling":
            self.agent = DQNAgent(input_dim=self.state_dim,output_dim=self.action_dim,device=self.device,args=args)
        elif args.model_type == "Rainbow":
            self.agent = RainbowAgent(input_dim=self.state_dim,output_dim=self.action_dim,device=self.device,args=args)
        else:
            raise ValueError("Invalid value-based algorithm! Should be in [DQN, DDQN, Dueling, Rainbow]")

        if args.pre_fill_buffer>0:
            run_random_policy(agent=self.agent,env_name=env_name,size=args.pre_fill_buffer)
            print("Buffer size: ", len(self.agent.buffer),flush=True)

        # set epsilon frames to 0 so no epsilon exploration
        self.eps_fixed = True if args.noisy else False

    def train(self, episodes, eps_frames, min_eps):
        ma_scores = deque(maxlen=100)  # store the last 100 scores (used to find 100 MA)
        eps = 0 if self.eps_fixed else 1
        eps_start = 1
        d_eps = eps_start - min_eps
        print("Running Agent",flush=True)
        total_step = 1
        for episode in range(episodes):
            state = self.env.reset()[0]
            done = False
            episode_reward = 0
            loss_list = []
            step = 0
            actions = []
            while not done and step<self.max_steps:
                # env.render()
                action = self.agent.act(state, eps)
                actions.append(action)
                next_state, reward, done, _, _ = self.env.step(action)
                loss = self.agent.step(state, action, reward, next_state, done)
                if loss:
                    loss_list.append(loss)
                state = next_state
                episode_reward += reward
                step += 1
                total_step += 1
                # only decay epsilon if not fixed
                if not self.eps_fixed:
                    eps = max(eps_start - ((total_step * d_eps) / eps_frames), min_eps)
            # if reward == 0:
            #     print(actions[-10:])
            ma_scores.append(episode_reward)  # save most recent score
            print(f'Episode:{episode} Step:{step} Average Reward: {np.mean(ma_scores)} Reward: {int(episode_reward)}',flush=True)
            if self.save_model and episode % 10 == 0:
                torch.save(self.agent.qnetwork_local.state_dict(), f"./rainbow/{episode}" + ".pth")

        return np.mean(ma_scores)