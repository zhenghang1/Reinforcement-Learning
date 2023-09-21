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
from agents import PPOAgent, DDPGAgent, SACAgent

class Agent():
    def __init__(self,env_name,args,output) -> None:
        self.env = gym.make(env_name)
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.action_bound = self.env.action_space.high[0]
        self.model_type = args.model_type
        self.device = get_device(cuda=args.cuda)
        self.output = output
        self.save_model = args.save_model
        # 模型类型
        if args.model_type == "PPO":
            self.agent = PPOAgent(state_dim=self.state_dim,action_dim=self.action_dim,action_bound=self.action_bound,device=self.device,args=args)
        elif args.model_type == "DDPG":
            self.agent = DDPGAgent(state_dim=self.state_dim,action_dim=self.action_dim,action_bound=self.action_bound,device=self.device,args=args)
        elif args.model_type == "SAC":
            self.agent = SACAgent(state_dim=self.state_dim,action_dim=self.action_dim,action_bound=self.action_bound,device=self.device,args=args)
        else:
            raise ValueError("Invalid policy-based algorithm! Should be in [PPO, DDPG, SAC]")

    def train(self, episodes, max_steps):
        ma_scores = deque(maxlen=100)  # store the last 100 scores (used to find 100 MA)
        print("Running Agent",flush=True)
        total_step = 1
        for episode in range(episodes):
            state = self.env.reset()[0]
            done = False
            episode_reward = 0
            loss_list = []
            step = 0
            for i in range(max_steps):
                action = self.agent.act(state)
                next_state, reward, done, _, _ = self.env.step(action)
                loss = self.agent.step(state, action, reward, next_state, done)
                if loss:
                    loss_list.append(loss)
                state = next_state
                episode_reward += reward
                step += 1
                total_step += 1
                if done:
                    break

            ma_scores.append(episode_reward)  # save most recent score
            print(f'Episode:{episode} Step:{step} Average Reward: {np.mean(ma_scores)} Reward: {int(episode_reward)}',flush=True)
            if self.save_model and episode % 10 == 0:
                torch.save(self.agent.qnetwork_local.state_dict(), f"./rainbow/{episode}" + ".pth")

        return np.mean(ma_scores)
    
