import os
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from matplotlib import pyplot as plt
import torch.nn.functional as F
import gym
import copy

class Policy(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action = 2 * torch.tanh(self.fc3(x))
        return action
    
    def get_action(self, state):
        return self.forward(state).squeeze(0).detach().cpu().numpy()

class Value(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size):
        super(Value, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, state, action):
        x = F.relu(self.fc1(torch.cat((state,action),dim=1)))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def predict(self, state, action):
        state, action = torch.tensor(state), torch.tensor(action)
        return self.forward(state, action).detach().numpy()

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

class DDPG_Agent():
    def __init__(self, args) -> None:
        self.env = gym.make(args.env_name, max_episode_steps=args.max_episode_length)
        # GPU
        if args.cuda == -1:
            self.device = torch.device("cpu")
            print("Using CPU")
        else:
            if torch.cuda.is_available():
                self.device = torch.device(f"cuda:{args.cuda}")
                print(f'There are {torch.cuda.device_count()} GPU(s) available.',flush=True)
                print('Device name:', torch.cuda.get_device_name(0),flush=True)
            else:
                print('No GPU available, using the CPU instead.',flush=True)
                self.device = torch.device("cpu")
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]

        self.buffer = ReplayBuffer(capacity=args.buffer_capacity)
        self.actor = Policy(input_size=self.state_dim,hidden_size=128,output_size=self.action_dim).to(self.device)
        self.critic = Value(state_dim=self.state_dim,action_dim=self.action_dim,hidden_size=128).to(self.device)

        self.target_actor = copy.deepcopy(self.actor).to(self.device)
        self.target_critic = copy.deepcopy(self.critic).to(self.device)

        self.global_rewards = []
        self.global_loss = []
        self.episode = args.episode
        self.gamma = args.gamma
        self.actor_lr = args.actor_lr
        self.critic_lr = args.critic_lr
        self.batch_size = args.batch_size
        self.tau = args.tau
        self.noise_std = args.noise_std
        self.max_episode_length = args.max_episode_length
        self.test_episode = args.test_episode
        self.update_interval = args.update_interval
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.critic_lr)
        self.output = f'tau_{args.tau}_interval{args.update_interval}_gamma{args.gamma}_l{args.max_episode_length}_idx{args.idx}_'        

    def update_target_network(self):
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            with torch.no_grad():
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            with torch.no_grad():
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def get_action(self, state, noise_std):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = self.actor.get_action(state)
        action = np.clip(np.random.normal(action, noise_std, self.action_dim), -2, 2)
        return action

    def train(self):
        print("--------------------------------------------------Training----------------------------------------------------")
        best_reward = -100000
        for i in range(self.episode):
            state = self.env.reset()[0]
            done = False
            episode_reward = 0
            loss_list = []
            step = 0
            
            while not done and step<self.max_episode_length:
                self.noise_std *= 0.995
                action = self.get_action(state=state,noise_std=self.noise_std)
                next_state, reward, done, _, _ = self.env.step(action)
                step += 1
                self.buffer.push(state, action, reward, next_state, done)
                state = next_state
                episode_reward += reward
                if step % self.update_interval == 0:
                    loss = self.train_model()
                    if loss:
                        loss_list.append(loss)

                    self.update_target_network()

            self.global_rewards.append(episode_reward)
            self.global_loss.append(np.mean(loss))
            print("Episode {}: step {}, reward {}".format(i,step,episode_reward),flush=True)

            # save best model
            if best_reward < episode_reward:
                os.makedirs('DDPG_models',exist_ok=True)
                torch.save(self.actor.state_dict(),'DDPG_models/'+self.output+'actor.pth')
                torch.save(self.critic.state_dict(),'DDPG_models/'+self.output+'critic.pth')
                print(f"Best Model Saved at episode {i}, episode reward {episode_reward}",flush=True)
                best_reward = episode_reward

    def train_model(self):
        if len(self.buffer) < self.batch_size:
            return
        
        state, action, reward, next_state, done = self.buffer.sample(self.batch_size)
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        action = torch.tensor(action, dtype=torch.long).to(self.device)
        reward = torch.tensor(reward, dtype=torch.float32).to(self.device)
        next_state = torch.tensor(next_state, dtype=torch.float32).to(self.device)
        done = torch.tensor(done, dtype=torch.float32).to(self.device)

        next_actions = self.target_actor(next_state)
        target_values = reward + self.gamma * self.target_critic(next_state, next_actions).view(-1) * (1-done)
        values = self.critic(state,action).view(-1)
        value_loss = F.mse_loss(values, target_values.detach())
        self.critic_optim.zero_grad()
        
        value_loss.backward()
        self.critic_optim.step()
        
        action_pred = self.actor(state)
        policy_loss = -torch.sum(self.critic(state, action_pred))
        self.actor_optim.zero_grad()
        policy_loss.backward()
        self.actor_optim.step()

        loss = value_loss.item() + policy_loss.item()

        return loss

    def test(self):
        print("\n\n--------------------------------------------------Testing----------------------------------------------------\n",flush=True)
        self.actor.load_state_dict(torch.load('DDPG_models/'+self.output+'actor.pth'))

        reward_list = []
        for i in range(self.test_episode):
            state = self.env.reset()[0].reshape(-1)
            done = False
            episode_reward = 0
            step = 0
            while not done and step<self.max_episode_length:
                action = self.get_action(state=state,noise_std=0)
                next_state, reward, done, _, _ = self.env.step(action)
                step += 1
                state = next_state
                episode_reward += reward
            reward_list.append(episode_reward)

        print("Testing for {} episodes, the reward in each episode:".format(self.test_episode),flush=True)
        print(reward_list,flush=True)
        print("The average reward for {} episodes is {}".format(self.test_episode,np.mean(np.array(reward_list))),flush=True)

    def plot_reward(self):
        global_rewards = np.array(self.global_rewards)
        os.makedirs('DDPG_log',exist_ok=True)
        np.save('DDPG_log/'+self.output+'reward.npy',global_rewards)

        plt.style.use('ggplot')
        global_rewards = np.array(self.global_rewards)
        max_value = np.amax(global_rewards)
        min_value = np.amin(global_rewards)
        y_interval = (max_value-min_value) / 100
        max_idx = np.argmax(global_rewards)
        x_interval = len(self.global_rewards) / 100
        plt.plot(np.arange(len(self.global_rewards)), global_rewards)
        plt.ylabel("Reward")
        plt.xlabel("Training episodes")
        plt.title("Reward in each episode")
        plt.annotate(f"Max: {max_value:.2f}", xy=(max_idx, max_value), xytext=(max_idx+x_interval, max_value+y_interval),
             arrowprops=dict(facecolor='black', arrowstyle='->'))
        path = "./DDPG_figure/" + self.output + '_reward.png'
        plt.savefig(path) 

        plt.cla()
        plt.style.use('ggplot')
        avg_global_rewards = global_rewards / self.max_episode_length
        max_value = np.amax(avg_global_rewards)
        min_value = np.amin(avg_global_rewards)
        y_interval = (max_value-min_value) / 100
        max_idx = np.argmax(avg_global_rewards)
        x_interval = len(self.global_rewards) / 100
        plt.plot(np.arange(len(self.global_rewards)), avg_global_rewards)
        plt.ylabel("Avg Reward")
        plt.xlabel("Training episodes")
        plt.title("Average reward in each episode")
        plt.annotate(f"Max: {max_value:.2f}", xy=(max_idx, max_value), xytext=(max_idx+x_interval, max_value+y_interval),
             arrowprops=dict(facecolor='black', arrowstyle='->'))
        path = "./DDPG_figure/" + self.output + '_avgreward.png'
        plt.savefig(path) 

