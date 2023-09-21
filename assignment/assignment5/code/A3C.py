import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from matplotlib import pyplot as plt
import torch.nn.functional as F
import gym
from threading import Thread
import threading
from torch.utils.data import TensorDataset, DataLoader
from torch.distributions.normal import Normal
import copy

class Policy(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.mu_fc2 = nn.Linear(hidden_size, output_size)
        self.sigma_fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = 2 * torch.tanh(self.mu_fc2(x))
        sigma = F.softplus(self.sigma_fc2(x))
        return mu,sigma

class Value(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Value, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class Actor(Thread):
    def __init__(self, critic, lock, id, args):
        super(Actor, self).__init__()
        self.critic = critic
        self.global_policy = critic.global_policy
        self.policy = copy.deepcopy(critic.global_policy).to('cpu')
        self.policy.load_state_dict(critic.global_policy.state_dict())
        self.global_value = critic.global_value
        self.value = copy.deepcopy(critic.global_value).to('cpu')
        self.value.load_state_dict(critic.global_value.state_dict())
        self.env = gym.make(args.env_name, max_episode_steps=args.max_episode_length)
        self.lock = lock
        self.id = id

        self.episode = args.episode
        self.max_episode_length = args.max_episode_length
        self.update_interval = args.update_interval

    def run(self):
        episode = 0
        while self.critic.global_episode < self.episode:
            trajectory = []
            state = self.env.reset()[0].reshape(-1)
            done = False
            episode_reward = 0
            step = 0
            while not done and step<self.max_episode_length:
                mu, sigma = self.policy(torch.tensor(state).float())
                distribution = Normal(mu,sigma)
                action = distribution.sample().detach().numpy()
                action = np.clip(action, -2, 2)
                next_state, reward, done, _, _ = self.env.step(action)
                episode_reward += reward
                next_state = next_state.reshape(-1)
                trajectory.append((state, action, reward, next_state, int(done)))
                state = next_state
                step += 1

                if step % self.update_interval == 0:
                    with self.lock:
                        trajectory = self.cal_target_value(trajectory)
                        self.critic.learn(trajectory)
                        self.policy.load_state_dict(self.critic.global_policy.state_dict())
                        self.value.load_state_dict(self.critic.global_value.state_dict())
                        trajectory = []

            self.critic.global_rewards.append(episode_reward)

            with self.lock:
                print("Episode {} in actor {}: step {}, reward {}".format(self.critic.global_episode,self.id,step,episode_reward),flush=True)
                # save best model
                if self.critic.best_reward < episode_reward:
                    os.makedirs('A3C_models/',exist_ok=True)
                    torch.save(self.global_policy.state_dict(),'A3C_models/'+self.critic.output+'policy.pth')
                    torch.save(self.global_value.state_dict(),'A3C_models/'+self.critic.output+'value.pth')
                    print(f"Best Model Saved at episode {self.critic.global_episode}, episode reward {episode_reward}",flush=True)
                    self.critic.best_reward = episode_reward
                    
                self.critic.global_episode += 1
            episode += 1

    def cal_target_value(self, trajectory):
        R_list = []
        trajectory.reverse()
        for i,t in enumerate(trajectory):
            if i == 0:
                R_end = self.value(torch.tensor(trajectory[0][3]))
                R = R_end
            else:
                reward = t[2]
                R = reward + self.critic.gamma * R
            R_list.append(R)
        assert len(R_list) == len(trajectory)
        R_list.reverse()
        trajectory.reverse()
        state_, action_, reward_, next_state_, done_ = zip(*trajectory)
        trajectory = zip(state_, action_, reward_, next_state_, done_, R_list)
        return trajectory

class Critic():
    def __init__(self, args):
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

        self.global_value = Value(input_size=self.state_dim,hidden_size=128,output_size=1).to(self.device)
        self.global_policy = Policy(input_size=self.state_dim,hidden_size=128,output_size=self.action_dim).to(self.device)

        self.global_rewards = []
        self.gamma = args.gamma
        self.policy_lr = args.policy_lr
        self.value_lr = args.value_lr
        self.entropy_beta = args.entropy_beta
        self.global_episode = 0
        self.best_reward = -100000
        self.batch_size = args.batch_size
        self.max_episode_length = args.max_episode_length
        self.test_episode = args.test_episode
        self.policy_optim = optim.Adam(self.global_policy.parameters(), lr=self.policy_lr)
        self.value_optim = optim.Adam(self.global_value.parameters(), lr=self.value_lr)
        self.output = f'actors{args.num_actors}_c{args.update_interval}_gamma{args.gamma}_idx{args.idx}_'

    def learn(self, trajectory):
        state, action, reward, next_state, done, target_value = zip(*trajectory)

        state = torch.tensor(state).float().to(self.device)
        action = torch.tensor(action).float().to(self.device)
        target_value = torch.tensor(target_value).float().to(self.device)

        dataset = TensorDataset(state,action,target_value)
        dataloader = DataLoader(dataset,batch_size=self.batch_size,shuffle=False)

        for state_b,action_b,target_value_b in dataloader:
            values = self.global_value(state_b).view(-1)
            td_targets = target_value_b.view(-1)
            advantages = td_targets - values
            mu,sigma = self.global_policy(state_b)
            sigma = torch.clamp(sigma,0.01,0.5)
            distribution = Normal(mu,sigma)
            policy_log_probs = distribution.log_prob(action_b)
            entropy = distribution.entropy()
            value_loss = F.mse_loss(values, td_targets.detach())
            policy_loss = -(policy_log_probs * advantages.detach() + self.entropy_beta * entropy).sum()
            self.policy_optim.zero_grad()
            self.value_optim.zero_grad()
            value_loss.backward()
            policy_loss.backward()
            self.value_optim.step()
            self.policy_optim.step()

    def test(self):
        policy = copy.deepcopy(self.global_policy).to('cpu')
        policy.load_state_dict(torch.load('A3C_models/'+self.output+'policy.pth'))
        value = copy.deepcopy(self.global_value).to('cpu')
        value.load_state_dict(torch.load('A3C_models/'+self.output+'value.pth'))

        reward_list = []
        for i in range(self.test_episode):
            state = self.env.reset()[0].reshape(-1)
            done = False
            episode_reward = 0
            step = 0
            while not done and step<self.max_episode_length:
                mu, sigma = policy(torch.tensor(state).float())
                distribution = Normal(mu,sigma)
                action = distribution.sample().detach().numpy()
                action = np.clip(action, -2, 2)
                next_state, reward, done, _, _ = self.env.step(action)
                episode_reward += reward
                next_state = next_state.reshape(-1)
                state = next_state
                step += 1
            reward_list.append(episode_reward)

        print("Testing for {} episodes, the reward in each episode:".format(self.test_episode),flush=True)
        print(reward_list,flush=True)
        print("The average reward for {} episodes is {}".format(self.test_episode,np.mean(np.array(reward_list))),flush=True)

    def plot_loss(self):
        global_rewards = np.array(self.global_rewards)
        os.makedirs('A3C_log/',exist_ok=True)
        np.save('A3C_log/'+self.output+'reward.npy',global_rewards)

        plt.style.use('ggplot')
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
        os.makedirs("A3C_figure/",exist_ok=True)
        path = "./A3C_figure/" + self.output + '_reward.png'
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
        path = "./A3C_figure/" + self.output + '_avgreward.png'
        plt.savefig(path) 

class A3C_Agent():
    def __init__(self,args) -> None:
        self.env_name = args.env_name

        lock = threading.Lock()
        self.critic = Critic(args=args)
        self.actors = [Actor(self.critic,lock=lock,id=i+1,args=args) for i in range(args.num_actors)]


    def train(self):
        print("\n--------------------------------------------------Training----------------------------------------------------\n",flush=True)

        for actor in self.actors:
            actor.start()

        for actor in self.actors:
            actor.join()

    def test(self):
        print("\n\n--------------------------------------------------Testing----------------------------------------------------\n",flush=True)
        self.critic.test()   

    def plot_reward(self):
        self.critic.plot_loss()   
