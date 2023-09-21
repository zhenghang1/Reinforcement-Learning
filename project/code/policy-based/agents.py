from networks import PPOActor, PPOCritic, DDPGActor, DDPGCritic, SACActor, SACCritic
from utils import RunningMeanStd, ReplayBuffer, ReplayBuffer_prob, OUNoise

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import torch.nn.functional as F
import gym

class PPOAgent:
    def __init__(self,
                 state_dim,
                 action_dim,
                 action_bound,
                 device,
                 args):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.device = device
        self.discount_factor = args.discount_factor
        self.update_frequency = args.update_frequency
        self.batch_size = args.batch_size
        self.t_step = 0
        self.n_step = args.n_step
        self.model_type = args.model_type
        self.seed = args.seed

        random.seed(args.seed)
        torch.manual_seed(args.seed)

        # Create networks
        self.actor = PPOActor(state_dim, action_dim, hidden_dim=args.hidden_dim, action_bound=action_bound).to(self.device)
        self.critic = PPOCritic(state_dim, 1, hidden_dim=args.hidden_dim).to(self.device)
 
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=args.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=args.critic_lr)

        self.buffer = ReplayBuffer_prob(args.buffer_capacity, args.batch_size, self.device, args.seed, self.discount_factor, args.n_step)
        self.state_rms = RunningMeanStd(state_dim)

    def step(self, state, action, reward, next_state, done):
        # Save data in replay buffer
        next_state = np.array(next_state)
        next_state = self.state_rms(next_state)

        self.buffer.add(self.state.copy(), action, reward, next_state, done, self.log_prob.copy())

        self.t_step = (self.t_step + 1) % self.update_frequency
        if self.t_step == 0:
            if len(self.buffer) > self.batch_size:
                experiences = self.buffer.sample()
                loss = self.learn(experiences)
                return loss
            else:
                return None
        else:
            return None

    def act(self, state):
        state = np.array(state)
        state = self.state_rms(state)
        self.state = state
        state = torch.from_numpy(state).float().to(self.device)
        mu,sigma = self.actor(state)
        dist = torch.distributions.Normal(mu,sigma)
        action = dist.sample()
        self.log_prob = dist.log_prob(action).sum(-1,keepdim = True).detach().cpu().numpy()
        action = np.clip(action.detach().cpu().numpy(),-1*self.action_bound,self.action_bound)
        return action
    
    def get_gae(self, states, rewards, dones):
        values = self.critic(states).detach()
        returns = torch.zeros_like(rewards)
        advants = torch.zeros_like(rewards)

        running_returns = 0
        previous_value = 0
        running_advants = 0
        masks = 1-dones
        for t in reversed(range(0, len(rewards))):
            running_returns = rewards[t] + 0.99 * running_returns * masks[t]
            running_tderror = rewards[t] + 0.99 * previous_value * masks[t] - values[t]
            running_advants = running_tderror + 0.99 * 0.95 * running_advants * masks[t]

            returns[t] = running_returns
            previous_value = values[t]
            advants[t] = running_advants

        advants = (advants - advants.mean()) / (advants.std()+1e-3)
        return returns, advants

    
    def learn(self, experiences):
        states, actions, rewards, next_states, dones, old_log_probs = experiences

        returns, advantages = self.get_gae(states, rewards, dones)

        for i in range(1):
            curr_mu,curr_sigma = self.actor(states)
            value = self.critic(states).float()
            curr_dist = torch.distributions.Normal(curr_mu,curr_sigma)
            entropy = curr_dist.entropy() * 1e-2
            curr_log_prob = curr_dist.log_prob(actions).sum(-1,keepdim = True)

            #policy clipping
            ratio = torch.exp(curr_log_prob - old_log_probs.detach())
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1-0.2, 1+0.2) * advantages
            actor_loss = (-torch.min(surr1, surr2) - entropy).mean() 

            #value clipping (PPO2 technic)
            loss_fn = nn.MSELoss()
            critic_loss = loss_fn(value,returns.detach().float())

            max_grad_norm = 0.5

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), max_grad_norm)
            self.actor_optimizer.step()
            
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), max_grad_norm)
            self.critic_optimizer.step()

        return actor_loss.detach().cpu().numpy() + critic_loss.detach().cpu().numpy()


class DDPGAgent:
    def __init__(self,
                 state_dim,
                 action_dim,
                 action_bound,
                 device,
                 args):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.device = device
        self.discount_factor = args.discount_factor
        self.update_frequency = args.update_frequency
        self.batch_size = args.batch_size
        self.t_step = 0
        self.n_step = args.n_step
        self.model_type = args.model_type
        self.seed = args.seed
        self.env_name = args.env_name
        self.tau = args.tau
        self.noise = OUNoise(action_dim,0)


        random.seed(args.seed)
        torch.manual_seed(args.seed)

        # Create networks
        self.actor = DDPGActor(state_dim, action_dim, hidden_dim=args.hidden_dim, action_bound=self.action_bound).to(self.device)
        self.critic = DDPGCritic(state_dim+action_dim, 1, hidden_dim=args.hidden_dim).to(self.device)
        self.target_actor = DDPGActor(state_dim, action_dim, hidden_dim=args.hidden_dim, action_bound=self.action_bound).to(self.device)
        self.target_critic = DDPGCritic(state_dim+action_dim, 1, hidden_dim=args.hidden_dim).to(self.device)
        
        self.soft_update(self.actor, self.target_actor, 1.)
        self.soft_update(self.critic, self.target_critic, 1.)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=args.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=args.critic_lr)

        self.buffer = ReplayBuffer(args.buffer_capacity, args.batch_size, self.device, args.seed, self.discount_factor, args.n_step)

        self.pre_fill_buffer(args.pre_fill_buffer)

    def soft_update(self, network, target_network, rate):
        for network_params, target_network_params in zip(network.parameters(), target_network.parameters()):
            target_network_params.data.copy_(target_network_params.data * (1.0 - rate) + network_params.data * rate)

    def step(self, state, action, reward, next_state, done):
        # Save data in replay buffer
        next_state = np.array(next_state)

        self.buffer.add(state, action, reward, next_state, done)

        self.t_step = (self.t_step + 1) % self.update_frequency
        if self.t_step == 0:
            if len(self.buffer) > self.batch_size:
                experiences = self.buffer.sample()
                loss = self.learn(experiences)
                return loss
            else:
                return None
        else:
            return None

    def act(self, state):
        state = np.array(state)
        state = torch.from_numpy(state).float().to(self.device)
        action = self.actor(state).detach().cpu().numpy()
        action = np.clip(action + self.noise.sample(), -self.action_bound, self.action_bound)

        return action
    
    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        for i in range(1):
            next_actions = self.target_actor(next_states)
            targets = rewards + self.discount_factor * (1 - dones) * self.target_critic(next_states, next_actions)
            values = self.critic(states,actions)
            
            critic_loss = F.smooth_l1_loss(values, targets.detach())
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            actions_pred = self.actor(states)
            actor_loss = - self.critic(states, actions_pred).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.soft_update(self.critic, self.target_critic, self.tau)
            self.soft_update(self.actor, self.target_actor, self.tau)

        return actor_loss.detach().cpu().numpy() + critic_loss.detach().cpu().numpy()


    def pre_fill_buffer(self,size):
        env = gym.make(self.env_name)
        state = env.reset()[0]
        score = 0
        episode = 1
        print(f"Running random policy to fill Replay Buffer to size: {size}",flush=True)
        for i in range(size):
            action = np.random.uniform(low=-self.action_bound, high=self.action_bound, size=self.action_dim)
            next_state, reward, done, _ , _ = env.step(action)
            self.buffer.add(state, action, reward, next_state, done)
            score += reward
            state = next_state
            if done:
                print(f"Episode {episode} Score {score}",flush=True)
                episode += 1
                score = 0
                state = env.reset()[0]
        

class SACAgent:
    def __init__(self,
                 state_dim,
                 action_dim,
                 action_bound,
                 device,
                 args):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.device = device
        self.discount_factor = args.discount_factor
        self.update_frequency = args.update_frequency
        self.batch_size = args.batch_size
        self.t_step = 0
        self.seed = args.seed
        self.env_name = args.env_name
        self.tau = args.tau

        random.seed(args.seed)
        torch.manual_seed(args.seed)

        # Create networks
        self.actor = SACActor(state_dim, action_dim, hidden_dim=args.hidden_dim, action_bound=self.action_bound).to(self.device)
        self.critic1 = SACCritic(state_dim+action_dim, 1, hidden_dim=args.hidden_dim).to(self.device)
        self.critic2 = SACCritic(state_dim+action_dim, 1, hidden_dim=args.hidden_dim).to(self.device)
        self.target_critic1 = SACCritic(state_dim+action_dim, 1, hidden_dim=args.hidden_dim).to(self.device)
        self.target_critic2 = SACCritic(state_dim+action_dim, 1, hidden_dim=args.hidden_dim).to(self.device)

        self.soft_update(self.critic1, self.target_critic1, 1.)
        self.soft_update(self.critic2, self.target_critic2, 1.)

        self.buffer = ReplayBuffer(args.buffer_capacity, args.batch_size, self.device, args.seed, self.discount_factor, args.n_step)
        self.alpha = nn.Parameter(torch.tensor(args.alpha_init))
        self.target_entropy = - torch.tensor(action_dim)

        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=args.critic_lr)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=args.critic_lr)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=args.actor_lr)
        self.alpha_optimizer = optim.Adam([self.alpha], lr=args.alpha_lr)

        self.pre_fill_buffer(args.pre_fill_buffer)

    def soft_update(self, network, target_network, rate):
        for network_params, target_network_params in zip(network.parameters(), target_network.parameters()):
            target_network_params.data.copy_(target_network_params.data * (1.0 - rate) + network_params.data * rate)

    def step(self, state, action, reward, next_state, done):
        # Save data in replay buffer
        next_state = np.array(next_state)
        self.buffer.add(state, action, reward, next_state, done)

        self.t_step = (self.t_step + 1) % self.update_frequency
        if self.t_step == 0:
            if len(self.buffer) > self.batch_size:
                experiences = self.buffer.sample()
                loss = self.learn(experiences)
                return loss
            else:
                return None
        else:
            return None

    def act(self, state):
        state = np.array(state)
        state = torch.from_numpy(state).float().to(self.device)
        mu,sigma = self.actor(state)
        dist = torch.distributions.Normal(mu,sigma)
        u = dist.sample()
        action = self.action_bound * torch.tanh(u)
        action = action.detach().cpu().numpy()

        return action


    def act_and_prob(self, state):
        mu,sigma = self.actor(state)
        dist = torch.distributions.Normal(mu,sigma)
        u = dist.sample()
        log_prob = dist.log_prob(u)
        action = self.action_bound * torch.tanh(u)
        a_log_prob = log_prob - torch.log(1 - torch.square(action) +1e-3)
        action = action
        a_log_prob = a_log_prob.sum(-1, keepdim=True)

        return action, a_log_prob

    def critic_update(self, critic, critic_optimizer, states, actions, rewards, next_states, dones):
        with torch.no_grad():
            next_actions, next_action_log_prob = self.act_and_prob(next_states)
            q_1 = self.target_critic1(next_states, next_actions)
            q_2 = self.target_critic2(next_states, next_actions)
            q = torch.min(q_1,q_2)
            v = (1 - dones) * (q - self.alpha * next_action_log_prob)
            targets = rewards + self.discount_factor * v
        
        q = critic(states, actions)
        loss = F.smooth_l1_loss(q, targets.detach())
        critic_optimizer.zero_grad()
        loss.backward()
        critic_optimizer.step()
        return loss
    
    def actor_update(self, states):
        now_actions, now_action_log_prob = self.act_and_prob(states)
        q_1 = self.critic1(states, now_actions)
        q_2 = self.critic2(states, now_actions)
        q = torch.min(q_1, q_2)
        
        loss = (self.alpha.detach() * now_action_log_prob - q).mean()
        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()
        return loss,now_action_log_prob
    
    def alpha_update(self, now_action_log_prob):
        loss = (- self.alpha * (now_action_log_prob.detach() + self.target_entropy)).mean()
        self.alpha_optimizer.zero_grad()    
        loss.backward()
        self.alpha_optimizer.step()
        return loss

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        for i in range(1):
            ###q update
            critic1_loss = self.critic_update(self.critic1, self.critic1_optimizer, states, actions, rewards, next_states, dones)
            critic2_loss = self.critic_update(self.critic2, self.critic2_optimizer, states, actions, rewards, next_states, dones)

            ### actor update
            actor_loss,prob = self.actor_update(states)
            
            ###alpha update
            alpha_loss = self.alpha_update(prob)

            self.soft_update(self.critic1, self.target_critic1, self.tau)
            self.soft_update(self.critic2, self.target_critic2, self.tau)

            loss = critic1_loss + critic2_loss + actor_loss + alpha_loss

        return loss.detach().cpu().numpy()


    def pre_fill_buffer(self,size):
        env = gym.make(self.env_name)
        state = env.reset()[0]
        score = 0
        episode = 1
        print(f"Running random policy to fill Replay Buffer to size: {size}",flush=True)
        for i in range(size):
            action = np.random.uniform(low=-self.action_bound, high=self.action_bound, size=self.action_dim)
            next_state, reward, done, _ , _ = env.step(action)
            self.buffer.add(state, action, reward, next_state, done)
            score += reward
            state = next_state
            if done:
                print(f"Episode {episode} Score {score}",flush=True)
                episode += 1
                score = 0
                state = env.reset()[0]
        