from networks import DuelingNetwork, DQN, Rainbow
from utils import ReplayBuffer, PrioritizedReplayBuffer

import numpy as np
import torch
import torch.nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
import random
import copy


class DQNAgent:
    def __init__(self,
                 input_dim,
                 output_dim,
                 device,
                 args):

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.eta = 0.1
        self.device = device
        self.discount_factor = args.discount_factor
        self.update_frequency = args.update_frequency
        self.batch_size = args.batch_size
        self.Q_updates = 0
        self.t_step = 0
        self.n_step = args.n_step
        self.model_type = args.model_type
        self.seed = args.seed

        random.seed(args.seed)
        torch.manual_seed(args.seed)

        self.per = args.per
        self.noisy = args.noisy

        # Create networks
        if args.model_type == "DQN" or args.model_type == "DDQN":
            self.qnetwork_local = DQN(input_dim=input_dim, output_dim=output_dim, conv=args.conv, seed=self.seed, noisy=self.noisy).to(self.device)
            self.qnetwork_target = DQN(input_dim=input_dim, output_dim=output_dim, conv=args.conv, seed=self.seed, noisy=self.noisy).to(self.device)
            self.tau = 1
            self.soft_update(self.qnetwork_local,self.qnetwork_target)
            self.tau = args.tau
        if args.model_type == "Dueling":
            self.qnetwork_local = DuelingNetwork(input_dim=input_dim, output_dim=output_dim, conv=args.conv, seed=self.seed,
                                                 noisy=self.noisy).to(self.device)
            self.qnetwork_target = DuelingNetwork(input_dim=input_dim, output_dim=output_dim, conv=args.conv, seed=self.seed,
                                                 noisy=self.noisy).to(self.device)
            self.tau = 1
            self.soft_update(self.qnetwork_local,self.qnetwork_target)
            self.tau = args.tau

        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=args.lr)

        # Create appropriate replay buffer
        if self.per:
            self.buffer = PrioritizedReplayBuffer(args.buffer_capacity, args.batch_size, seed=args.seed,
                                                  discount_factor=self.discount_factor, n_step=args.n_step)
        else:
            self.buffer = ReplayBuffer(args.buffer_capacity, args.batch_size, self.device, args.seed, self.discount_factor, args.n_step)

    def step(self, state, action, reward, next_state, done):
        # Save data in replay buffer
        self.buffer.add(state, action, reward, next_state, done)

        self.t_step = (self.t_step + 1) % self.update_frequency
        if self.t_step == 0:
            if len(self.buffer) > self.batch_size:
                experiences = self.buffer.sample()
                if not self.per:
                    loss = self.learn(experiences)
                else:
                    loss = self.learn_per(experiences)
                self.Q_updates += 1
            return loss
        else:
            return None

    def act(self, state, eps=0.0):
        # Use epsilon greedy to pick action 
        if random.random() > eps:
            state = np.array(state)
            state = torch.from_numpy(state).float().to(self.device)
            self.qnetwork_local.eval()
            with torch.no_grad():
                action_values = self.qnetwork_local(state)
            self.qnetwork_local.train()
            action = np.argmax(action_values.cpu().data.numpy(), axis=1)
            return action[0]
        else:
            action = random.choices(np.arange(self.output_dim), k=1)
            return action[0]

    def learn(self, experiences):
        self.optimizer.zero_grad()
        states, actions, rewards, next_states, dones = experiences

        if self.model_type == "DQN":
            Q_targets_next = self.qnetwork_local(next_states).detach().max(1)[0].unsqueeze(1)
        else:
            Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        
        Q_targets = rewards + (self.discount_factor ** self.n_step * Q_targets_next * (1 - dones))
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        loss = F.mse_loss(Q_expected, Q_targets)
        loss.backward()
        clip_grad_norm_(self.qnetwork_local.parameters(), 1)
        self.optimizer.step()

        if self.model_type == "DDQN":
            self.soft_update(self.qnetwork_local, self.qnetwork_target)
        return loss.detach().cpu().numpy()

    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)

    def learn_per(self, experiences):
        self.optimizer.zero_grad()
        states, actions, rewards, next_states, dones, idx, weights = experiences
        states = torch.FloatTensor(states).to(self.device)
        next_states = torch.FloatTensor(np.float32(next_states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).to(self.device).unsqueeze(1)
        dones = torch.FloatTensor(dones).to(self.device).unsqueeze(1)
        weights = torch.FloatTensor(weights).unsqueeze(1).to(self.device)
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (self.discount_factor ** self.n_step * Q_targets_next * (1 - dones))
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        td_error = Q_targets - Q_expected
        loss = (td_error.pow(2) * weights).mean().to(self.device)
        loss.backward()
        clip_grad_norm_(self.qnetwork_local.parameters(), 1)
        self.optimizer.step()

        # update target network
        if self.model_type == "DDQN":
            self.soft_update(self.qnetwork_local, self.qnetwork_target)

        self.buffer.update_priorities(idx, abs(td_error.data.cpu().numpy()))

        return loss.detach().cpu().numpy()


class RainbowAgent:
    def __init__(self,
                 input_dim,
                 output_dim,
                 device,
                 args):

        random.seed(args.seed)
        torch.manual_seed(args.seed)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device
        self.discount_factor = args.discount_factor
        self.update_frequency = 300
        self.batch_size = args.batch_size
        self.Q_updates = 0
        self.t_step = 0
        self.n_step = args.n_step
        self.conv = args.conv
        self.atom_size = args.atom_size
        self.Vmax = args.vmax
        self.Vmin = args.vmin

        self.per = args.per
        self.noisy = args.noisy

        # Create Rainbow network
        self.qnetwork_local = Rainbow(input_dim=input_dim, output_dim=output_dim, conv=args.conv, seed=args.seed,
                                      atom_size=self.atom_size, Vmax=self.Vmax, Vmin=self.Vmin).to(device)
        self.qnetwork_target = Rainbow(input_dim=input_dim, output_dim=output_dim, conv=args.conv, seed=args.seed,
                                       atom_size=self.atom_size, Vmax=self.Vmax, Vmin=self.Vmin).to(device)
        self.tau = 1
        self.soft_update(self.qnetwork_local,self.qnetwork_target)
        self.tau = args.tau

        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=args.lr)
        # Create replay buffer
        if self.per:
            self.buffer = PrioritizedReplayBuffer(args.buffer_capacity, args.batch_size, seed=args.seed,
                                                  discount_factor=self.discount_factor, n_step=args.n_step)
        else:
            self.buffer = ReplayBuffer(args.buffer_capacity, args.batch_size, self.device, args.seed, self.discount_factor, args.n_step)

    def step(self, state, action, reward, next_state, done):
        self.buffer.add(state, action, reward, next_state, done)

        self.t_step = (self.t_step + 1) % self.update_frequency
        if self.t_step == 0:
            if len(self.buffer) > self.batch_size:
                experiences = self.buffer.sample()
                loss = self.learn_per(experiences)
                self.Q_updates += 1
            return loss
        else:
            return None

    def act(self, state, eps=0.):
        if random.random() > eps:  # select greedy action if random number is higher than epsilon or noisy network is used!
            state = np.array(state)
            if len(self.input_dim) > 1:
                state = torch.from_numpy(state).float().to(self.device)
            else:
                state = torch.from_numpy(state).float().to(self.device)
            self.qnetwork_local.eval()
            with torch.no_grad():
                action_values = self.qnetwork_local.act(state)
            self.qnetwork_local.train()
            action = np.argmax(action_values.cpu().data.numpy(), axis=1)

            return action[0]

        else:
            action = random.choices(np.arange(self.output_dim), k=1)
            return action[0]

    def projection_distribution(self, next_distribution, next_state, rewards, dones):
        batch_size = next_state.size(0)

        delta_z = float(self.Vmax - self.Vmin) / (self.atom_size - 1)
        support = torch.linspace(self.Vmin, self.Vmax, self.atom_size)
        support = support.unsqueeze(0).expand_as(next_distribution).to(self.device)

        rewards = rewards.expand_as(next_distribution)
        dones = dones.expand_as(next_distribution)

        t_z = rewards + (1 - dones) * self.discount_factor ** self.n_step * support
        t_z = t_z.clamp(min=self.Vmin, max=self.Vmax)
        b = ((t_z - self.Vmin) / delta_z).cpu()
        l = b.floor().long().cpu()
        u = b.ceil().long().cpu()

        offset = torch.linspace(0, (batch_size - 1) * self.atom_size, batch_size).long() \
            .unsqueeze(1).expand(batch_size, self.atom_size)

        proj_dist = torch.zeros(next_distribution.size())
        proj_dist.view(-1).index_add_(0, (l + offset).view(-1), (next_distribution.cpu() * (u.float() - b)).view(-1))
        proj_dist.view(-1).index_add_(0, (u + offset).view(-1), (next_distribution.cpu() * (b - l.float())).view(-1))

        return proj_dist

    def learn_per(self, experiences):
        states, actions, rewards, next_states, dones, idx, weights = experiences

        states = torch.FloatTensor(states).to(self.device)
        next_states = torch.FloatTensor(np.float32(next_states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).to(self.device).unsqueeze(1)
        dones = torch.FloatTensor(dones).to(self.device).unsqueeze(1)
        weights = torch.FloatTensor(weights).unsqueeze(1).to(self.device)

        batch_size = self.batch_size
        self.optimizer.zero_grad()

        next_distr = self.qnetwork_target(next_states)
        next_actions = self.qnetwork_target.act(next_states)

        next_actions = next_actions.max(1)[1].data.cpu().numpy()

        next_best_distr = next_distr[range(batch_size), next_actions]

        proj_distr = self.projection_distribution(next_best_distr, next_states, rewards, dones).to(self.device)
        # Calculate loss
        prob_distr = self.qnetwork_local(states)

        actions = actions.unsqueeze(1).expand(batch_size, 1, self.atom_size)
        state_action_prob = prob_distr.gather(1, actions).squeeze(1)
        if not self.conv:
            state_action_prob += 1e-34

        loss_prio = -((state_action_prob.log() * proj_distr.detach()).sum(dim=1).unsqueeze(1) * weights)
        loss = loss_prio.mean()

        loss.backward()
        clip_grad_norm_(self.qnetwork_local.parameters(), 1)
        self.optimizer.step()

        self.soft_update(self.qnetwork_local, self.qnetwork_target)
        self.buffer.update_priorities(idx, abs(loss_prio.data.cpu().numpy()))
        return loss.detach().cpu().numpy()

    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau * local_param.data + (1.0 - self.tau) * target_param.data)
