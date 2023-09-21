import os
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from matplotlib import pyplot as plt

# GPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f'There are {torch.cuda.device_count()} GPU(s) available.',flush=True)
    print('Device name:', torch.cuda.get_device_name(0),flush=True)

else:
    print('No GPU available, using the CPU instead.',flush=True)
    device = torch.device("cpu")

class Q_Network1(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Q_Network1, self).__init__()
        self.fc1 = nn.Linear(state_dim, 16)
        self.fc2 = nn.Linear(16, action_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class Q_Network2(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Q_Network2, self).__init__()
        self.fc1 = nn.Linear(state_dim, 8)
        self.fc2 = nn.Linear(8, 16)
        self.fc3 = nn.Linear(16, action_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class DuelingDQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DuelingDQN, self).__init__()

        self.feature = nn.Sequential(
            nn.Linear(state_dim, 16),
            nn.ReLU()
        )

        self.advantage = nn.Sequential(
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, action_dim)
        )

        self.value = nn.Sequential(
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

    def forward(self, x):
        feature = self.feature(x)
        advantage = self.advantage(feature)
        value = self.value(feature)
        q = value + advantage - advantage.mean(0, keepdim=True)
        return q

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
    
class Agent():
    def __init__(self,env,model_type,network_type,buffer_capacity,epsilon,fix_step,reward_type) -> None:
        self.env = env
        # Q网络类型
        if network_type == 1:
            Network = Q_Network1
        elif network_type == 2:
            Network = Q_Network2
        elif network_type == 3:
            Network = DuelingDQN
        
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.model_type = model_type

        # 模型类型
        if model_type == "DQN":
            self.model = Network(self.state_dim,self.action_dim).to(device)
            self.fix_model = Network(self.state_dim,self.action_dim).to(device)
            self.fix_model.load_state_dict(self.model.state_dict())
        elif model_type == "Double_DQN":
            self.model = Network(self.state_dim,self.action_dim).to(device)
            self.target_model = Network(self.state_dim,self.action_dim).to(device)
            self.target_model.load_state_dict(self.model.state_dict())
        elif model_type == "Dueling_DQN":
            self.model = Network(self.state_dim,self.action_dim).to(device)
            self.target_model = Network(self.state_dim,self.action_dim).to(device)
            self.target_model.load_state_dict(self.model.state_dict())

        self.buffer = ReplayBuffer(buffer_capacity)
        self.epsilon = epsilon
        self.fix_step = fix_step
        self.reward_type = reward_type
        self.network_type = network_type
        self.total_step = 0
        self.loss_plot_list = []
        self.loss_per_episode = []
        self.step_per_episode = []
        self.time_per_episode = []

    def train(self,episodes,batch_size,lr,alpha):
        print("--------------------------------------------------Training----------------------------------------------------")
        s_time = time.time()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        best_step = 100000
        f = open('models/tmp.pth','+w')
        f.close
        last_model_path = 'models/tmp.pth'
        self.alpha = alpha
        self.output = self.model_type + '_alpha'+ str(self.alpha) + '_network'+ str(self.network_type) + '_reward' + str(self.reward_type)
        for i in range(episodes):
            start_time = time.time()
            state = self.env.reset()[0]
            done = False
            episode_reward = 0
            loss_list = []
            step = 0
            while not done:
                with torch.no_grad():
                    if self.model_type == "DQN":
                        q_values = self.fix_model(torch.tensor(state, dtype=torch.float32).to(device)).to('cpu')
                    elif self.model_type == "Double_DQN":
                        # 使用预测网络进行q值的计算，用以选取动作
                        q_values = self.model(torch.tensor(state, dtype=torch.float32).to(device)).to('cpu')
                    elif self.model_type == "Dueling_DQN":
                        # 使用预测网络进行q值的计算，用以选取动作
                        q_values = self.target_model(torch.tensor(state, dtype=torch.float32).to(device)).to('cpu')
                best_action = q_values.argmax().item()
                action = self.epsilon_greedy(best_action)
                next_state, reward, done, _, _ = self.env.step(action)
                step += 1
                self.total_step += 1
                # 返回对应的reward值
                reward = self.getReward(state,reward)
                self.buffer.push(state, action, reward, next_state, done)
                state = next_state
                episode_reward += reward
                loss = self.train_model(batch_size,loss_fn,alpha)
                if loss:
                    loss_list.append(loss)
                    self.loss_plot_list.append(loss)

                if step % self.fix_step == 0:
                    if self.model_type == 'DQN':
                        self.fix_model.load_state_dict(self.model.state_dict())
                    elif self.model_type == 'Double_DQN':
                        self.target_model.load_state_dict(self.model.state_dict())
                    elif self.model_type == 'Dueling_DQN':
                        self.target_model.load_state_dict(self.model.state_dict())

            # self.target_model.load_state_dict(self.model.state_dict())

            if step < best_step:
                path = 'models/{}_{}.pth'.format(self.output,step)
                if self.model_type == 'DQN':
                    torch.save(self.fix_model.state_dict(),path)
                elif self.model_type == 'Double_DQN':
                    torch.save(self.model.state_dict(),path)
                elif self.model_type == 'Dueling_DQN':
                    torch.save(self.model.state_dict(),path)
                os.system('rm {}'.format(last_model_path))
                last_model_path = path
                self.best_model_path = path
                best_step = step
            episode_loss = np.sum(np.array(loss_list))
            print("Episode {}: step {}, reward {}, loss {}".format(i,step,episode_reward,episode_loss),flush=True)
            self.time_per_episode.append(time.time()-start_time)
            self.step_per_episode.append(step)
            self.loss_per_episode.append(episode_loss)
        
            if (i+1) % 100 == 0:
                # save log file
                np.save('log/'+self.output+'_loss_per_episode.npy',np.array(self.loss_per_episode))
                np.save('log/'+self.output+'_step_per_episode.npy',np.array(self.step_per_episode))
                np.save('log/'+self.output+'_time_per_episode.npy',np.array(self.time_per_episode))
        
        # save log file
        np.save('log/'+self.output+'_loss_per_episode.npy',np.array(self.loss_per_episode))
        np.save('log/'+self.output+'_step_per_episode.npy',np.array(self.step_per_episode))
        np.save('log/'+self.output+'_time_per_episode.npy',np.array(self.time_per_episode))
        print("\nModel training done, takes totally {} s, best model takes {} steps".format(time.time()-s_time, best_step),flush=True)

    def train_model(self,batch_size,loss_fn,alpha):
        if len(self.buffer) < batch_size:
            return
        
        state, action, reward, next_state, done = self.buffer.sample(batch_size)
        state = torch.tensor(state, dtype=torch.float32).to(device)
        action = torch.tensor(action, dtype=torch.long).to(device)
        reward = torch.tensor(reward, dtype=torch.float32).to(device)
        next_state = torch.tensor(next_state, dtype=torch.float32).to(device)
        done = torch.tensor(done, dtype=torch.float32).to(device)

        if self.model_type == 'DQN':
            q_values = self.model(state)
            q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
            next_q_values = self.model(next_state)
            next_q_value = next_q_values.max(1)[0]
            expected_q_value = reward + alpha * next_q_value * (1 - done)
        elif self.model_type == 'Double_DQN':
            # 使用目标网络计算下一个状态的q值
            q_values = self.model(state)
            q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
            next_q_values = self.target_model(next_state)
            next_actions = torch.argmax(self.model(next_state),dim=1).unsqueeze(1)
            next_q_value = next_q_values.gather(1, next_actions).view(-1)
            expected_q_value = reward + alpha * next_q_value * (1 - done)
        elif self.model_type == 'Dueling_DQN':
            q_values = self.model(state)
            q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
            next_q_values = self.model(next_state)
            next_q_value = next_q_values.max(1)[0]
            expected_q_value = reward + alpha * next_q_value * (1 - done)

        loss = loss_fn(q_value, expected_q_value.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        l = loss.cpu().item()
        return l
    
    def test(self, test_episode):
        start_time = time.time()
        print("\n--------------------------------------------------Testing----------------------------------------------------",flush=True)
        if self.network_type == 1:
            model = Q_Network1(self.state_dim,self.action_dim)
        elif self.network_type == 2:
            model = Q_Network2(self.state_dim,self.action_dim)
        elif self.network_type == 3:
            model = DuelingDQN(self.state_dim,self.action_dim)
        
        model.load_state_dict(torch.load(self.best_model_path))
        print("Model load successfully",flush=True)

        step_list = []
        for i in range(test_episode):
            while True:
                state = self.env.reset()[0]
                done = False
                step = 0

                while not done:
                    q_values = model(torch.tensor(state, dtype=torch.float32))
                    action = q_values.argmax().item()
                    next_state, reward, done, _, _ = self.env.step(action)
                    step += 1
                    state = next_state
                    if step > 2000:
                        break
                if step < 2000:
                    break
            step_list.append(step)
            print("Episode {} done, takes {} steps".format(i+1,step),flush=True)
        
        print("Model testing done, takes total {} s".format(time.time()-start_time),flush=True)
        print("Testing for {} episodes, the steps taken in each episode:".format(test_episode),flush=True)
        print(step_list,flush=True)
        print("The average step for {} episodes is {} step".format(test_episode,np.mean(np.array(step_list))),flush=True)

    def test_model(self, test_episode, model_path):
        start_time = time.time()
        print("\n--------------------------------------------------Testing----------------------------------------------------",flush=True)
        if self.network_type == 1:
            model = Q_Network1(self.state_dim,self.action_dim)
        elif self.network_type == 2:
            model = Q_Network2(self.state_dim,self.action_dim)
        elif self.network_type == 3:
            model = DuelingDQN(self.state_dim,self.action_dim)
                    
        model.load_state_dict(torch.load(model_path))
        print("Model load successfully",flush=True)

        step_list = []
        for i in range(test_episode):
            while True:
                state = self.env.reset()[0]
                done = False
                step = 0

                while not done:
                    q_values = model(torch.tensor(state, dtype=torch.float32))
                    action = q_values.argmax().item()
                    next_state, reward, done, _, _ = self.env.step(action)
                    step += 1
                    state = next_state
                    if step > 2000:
                        print("Episode {} repeat".format(i+1),flush=True)
                        break
                if step < 2000:
                    break
            step_list.append(step)
            print("Episode {} done, takes {} steps".format(i+1,step),flush=True)
        
        print("Model testing done, takes total {} s".format(time.time()-start_time),flush=True)
        print("Testing for {} episodes, the steps taken in each episode:".format(test_episode),flush=True)
        print(step_list,flush=True)
        print("The average step for {} episodes is {} step".format(test_episode,np.mean(np.array(step_list))),flush=True)


    def getReward(self,state,reward):
        if self.reward_type == 0:
            return reward
        elif self.reward_type == 1:
            x = state[0]
            if x <= -0.5:
                return 5*pow(x+0.5,2)
            else:
                return 10*(1-pow(x-0.5,2))
            
        elif self.reward_type == 2:
            x,v = state[0],state[1]
            if x <= -0.5:
                return 5*pow(x+0.5,2) + 100*abs(v)
            elif x >= 0.5:
                return 100 - 100*v
            else:
                return (100-pow(10*(x-0.5),2)) + pow(100*v,2)           

        
    def epsilon_greedy(self,action):
        e = 0.9**(int(self.total_step/3))
        if e < self.epsilon:
            e = self.epsilon
        
        m = self.action_dim
        p = random.randint(0,1000000) * 1e-6
        assert p>=0 and p<=1
        if p < e:
            while True:
                ran = random.randint(0,m-1)
                if ran != action:
                    return ran
        else:
            return action

    def plot_loss(self):
        plt.plot(np.arange(len(self.loss_plot_list)), self.loss_plot_list)
        plt.ylabel("Loss")
        plt.xlabel("Training steps")
        path = "./figure/" + self.output + '_loss.png'
        plt.savefig(path)