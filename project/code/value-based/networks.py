import torch
import torch.nn as nn
import torch.nn.functional as F
import math

layer_size = 512

def set_weights(layers):
    for layer in layers:
        torch.nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')


class NoisyLinear(nn.Linear):
    """Noisy Layer to replace Epsilon-Greedy Exploration"""

    def __init__(self, in_features, out_features, bias=True, sigma_init=0.017):
        super(NoisyLinear, self).__init__(in_features, out_features, bias=bias)
        self.sigma_weight = nn.Parameter(torch.full((out_features, in_features), sigma_init))
        self.register_buffer("epsilon_weight", torch.zeros(out_features, in_features))
        if bias:
            self.sigma_bias = nn.Parameter(torch.full((out_features,), sigma_init))
            self.register_buffer("epsilon_bias", torch.zeros(out_features))

        self.reset_parameter()

    def reset_parameter(self):
        std = math.sqrt(3 / self.in_features)
        self.weight.data.uniform_(-std, std)
        self.bias.data.uniform_(-std, std)

    def forward(self, state):
        self.epsilon_weight.normal_()
        bias = self.bias
        if bias is not None:
            self.epsilon_bias.normal_()
            bias = bias + self.sigma_bias * self.epsilon_bias
        return F.linear(state, self.weight + self.sigma_weight * self.epsilon_weight, bias)


class ConvLayer(nn.Module):
    """ConvLayer to read process image (state)"""

    def __init__(self):
        super(ConvLayer, self).__init__()
        self.cnn_1 = nn.Conv2d(3, out_channels=32, kernel_size=8, stride=4)
        self.bn_1 = nn.BatchNorm2d(32)
        self.cnn_2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.bn_2 = nn.BatchNorm2d(64)
        self.cnn_3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.bn_3 = nn.BatchNorm2d(64)
        set_weights([self.cnn_1, self.cnn_2, self.cnn_3])

    def forward(self, state):
        if len(state.shape)==3:
            state = state.permute(2,0,1).unsqueeze(0)
        else:
            state = state.permute(0,3,1,2)
        x = torch.relu(self.bn_1(self.cnn_1(state)))
        x = torch.relu(self.bn_2(self.cnn_2(x)))
        x = torch.relu(self.bn_3(self.cnn_3(x)))
        return x

class DQN(nn.Module):
    """DQN/DDQN Network Setup"""

    def __init__(self, input_dim, output_dim, conv, seed, noisy):
        super(DQN, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.conv = conv

        if conv:
            self.conv_layer = ConvLayer()
            linear_input_dim = self.input_layer_size()
        else:
            linear_input_dim = input_dim[0]

        if noisy:
            Forward_layer = NoisyLinear
        else:
            Forward_layer = nn.Linear

        
        self.feed_forward_1 = Forward_layer(linear_input_dim, layer_size)
        self.feed_forward_2 = Forward_layer(layer_size, output_dim)  
        set_weights([self.feed_forward_1])

    def input_layer_size(self):
        x = torch.zeros(self.input_dim)
        x = self.conv_layer.forward(x)
        return x.flatten().shape[0]

    def forward(self, state):
        if self.conv:
            state = self.conv_layer.forward(state)

        if len(state.shape)==1:
            state = state.unsqueeze(0)

        state = state.reshape(state.size(0), -1)

        state = torch.relu(self.feed_forward_1(state))
        out = self.feed_forward_2(state)

        return out


class DuelingNetwork(nn.Module):
    """Dueling Network Setup"""

    def __init__(self, input_dim, output_dim, conv, seed, noisy):
        super(DuelingNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.input_dim = input_dim
        self.state_dim = len(self.input_dim)
        self.output_dim = output_dim
        self.conv = conv

        if conv:
            self.conv_layer = ConvLayer()
            linear_input_dim = self.input_layer_size()
        else:
            linear_input_dim = input_dim[0]

        if noisy:
            Forward_layer = NoisyLinear
        else:
            Forward_layer = nn.Linear
        self.advantage_hidden = Forward_layer(linear_input_dim, layer_size)
        self.value_hidden = Forward_layer(linear_input_dim, layer_size)
        self.advantage = Forward_layer(layer_size, output_dim)
        self.value = Forward_layer(layer_size, 1)
        set_weights([self.advantage_hidden, self.value_hidden])        


    def input_layer_size(self):
        x = torch.zeros(self.input_dim)
        x = self.conv_layer.forward(x)
        return x.flatten().shape[0]

    def forward(self, state):
        if self.conv:
            state = self.conv_layer.forward(state)

        if len(state.shape)==1:
            state = state.unsqueeze(0)

        state = state.reshape(state.size(0), -1)
        state_A = torch.relu(self.advantage_hidden(state))
        state_V = torch.relu(self.value_hidden(state))

        value = self.value(state_V)
        value = value.expand(state.size(0), self.output_dim)
        advantage = self.advantage(state_A)
        qval = value + advantage - advantage.mean()
        return qval


class Rainbow(nn.Module):
    """Rainbow Network Setup"""

    def __init__(self, input_dim, output_dim, conv, seed, atom_size, Vmax, Vmin):
        super(Rainbow, self).__init__()
        torch.manual_seed(seed)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.atom_size = atom_size
        self.Vmax = Vmax
        self.Vmin = Vmin
        self.delta = (Vmax - Vmin) / (atom_size - 1)
        self.conv = conv

        if conv:
            self.conv_layer = ConvLayer()
            linear_input_dim = self.input_layer_size()
        else:
            linear_input_dim = input_dim[0]

        self.advantage_hidden = NoisyLinear(linear_input_dim, layer_size)
        self.value_hidden = NoisyLinear(linear_input_dim, layer_size)
        self.advantage = NoisyLinear(layer_size, output_dim * atom_size)
        self.value = NoisyLinear(layer_size, atom_size)
        # set_weights([self.advantage_hidden, self.value_hidden])

        self.register_buffer("supports", torch.arange(self.Vmin, self.Vmax + self.delta, self.delta))
        self.softmax = nn.Softmax(dim=1)

    def input_layer_size(self):
        x = torch.zeros(self.input_dim)
        x = self.conv_layer.forward(x)
        return x.flatten().shape[0]

    def forward(self, state):
        if self.conv:
            state = self.conv_layer.forward(state)

        if len(state.shape)==1:
            state = state.unsqueeze(0)
        
        batch_size = state.shape[0]
        state = state.reshape(state.size(0), -1)
        advantage_hidden = torch.relu(self.advantage_hidden(state))
        value_hidden = torch.relu(self.value_hidden(state))
        value = self.value(value_hidden).view(batch_size, 1, self.atom_size)
        advantage = self.advantage(advantage_hidden).view(batch_size, -1, self.atom_size)
        q_distr = value + advantage - advantage.mean(dim=1, keepdim=True)
        prob = self.softmax(q_distr.view(-1, self.atom_size)).view(-1, self.output_dim, self.atom_size)

        return prob

    def act(self, state):
        prob = self.forward(state).data.cpu()
        expected_value = prob.cpu() * self.supports.cpu()
        actions = expected_value.sum(2)
        return actions
