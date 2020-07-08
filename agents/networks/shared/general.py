# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LayerNorm(nn.Module):
    def __init__(self, size, variance_epsilon=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(size))
        self.beta = nn.Parameter(torch.zeros(size))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta


class LinearLayer(nn.Module):  # Adds weight initialization options on top of nn.Linear.
    def __init__(self, input_size, output_size):
        super(LinearLayer, self).__init__()
        self.layer = nn.Linear(input_size, output_size)
        self.layer.bias.data.fill_(0.)

    def forward(self, x):
        output = self.layer(x)
        return output


class ResidualLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super(ResidualLayer, self).__init__()
        self.linear_layer = LinearLayer(input_size, output_size)

    def forward(self, x, prev_input):
        output = self.linear_layer(x)
        output += prev_input
        return output


class SeparateActorCriticLayers(nn.Module):
    def __init__(self, input_size, num_layers, hidden_layer_size, action_space_size):
        super(SeparateActorCriticLayers, self).__init__()
        assert num_layers == 2
        self.critic_linear_1 = LinearLayer(input_size, hidden_layer_size)
        self.critic_linear_2 = LinearLayer(hidden_layer_size, 1)
        self.actor_linear_1 = LinearLayer(input_size, hidden_layer_size)
        self.actor_linear_2 = LinearLayer(hidden_layer_size, action_space_size)
        self.actor_linear_2.layer.weight.data.fill_(0.)

    def forward(self, x):
        value = self.critic_linear_1(x)
        value = F.relu(value)
        value = self.critic_linear_2(value)
        policy = self.actor_linear_1(x)
        policy = F.relu(policy)
        policy = self.actor_linear_2(policy)
        return policy, value


class SharedActorCriticLayers(nn.Module):
    def __init__(self, input_size, num_layers, hidden_layer_size, action_space_size):
        super(SharedActorCriticLayers, self).__init__()
        assert num_layers == 2
        self.linear_1 = LinearLayer(input_size, hidden_layer_size)
        self.critic_linear_2 = LinearLayer(hidden_layer_size, 1)
        self.actor_linear_2 = LinearLayer(hidden_layer_size, action_space_size)
        self.actor_linear_2.layer.weight.data.fill_(0.)

    def forward(self, x):
        shared = self.linear_1(x)
        shared = F.relu(shared)
        value = self.critic_linear_2(shared)
        policy = self.actor_linear_2(shared)
        return policy, value
