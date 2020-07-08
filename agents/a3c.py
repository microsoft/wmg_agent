# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.spec_reader import spec
AGENT_RANDOM_SEED = spec.val("AGENT_RANDOM_SEED")
A3C_T_MAX = spec.val("A3C_T_MAX")
LEARNING_RATE = spec.val("LEARNING_RATE")
DISCOUNT_FACTOR = spec.val("DISCOUNT_FACTOR")
GRADIENT_CLIP = spec.val("GRADIENT_CLIP")
WEIGHT_DECAY = spec.val("WEIGHT_DECAY")
AGENT_NET = spec.val("AGENT_NET")
ENTROPY_TERM_STRENGTH = spec.val("ENTROPY_TERM_STRENGTH")
REWARD_SCALE = spec.val("REWARD_SCALE")
ADAM_EPS = spec.val("ADAM_EPS")
ANNEAL_LR = spec.val("ANNEAL_LR")
if ANNEAL_LR:
    LR_GAMMA = spec.val("LR_GAMMA")
    from torch.optim.lr_scheduler import StepLR

torch.manual_seed(AGENT_RANDOM_SEED)


class A3cAgent(object):
    ''' A single-worker version of Asynchronous Advantage Actor-Critic (Mnih et al., 2016)'''
    def __init__(self, observation_space_size, action_space_size):
        if AGENT_NET == "GRU_Network":
            from agents.networks.gru import GRU_Network
            self.network = GRU_Network(observation_space_size, action_space_size)
        elif AGENT_NET == "WMG_Network":
            from agents.networks.wmg import WMG_Network
            self.network =  WMG_Network(observation_space_size, action_space_size)
        else:
            assert False  # The specified agent network was not found.
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=LEARNING_RATE,
                                          weight_decay=WEIGHT_DECAY, eps=ADAM_EPS)
        if ANNEAL_LR:
            self.scheduler = StepLR(self.optimizer, step_size=1, gamma=LR_GAMMA)
        print("{:11,d} trainable parameters".format(self.count_parameters(self.network)))

    def count_parameters(self, network):
        return sum(p.numel() for p in network.parameters() if p.requires_grad)

    def reset_adaptation_state(self):
        self.num_training_frames_in_buffer = 0
        self.values = []
        self.logps = []
        self.actions = []
        self.rewards = []

    def reset_state(self):
        self.net_state = self.network.init_state()
        self.last_action = None
        self.reset_adaptation_state()
        return 0

    def load_model(self, input_model):
        state_dict = torch.load(input_model)
        self.network.load_state_dict(state_dict)
        print('loaded agent model from {}'.format(input_model))

    def save_model(self, output_model):
        torch.save(self.network.state_dict(), output_model)

    def step(self, observation):
        self.last_observation = observation
        logits, self.value_tensor, self.net_state = self.network(self.last_observation, self.net_state)
        self.logp_tensor = F.log_softmax(logits, dim=-1)
        action_probs = torch.exp(self.logp_tensor)
        self.action_tensor = action_probs.multinomial(num_samples=1).data[0]
        self.last_action = self.action_tensor.numpy()[0]
        return self.last_action

    def adapt(self, reward, done, next_observation):
        ''' Buffer one frame of data for eventual training. '''
        self.values.append(self.value_tensor)
        self.logps.append(self.logp_tensor)
        self.actions.append(self.action_tensor)
        self.rewards.append(reward * REWARD_SCALE)
        self.num_training_frames_in_buffer += 1
        if done:
            self.adapt_on_end_of_episode()
        elif self.num_training_frames_in_buffer == A3C_T_MAX:
            self.adapt_on_end_of_sequence(next_observation)

    def adapt_on_end_of_episode(self):
        # Train with a next state value of zero, because there aren't any rewards after the end of the episode.
        # Reset_state will get called after this.
        self.train(0.)

    def adapt_on_end_of_sequence(self, next_observation):
        ''' Peek at the state value of the next observation, for TD calculation. '''
        _, next_value, _ = self.network(next_observation, self.net_state)
        # Train.
        self.train(next_value.squeeze().data.numpy())
        # Stop the gradients from flowing back into this window that we just trained on.
        self.net_state = self.network.detach_from_history(self.net_state)
        # Clear the experiment buffer.
        self.reset_adaptation_state()

    def train(self, next_value):
        ''' Update the weights. '''
        loss = self.loss_function(next_value, torch.cat(self.values), torch.cat(self.logps), torch.cat(self.actions), np.asarray(self.rewards))
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), GRADIENT_CLIP)
        self.optimizer.step()

    def anneal_lr(self):
        print('Scaling down the learning rate by {}'.format(LR_GAMMA))
        self.scheduler.step()

    def loss_function(self, next_value, values, logps, actions, rewards):
        td_target = next_value
        np_values = values.view(-1).data.numpy()
        buffer_size = len(rewards)
        td_targets = np.zeros(buffer_size)
        advantages = np.zeros(buffer_size)

        # Populate the td_target array (for value update) and the advantage array (for policy update).
        # Note that value errors should be backpropagated through the current value calculation for value updates,
        # but not for policy updates.
        for i in range(buffer_size - 1, -1, -1):
            td_target = rewards[i] + DISCOUNT_FACTOR * td_target
            advantage = td_target - np_values[i]
            td_targets[i] = td_target
            advantages[i] = advantage

        chosen_action_log_probs = logps.gather(1, actions.view(-1, 1))
        advantages_tensor = torch.FloatTensor(advantages.copy())
        policy_losses = chosen_action_log_probs.view(-1) * advantages_tensor
        policy_loss = -policy_losses.sum()

        td_targets_tensor = torch.FloatTensor(td_targets.copy())

        value_losses = td_targets_tensor - values[:, 0]
        value_loss = value_losses.pow(2).sum()

        entropy_losses = -logps * torch.exp(logps)
        entropy_loss = entropy_losses.sum()
        return policy_loss + 0.5 * value_loss - ENTROPY_TERM_STRENGTH * entropy_loss
