from __future__ import annotations
import time
import argparse
import multiprocessing as mp
import os
import torch
import torch as T
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from copy import deepcopy
from functools import partial
from torch import nn as nn, Tensor
from torch.nn import init
from math import sqrt
import math
import matplotlib.pyplot as plt
from DolphinEnv import DolphinEnv

"""
This is the Beyond The Rainbow algorithm from ICML 2025 (https://arxiv.org/abs/2411.03820)
This is setup to play Mario Kart Wii.
"""

############################################## MPS Spectral Norm Patch

def _patch_spectral_norm_for_mps():
    """
    Patch PyTorch's spectral norm to work on MPS by replacing vdot with mul+sum.
    vdot is not implemented for MPS, but the equivalent (a * b).sum() works.
    """
    from torch.nn.utils import parametrizations

    def _metal_vdot(a, b):
        return (a * b).sum()

    original_forward = parametrizations._SpectralNorm.forward

    def patched_forward(self, weight):
        weight_mat = parametrizations._SpectralNorm._reshape_weight_to_matrix(self, weight)
        if self.training:
            with torch.no_grad():
                for _ in range(self.n_power_iterations):
                    self._v = F.normalize(torch.mv(weight_mat.t(), self._u), dim=0, eps=self.eps)
                    self._u = F.normalize(torch.mv(weight_mat, self._v), dim=0, eps=self.eps)
        sigma = _metal_vdot(self._u, torch.mv(weight_mat, self._v))
        return weight / sigma

    parametrizations._SpectralNorm.forward = patched_forward

# Note: _patch_spectral_norm_for_mps() is called later only when --device mps is used

############################################## Networks Section

class FactorizedNoisyLinear(nn.Module):
    """ The factorized Gaussian noise layer for noisy-nets dqn. """
    def __init__(self, in_features: int, out_features: int, sigma_0=0.5, self_norm=False) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_0 = sigma_0

        # weight: w = \mu^w + \sigma^w . \epsilon^w
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))

        # bias: b = \mu^b + \sigma^b . \epsilon^b
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))

        if self_norm:
            self.reset_parameters_self_norm()
        else:
            self.reset_parameters()
        self.reset_noise()

        self.disable_noise()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        # initialization is similar to Kaiming uniform (He. initialization) with fan_mode=fan_in
        scale = 1 / sqrt(self.in_features)

        init.uniform_(self.weight_mu, -scale, scale)
        init.uniform_(self.bias_mu, -scale, scale)

        init.constant_(self.weight_sigma, self.sigma_0 * scale)
        init.constant_(self.bias_sigma, self.sigma_0 * scale)

    @torch.no_grad()
    def reset_parameters_self_norm(self) -> None:
        # initialization is similar to Kaiming uniform (He. initialization) with fan_mode=fan_in

        nn.init.normal_(self.weight_mu, std=1 / math.sqrt(self.out_features))
        if self.bias_mu is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_mu)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias_mu, -bound, bound)

    @torch.no_grad()
    def _get_noise(self, size: int) -> Tensor:
        noise = torch.randn(size, device=self.weight_mu.device)
        # f(x) = sgn(x)sqrt(|x|)
        return noise.sign().mul_(noise.abs().sqrt_())

    @torch.no_grad()
    def reset_noise(self) -> None:
        # like in eq 10 and 11 of the paper
        epsilon_in = self._get_noise(self.in_features)
        epsilon_out = self._get_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    @torch.no_grad()
    def disable_noise(self) -> None:
        self.weight_epsilon[:] = 0
        self.bias_epsilon[:] = 0

    def forward(self, input: Tensor) -> Tensor:
        # y = wx + d, where
        # w = \mu^w + \sigma^w * \epsilon^w
        # b = \mu^b + \sigma^b * \epsilon^b
        return F.linear(input,
                        self.weight_mu + self.weight_sigma*self.weight_epsilon,
                        self.bias_mu + self.bias_sigma*self.bias_epsilon)


class Dueling(nn.Module):
    """ The dueling branch used in all nets that use dueling-dqn. """
    def __init__(self, value_branch, advantage_branch):
        super().__init__()
        self.flatten = nn.Flatten()
        self.value_branch = value_branch
        self.advantage_branch = advantage_branch

    def forward(self, x, advantages_only=False):
        x = self.flatten(x)
        advantages = self.advantage_branch(x)
        if advantages_only:
            return advantages

        value = self.value_branch(x)
        return value + (advantages - torch.mean(advantages, dim=1, keepdim=True))


class ImpalaCNNResidual(nn.Module):
    """
    Simple residual block used in the large IMPALA CNN.
    """
    def __init__(self, depth, norm_func, activation=nn.ReLU):
        super().__init__()

        self.activation = activation()

        self.conv_0 = norm_func(nn.Conv2d(in_channels=depth, out_channels=depth, kernel_size=3, stride=1, padding=1))
        self.conv_1 = norm_func(nn.Conv2d(in_channels=depth, out_channels=depth, kernel_size=3, stride=1, padding=1))

    #@torch.autocast('cuda')
    def forward(self, x):

        x_ = self.conv_0(self.activation(x))

        x_ = self.conv_1(self.activation(x_))
        return x + x_


class ImpalaCNNBlock(nn.Module):
    """
    Three of these blocks are used in the large IMPALA CNN.
    """
    def __init__(self, depth_in, depth_out, norm_func, activation=nn.ReLU, layer_norm=False,
                 layer_norm_shapes=False):
        super().__init__()
        self.layer_norm = layer_norm

        self.conv = nn.Conv2d(in_channels=depth_in, out_channels=depth_out, kernel_size=3, stride=1, padding=1)
        self.max_pool = nn.MaxPool2d(3, 2, padding=1)

        if self.layer_norm:
            self.norm_layer1 = nn.LayerNorm(layer_norm_shapes[0])

        self.residual_0 = ImpalaCNNResidual(depth_out, norm_func=norm_func, activation=activation)
        self.residual_1 = ImpalaCNNResidual(depth_out, norm_func=norm_func, activation=activation)

    def forward(self, x):
        x = self.conv(x)

        if self.layer_norm:
            x = self.norm_layer1(x)

        x = self.max_pool(x)

        x = self.residual_0(x)

        x = self.residual_1(x)

        return x


class ImpalaCNNLargeIQN(nn.Module):
    """
    Implementation of the large variant of the IMPALA CNN introduced in Espeholt et al. (2018).
    """
    def __init__(self, in_depth, actions, model_size=2, spectral=True, device='cuda:0', num_tau=8, maxpool_size=6,
                 linear_size=512, ncos=64, layer_norm=True):
        super().__init__()

        self.start = time.time()
        self.model_size = model_size
        self.actions = actions
        self.device = device

        self.in_depth = in_depth

        conv_activation = nn.ReLU
        activation = nn.ReLU

        self.linear_size = linear_size
        self.num_tau = num_tau

        self.maxpool_size = maxpool_size

        self.layer_norm = layer_norm

        self.n_cos = ncos
        #self.pis = torch.FloatTensor([np.pi * i for i in range(self.n_cos)]).view(1, 1, self.n_cos).to(device)
        self.register_buffer('pis', torch.tensor([np.pi * i for i in range(self.n_cos)], dtype=torch.float32).view(1, 1,
                                                                                                                   self.n_cos))

        linear_layer = FactorizedNoisyLinear

        def identity(p): return p

        if spectral:
            norm_func = torch.nn.utils.parametrizations.spectral_norm
        else:
            norm_func = identity

        self.conv = nn.Sequential(
            ImpalaCNNBlock(in_depth, int(16*model_size), norm_func=norm_func, activation=conv_activation,),
            ImpalaCNNBlock(int(16*model_size), int(32*model_size), norm_func=norm_func, activation=conv_activation),
            ImpalaCNNBlock(int(32 * model_size), int(32 * model_size), norm_func=norm_func, activation=conv_activation),
            torch.nn.AdaptiveMaxPool2d((6, 6))
        )

        self.conv.add_module('conv_activation', activation())

        self.conv_out_size = int(32 * model_size * 6 * 6)

        self.cos_embedding = nn.Linear(self.n_cos, self.conv_out_size)

        self.linear_layersV = nn.Sequential()
        self.linear_layersA = nn.Sequential()

        self.linear_layersV.add_module('fc1V', linear_layer(self.conv_out_size, self.linear_size))
        self.linear_layersA.add_module('fc1A', linear_layer(self.conv_out_size, self.linear_size))

        if self.layer_norm:
            self.linear_layersV.add_module('LN_V', nn.LayerNorm(self.linear_size))
            self.linear_layersA.add_module('LN_A', nn.LayerNorm(self.linear_size))

        self.linear_layersV.add_module('actV', activation())
        self.linear_layersA.add_module('actA', activation())

        self.linear_layersV.add_module('fc2V', linear_layer(self.linear_size, 1))
        self.linear_layersA.add_module('fc2A', linear_layer(self.linear_size, actions))

        self.linear_layers = Dueling(self.linear_layersV, self.linear_layersA)

        self.to(device)

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, inputt, advantages_only=False):
        """
        Quantile Calculation depending on the number of tau

        Return:
        quantiles [ shape of (batch_size, num_tau, action_size)]
        taus [shape of ((batch_size, num_tau, 1))]

        """
        batch_size = inputt.size()[0]

        #print("Forward Func")
        inputt = inputt.float() / 255
        #print(input.abs().sum().item())

        x = self.conv(inputt)

        #print(x.device)
        x = x.view(batch_size, -1)

        cos, taus = self.calc_cos(batch_size, self.num_tau)  # cos shape (batch, num_tau, layer_size)
        cos = cos.view(batch_size * self.num_tau, self.n_cos)
        cos_x = torch.relu(self.cos_embedding(cos)).view(batch_size, self.num_tau, self.conv_out_size)  # (batch, n_tau, layer)

        # x has shape (batch, layer_size) for multiplication –> reshape to (batch, 1, layer)
        x = (x.unsqueeze(1) * cos_x).view(batch_size * self.num_tau, self.conv_out_size)

        out = self.linear_layers(x)

        #print(out.device)
        return out.view(batch_size, self.num_tau, self.actions), taus

    def qvals(self, inputs, advantages_only=False):
        quantiles, _ = self.forward(inputs, advantages_only)

        actions = quantiles.mean(dim=1)

        return actions

    def calc_cos(self, batch_size, n_tau=8):
        """
        Calculating the cosinus values depending on the number of tau samples
        """
        taus = torch.rand(batch_size, n_tau, 1, device=self.pis.device)#(batch_size, n_tau, 1)

        #taus = torch.linspace(0.01, 1.0, 100).unsqueeze(0).repeat(batch_size, 1).unsqueeze(-1)

        cos = torch.cos(taus*self.pis)

        #assert cos.shape == (batch_size, n_tau, self.n_cos), "cos shape is incorrect"
        return cos, taus

    def save_checkpoint(self, name):
        #print('... saving checkpoint ...')
        torch.save(self.state_dict(), name + ".model")

    def load_checkpoint(self, name):
        #print('... loading checkpoint ...')
        self.load_state_dict(torch.load(name, map_location=self.device))


################# Now Entering the Prioritized Experience Replay Section

# SumTree
# a binary tree data structure where the parent’s value is the sum of its children
class SumTree():
  def __init__(self, size, procgen=False):
    self.index = 0
    self.size = size
    self.full = False  # Used to track actual capacity
    self.tree_start = 2**(size-1).bit_length()-1  # Put all used node leaves on last tree level
    self.sum_tree = np.zeros((self.tree_start + self.size,), dtype=np.float32)
    self.max = 1  # Initial max value to return (1 = 1^ω)

  # Updates nodes values from current tree
  def _update_nodes(self, indices):
    children_indices = indices * 2 + np.expand_dims([1, 2], axis=1)
    self.sum_tree[indices] = np.sum(self.sum_tree[children_indices], axis=0)

  # Propagates changes up tree given tree indices
  def _propagate(self, indices):
    parents = (indices - 1) // 2
    unique_parents = np.unique(parents)
    self._update_nodes(unique_parents)
    if parents[0] != 0:
      self._propagate(parents)

  # Propagates single value up tree given a tree index for efficiency
  def _propagate_index(self, index):
    parent = (index - 1) // 2
    left, right = 2 * parent + 1, 2 * parent + 2
    self.sum_tree[parent] = self.sum_tree[left] + self.sum_tree[right]
    if parent != 0:
      self._propagate_index(parent)

  # Updates values given tree indices
  def update(self, indices, values):
    self.sum_tree[indices] = values  # Set new values
    self._propagate(indices)  # Propagate values
    current_max_value = np.max(values)
    self.max = max(current_max_value, self.max)

  # Updates single value given a tree index for efficiency
  def _update_index(self, index, value):
    self.sum_tree[index] = value  # Set new value
    self._propagate_index(index)  # Propagate value
    self.max = max(value, self.max)

  def append(self, value):
    self._update_index(self.index + self.tree_start, value)  # Update tree
    self.index = (self.index + 1) % self.size  # Update index
    self.full = self.full or self.index == 0  # Save when capacity reached
    self.max = max(value, self.max)

  # Searches for the location of values in sum tree
  def _retrieve(self, indices, values):
    children_indices = (indices * 2 + np.expand_dims([1, 2], axis=1)) # Make matrix of children indices
    # If indices correspond to leaf nodes, return them
    if children_indices[0, 0] >= self.sum_tree.shape[0]:
      return indices
    # If children indices correspond to leaf nodes, bound rare outliers in case total slightly overshoots
    elif children_indices[0, 0] >= self.tree_start:
      children_indices = np.minimum(children_indices, self.sum_tree.shape[0] - 1)
    left_children_values = self.sum_tree[children_indices[0]]
    successor_choices = np.greater(values, left_children_values).astype(np.int32)  # Classify which values are in left or right branches
    successor_indices = children_indices[successor_choices, np.arange(indices.size)] # Use classification to index into the indices matrix
    successor_values = values - successor_choices * left_children_values  # Subtract the left branch values when searching in the right branch
    return self._retrieve(successor_indices, successor_values)

  # Searches for values in sum tree and returns values, data indices and tree indices
  def find(self, values):
    indices = self._retrieve(np.zeros(values.shape, dtype=np.int32), values)
    data_index = indices - self.tree_start
    return (self.sum_tree[indices], data_index, indices)  # Return values, data indices, tree indices

  def total(self):
    return self.sum_tree[0]


class PER:
    def __init__(self, size, device, n, envs, gamma, alpha=0.2, beta=0.4, framestack=4, imagex=84, imagey=84, rgb=False):

        self.st = SumTree(size)
        self.data = [None for _ in range(size)]
        self.index = 0
        self.size = size

        # this is the number of frames, not the number of transitions
        # the technical size to ensure there are no errors with overwritten memory in theory is very high-
        # (2*framestack - overlap) * first_states + non_first_states
        # with N=3, framestack=4, size=1M, average ep length 20, we need a total frame storage of around 1.35M
        # this however is still pretty light given it uses discrete memory. Careful when using RGB though,
        # as you have to store every frame so memory usage will be notably higher.
        if rgb:
            self.storage_size = int(size * 4)
        else:
            self.storage_size = int(size * 1.25)
        self.gamma = gamma
        self.capacity = 0

        self.point_mem_idx = 0

        self.state_mem_idx = 0
        self.reward_mem_idx = 0

        self.imagex = imagex
        self.imagey = imagey

        self.max_prio = 1

        self.framestack = framestack

        self.alpha = alpha
        self.beta = beta
        self.eps = 1e-6  # small constant to stop 0 probability
        self.device = device

        self.last_terminal = [True for i in range(envs)]
        self.tstep_counter = [0 for i in range(envs)]

        self.n_step = n
        self.state_buffer = [[] for i in range(envs)]
        self.reward_buffer = [[] for i in range(envs)]

        if rgb:
            self.state_mem = np.zeros((self.storage_size, 3, self.imagey, self.imagex), dtype=np.uint8)
        else:
            self.state_mem = np.zeros((self.storage_size, self.imagey, self.imagex), dtype=np.uint8)
        self.action_mem = np.zeros(self.storage_size, dtype=np.int64)
        self.reward_mem = np.zeros(self.storage_size, dtype=float)
        self.done_mem = np.zeros(self.storage_size, dtype=bool)
        self.trun_mem = np.zeros(self.storage_size, dtype=bool)

        # everything here is stored as ints as they are just pointers to the actual memory
        # reward contains N values. The first value contains the action. The set of N contains the pointers for both
        # the reward and dones
        self.trans_dtype = np.dtype([('state', int, self.framestack), ('n_state', int, self.framestack),
                                     ('reward', int, self.n_step)])

        self.blank_trans = (np.zeros(self.framestack, dtype=int), np.zeros(self.framestack, dtype=int),
                            np.zeros(self.n_step, dtype=int))

        self.pointer_mem = np.array([self.blank_trans] * size, dtype=self.trans_dtype)

        self.overlap = self.framestack - self.n_step

        # the "technically correct" way to do this is to use the min priority in the whole buffer. However,
        # we instead just use the min from each batch. In our experience, this makes effectively no difference
        # and is significantly faster. The code is still here, but has been commented out
        #self.priority_min = [float('inf') for _ in range(2 * self.size)]

    def append(self, state, action, reward, n_state, done, trun, stream, prio=True):

        # append to memory
        self.append_memory(state, action, reward, n_state, done, trun, stream)

        # append to pointer
        self.append_pointer(stream, prio)

        if done or trun:
            self.finalize_experiences(stream)
            self.state_buffer[stream] = []
            self.reward_buffer[stream] = []

        self.last_terminal[stream] = done or trun

    # def _set_priority_min(self, idx, priority_alpha):
    #     idx += self.size
    #     self.priority_min[idx] = priority_alpha
    #     while idx >= 2:
    #         idx //= 2
    #         self.priority_min[idx] = min(self.priority_min[2 * idx], self.priority_min[2 * idx + 1])

    def append_pointer(self, stream, prio):

        while len(self.state_buffer[stream]) >= self.framestack + self.n_step and len(self.reward_buffer[stream]) >= self.n_step:
            # First array in the experience
            state_array = self.state_buffer[stream][:self.framestack]

            # Second array in the experience (starts after N frames)
            n_state_array = self.state_buffer[stream][self.n_step:self.n_step + self.framestack]

            # Reward array (first N rewards)
            reward_array = self.reward_buffer[stream][:self.n_step]

            # Add the experience to the list
            self.pointer_mem[self.point_mem_idx] = (np.array(state_array, dtype=int), np.array(n_state_array, dtype=int),
                                                             np.array(reward_array, dtype=int))

            #self._set_priority_min(self.point_mem_idx, sqrt(self.max_prio))
            self.st.append(self.max_prio ** self.alpha)

            self.capacity = min(self.size, self.capacity + 1)
            self.point_mem_idx = (self.point_mem_idx + 1) % self.size

            # Remove the first state and reward from the buffers to slide the window
            self.state_buffer[stream].pop(0)
            self.reward_buffer[stream].pop(0)
            self.beta = 0

    def finalize_experiences(self, stream):
        # Process remaining states and rewards at the end of an episode
        while len(self.state_buffer[stream]) >= self.framestack and len(self.reward_buffer[stream]) > 0:
            # First array in the experience
            first_array = self.state_buffer[stream][:self.framestack]

            # Second array in the experience (Final `framestack` elements)
            second_array = self.state_buffer[stream][-self.framestack:]

            # Reward array
            reward_array = self.reward_buffer[stream][:]
            while len(reward_array) < self.n_step:
                reward_array.extend([0])

            # Add the experience
            self.pointer_mem[self.point_mem_idx] = (np.array(first_array, dtype=int), np.array(second_array, dtype=int),
                                                             np.array(reward_array, dtype=int))

            #self._set_priority_min(self.point_mem_idx, sqrt(self.max_prio))
            self.st.append(self.max_prio ** self.alpha)

            self.point_mem_idx = (self.point_mem_idx + 1) % self.size
            self.capacity = min(self.size, self.capacity + 1)

            # Remove the first state and reward from the buffers to slide the window
            self.state_buffer[stream].pop(0)
            if len(self.reward_buffer[stream]) > 0:
                self.reward_buffer[stream].pop(0)

    def append_memory(self, state, action, reward, n_state, done, trun, stream):

        if self.last_terminal[stream]:
            # add full transition
            for i in range(self.framestack):
                self.state_mem[self.state_mem_idx] = state[i]
                self.state_buffer[stream].append(self.state_mem_idx)
                self.state_mem_idx = (self.state_mem_idx + 1) % self.storage_size

            # remember n_step is not applied in this memory
            self.state_mem[self.state_mem_idx] = n_state[self.framestack - 1]
            self.state_buffer[stream].append(self.state_mem_idx)
            self.state_mem_idx = (self.state_mem_idx + 1) % self.storage_size

            self.action_mem[self.reward_mem_idx] = action
            self.reward_mem[self.reward_mem_idx] = reward
            self.done_mem[self.reward_mem_idx] = done
            self.trun_mem[self.reward_mem_idx] = trun

            self.reward_buffer[stream].append(self.reward_mem_idx)
            self.reward_mem_idx = (self.reward_mem_idx + 1) % self.storage_size

            self.tstep_counter[stream] = 0

        else:
            # just add relevant info
            self.state_mem[self.state_mem_idx] = n_state[self.framestack - 1]
            self.state_buffer[stream].append(self.state_mem_idx)
            self.state_mem_idx = (self.state_mem_idx + 1) % self.storage_size

            self.action_mem[self.reward_mem_idx] = action
            self.reward_mem[self.reward_mem_idx] = reward
            self.done_mem[self.reward_mem_idx] = done
            self.trun_mem[self.reward_mem_idx] = trun

            self.reward_buffer[stream].append(self.reward_mem_idx)
            self.reward_mem_idx = (self.reward_mem_idx + 1) % self.storage_size

    def sample(self, batch_size, count=0):

        # get total sumtree priority
        p_total = self.st.total()

        # first use sumtree prios to get the indices
        segment_length = p_total / batch_size
        segment_starts = np.arange(batch_size) * segment_length

        samples = np.random.uniform(0.0, segment_length, [batch_size]) + segment_starts

        prios, idxs, tree_idxs = self.st.find(samples)

        probs = prios / p_total

        # fetch the pointers by using indices
        pointers = self.pointer_mem[idxs]

        # Extract the pointers into separate arrays
        state_pointers = np.array([p[0] for p in pointers])
        n_state_pointers = np.array([p[1] for p in pointers])
        reward_pointers = np.array([p[2] for p in pointers])
        if self.n_step > 1:
            action_pointers = np.array([p[2][0] for p in pointers])
        else:
            action_pointers = np.array([p[2] for p in pointers])

        # get state info
        states = torch.tensor(self.state_mem[state_pointers], dtype=torch.uint8)
        n_states = torch.tensor(self.state_mem[n_state_pointers], dtype=torch.uint8)

        # reward and dones just use the same pointer. actions just use the first one
        rewards = self.reward_mem[reward_pointers]
        dones = self.done_mem[reward_pointers]
        truns = self.trun_mem[reward_pointers]
        actions = self.action_mem[action_pointers]

        # apply n_step cumulation to rewards and dones
        if self.n_step > 1:
            rewards, dones = self.compute_discounted_rewards_batch(rewards, dones, truns)

        #prob_min = self.priority_min[1] / p_total
        #max_weight = (prob_min * self.capacity) ** (-self.beta)

        # Compute importance-sampling weights w
        weights = (self.capacity * probs) ** -self.alpha  # self.beta originally this was an accident but actually performed better
        # seems to perform better without this for some reason? This is disabled from the agent class

        weights = torch.tensor(weights / weights.max(), dtype=torch.float32,
                               device=self.device)  # Normalise by max importance-sampling weight from batch

        if torch.isnan(weights).any():
            # There is a very very small chance to sample something outside of the currently filled range before the
            # buffer is full. In this case, we just sample again. If this happens more than a couple of times,
            # something else is probably broken
            if count >= 5:
                raise Exception("Weights Contained NaNs!")
            return self.sample(batch_size, count + 1)

        # move to pytorch GPU tensors
        states = states.to(torch.float32).to(self.device)
        n_states = n_states.to(torch.float32).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.bool, device=self.device)
        actions = torch.tensor(actions, dtype=torch.int64, device=self.device)

        # return batch
        return tree_idxs, states, actions, rewards, n_states, dones, weights

    # def compute_discounted_rewards_batch_no_loop(self, rewards_batch, dones_batch, truns_batch):
    #     """
    #     Compute discounted rewards for a batch using NumPy, in a vectorized manner.
    #
    #     Parameters:
    #       rewards_batch (np.ndarray): 2D array of rewards with shape (batch_size, n_step)
    #       dones_batch (np.ndarray): 2D array of dones with shape (batch_size, n_step)
    #       truns_batch (np.ndarray): 2D array of truncation flags with shape (batch_size, n_step)
    #
    #     Returns:
    #       discounted_rewards (np.ndarray): 1D array of discounted rewards for each batch element,
    #                                        computed only up to (and including) the first step
    #                                        where either dones or truncation occurs.
    #       cumulative_dones (np.ndarray): 1D boolean array indicating for each sample whether
    #                                      the break was due to a done event.
    #     """
    #     batch_size, n_step = rewards_batch.shape
    #
    #     # Compute a boolean mask indicating where a break occurs
    #     break_mask = (dones_batch == 1) | (truns_batch == 1)  # shape: (batch_size, n_step)
    #
    #     # For each sample, find the first index where break_mask is True.
    #     # If no break occurs in a sample, set the break index to n_step - 1.
    #     break_exists = break_mask.any(axis=1)
    #     break_indices = np.empty(batch_size, dtype=int)
    #     if np.any(break_exists):
    #         # np.argmax returns the first index where the value is maximum.
    #         break_indices[break_exists] = np.argmax(break_mask[break_exists], axis=1)
    #     break_indices[~break_exists] = n_step - 1
    #
    #     # Create a mask that is True for all steps up to (and including) the break index.
    #     j_idx = np.arange(n_step)  # shape: (n_step,)
    #     valid_mask = j_idx[None, :] <= break_indices[:, None]  # shape: (batch_size, n_step)
    #
    #     # Create discount factors: [gamma^0, gamma^1, ..., gamma^(n_step-1)]
    #     discounts = self.gamma ** np.arange(n_step)  # shape: (n_step,)
    #
    #     # Compute the discounted rewards and zero out entries after the break index.
    #     discounted = rewards_batch * discounts[None, :]  # broadcast to (batch_size, n_step)
    #     discounted *= valid_mask.astype(rewards_batch.dtype)
    #     discounted_rewards = np.sum(discounted, axis=1)
    #
    #     # For cumulative_dones, take the 'dones' value at the break index for each sample.
    #     cumulative_dones = break_mask[np.arange(batch_size), break_indices]
    #
    #     return discounted_rewards, cumulative_dones

    def compute_discounted_rewards_batch(self, rewards_batch, dones_batch, truns_batch):
        """
        Compute discounted rewards for a batch of rewards and dones.

        Parameters:
        rewards_batch (np.ndarray): 2D array of rewards with shape (batch_size, n_step)
        dones_batch (np.ndarray): 2D array of dones with shape (batch_size, n_step)

        Returns:
        np.ndarray: 1D array of discounted rewards for each element in the batch
        np.ndarray: 1D array of cumulative dones (True if any done is True in the sequence)
        """
        batch_size, n_step = rewards_batch.shape
        discounted_rewards = np.zeros(batch_size)
        cumulative_dones = np.zeros(batch_size, dtype=bool)

        for i in range(batch_size):
            cumulative_discount = 1
            for j in range(n_step):
                discounted_rewards[i] += cumulative_discount * rewards_batch[i, j]
                if dones_batch[i, j] == 1:
                    cumulative_dones[i] = True
                    break
                elif truns_batch[i, j] == 1:
                    break
                cumulative_discount *= self.gamma

        return discounted_rewards, cumulative_dones

    def update_priorities(self, idxs, priorities):
        priorities = priorities + self.eps

        # for idx, priority in zip(idxs, priorities):
        #     self._set_priority_min(idx - self.size + 1, sqrt(priority))

        if np.isnan(priorities).any():
            print("NaN found in priority!")
            print(f"priorities: {priorities}")

        self.max_prio = max(self.max_prio, np.max(priorities))
        self.st.update(idxs, priorities ** self.alpha)


############## A few smaller functions to assist the main program

class EpsilonGreedy:
    def __init__(self, eps_start, eps_steps, eps_final, action_space):
        self.eps = eps_start
        self.steps = eps_steps
        self.eps_final = eps_final
        self.action_space = action_space

    def update_eps(self):
        self.eps = max(self.eps - (self.eps - self.eps_final) / self.steps, self.eps_final)

    def choose_action(self):
        if np.random.random() > self.eps:
            return None
        else:
            return np.random.choice(self.action_space)


def randomise_action_batch(x, probs, n_actions):
    mask = torch.rand(x.shape) < probs

    # Generate random values to replace the selected elements
    random_values = torch.randint(0, n_actions, x.shape)

    # Apply the mask to replace elements in the tensor with random values
    x[mask] = random_values[mask]

    return x


def choose_eval_action(observation, eval_net, n_actions, device, rng):
    with torch.no_grad():
        state = T.tensor(observation, dtype=T.float).to(device)
        qvals = eval_net.qvals(state, advantages_only=True)
        x = T.argmax(qvals, dim=1).cpu()

        if rng > 0.:
            # Generate a mask with the given probability
            x = randomise_action_batch(x, 0.01, n_actions)

    return x


def create_network(input_dims, n_actions, spectral_norm, device, model_size, maxpool_size,
                   linear_size, num_tau, ncos, layer_norm=True):

    return ImpalaCNNLargeIQN(input_dims[0], n_actions, spectral=spectral_norm, device=device,
                             model_size=model_size, num_tau=num_tau, maxpool_size=maxpool_size,
                             linear_size=linear_size, ncos=ncos, layer_norm=layer_norm)


#################### The big ol agent class, be prepared

class Agent:
    def __init__(self, n_actions, input_dims, device, num_envs, agent_name, total_frames, testing=False, batch_size=256
                 , rr=1, maxpool_size=6, lr=1e-4, target_replace=500, spectral=True, discount=0.997, taus=8, model_size=2,
                 linear_size=512, ncos=64, non_factorised=False, replay_period=1, framestack=4, rgb=False, imagex=84,
                 imagey=84, per_alpha=0.2, max_mem_size=1048576, eps_steps=2000000, eps_disable=True, n=3,
                 munch_alpha=0.9, grad_clip=10, layer_norm=True, spi=1):

        self.per_alpha = per_alpha

        self.procgen = True if input_dims[1] == 64 else False
        self.grad_clip = grad_clip

        self.n_actions = n_actions
        self.input_dims = input_dims
        self.device = device
        self.agent_name = agent_name
        self.testing = testing

        self.layer_norm = layer_norm

        self.loading_checkpoint = False

        self.per_beta = 0.45

        self.replay_ratio = int(rr) if rr > 0.99 else float(rr)
        self.total_frames = total_frames
        self.num_envs = num_envs

        if self.testing:
            self.min_sampling_size = 8000
        else:
            self.min_sampling_size = 200000

        self.lr = lr

        # this is the number of env steps per grad step
        self.replay_period = replay_period
        self.replay_period_cnt = 0

        # samples per insert ratio (SPI)
        self.spi = spi

        # replay period is how many vectorized env steps we take before doing a grad step (int)
        # spi is how many grad steps we do when doing a grad step

        self.total_grad_steps = (self.total_frames - self.min_sampling_size) / ((self.replay_period * self.num_envs) / self.spi)

        self.priority_weight_increase = (1 - self.per_beta) / self.total_grad_steps

        self.action_space = [i for i in range(self.n_actions)]
        self.learn_step_counter = 0

        self.chkpt_dir = ""

        self.n = n
        self.gamma = discount
        self.discount_anneal = False
        self.batch_size = batch_size

        self.model_size = model_size  # Scaling of IMPALA network
        self.maxpool_size = maxpool_size

        self.spectral_norm = spectral

        # this option is only available for non-impala. I could add it, but factorised seemed
        # to perform the same and is faster
        self.non_factorised = non_factorised

        self.ncos = ncos

        self.entropy_tau = 0.03
        self.lo = -1
        self.alpha = munch_alpha

        # 1 Million rounded to the nearest power of 2 for tree implementation
        self.max_mem_size = max_mem_size

        self.replace_target_cnt = target_replace  # This is the number of grad steps - could be a little jank
        # when changing num_envs/batch size/replay ratio

        # Best used value is 32000 frames per replace. For bs 256, this is 500. For bs 16, this is every 8000!

        self.num_tau = taus

        if not self.loading_checkpoint and not self.testing:
            self.eps_start = 1.0
            # divided by 4 is due to frameskip
            self.eps_steps = eps_steps
            self.eps_final = 0.01
        else:
            self.eps_start = 0.00
            self.eps_steps = eps_steps
            self.eps_final = 0.00

        self.eps_disable = eps_disable
        self.epsilon = EpsilonGreedy(self.eps_start, self.eps_steps, self.eps_final, self.action_space)

        self.linear_size = linear_size

        self.framestack = framestack
        self.rgb = rgb
        self.memory = PER(self.max_mem_size, device, self.n, num_envs, self.gamma, alpha=self.per_alpha,
                          beta=self.per_beta, framestack=self.framestack, rgb=self.rgb, imagex=imagex, imagey=imagey)

        self.network_creator_fn = partial(create_network, self.input_dims, self.n_actions, self.spectral_norm,
                                          self.device, self.model_size, self.maxpool_size, self.linear_size,
                                          self.num_tau, self.ncos, layer_norm=self.layer_norm)

        self.net = self.network_creator_fn()
        self.tgt_net = self.network_creator_fn()

        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr, eps=0.005 / self.batch_size)  # 0.00015

        self.net.train()

        self.eval_net = None

        for param in self.tgt_net.parameters():
            param.requires_grad = False

        self.env_steps = 0
        self.grad_steps = 0

        self.replay_ratio_cnt = 0
        self.eval_mode = False

    def prep_evaluation(self):
        self.eval_net = deepcopy(self.net)
        self.disable_noise(self.eval_net)

    @torch.no_grad()
    def reset_noise(self, net):
        for m in net.modules():
            if isinstance(m, FactorizedNoisyLinear):
                m.reset_noise()

    @torch.no_grad()
    def disable_noise(self, net):
        for m in net.modules():
            if isinstance(m, FactorizedNoisyLinear):
                m.disable_noise()

    def choose_action(self, observation):
        # this chooses an action for a batch. Can be used with a batch of 1 if needed though
        with T.no_grad():
            if not self.eval_mode:
                self.reset_noise(self.net)

            state = T.tensor(observation, dtype=T.float).to(self.net.device)

            qvals = self.net.qvals(state, advantages_only=True)
            x = T.argmax(qvals, dim=1).cpu()

            # epsilon actions are disabled after 100M frames
            if self.env_steps < self.min_sampling_size or (self.env_steps < 25000000 and self.eps_disable):

                probs = self.epsilon.eps
                x = randomise_action_batch(x, probs, self.n_actions)

            return x

    def store_transition(self, state, action, reward, next_state, done, trun, stream, prio=True):

        if self.rgb:
            # expand dims to create "framestack" dim, so it works with my replay buffer
            state = np.expand_dims(state, axis=0)
            next_state = np.expand_dims(next_state, axis=0)

        self.memory.append(state, action, reward, next_state, done, trun, stream, prio=prio)

        self.epsilon.update_eps()
        self.env_steps += 1

        if self.env_steps == self.min_sampling_size + 100:
            print("Training is running successfully!")

    def replace_target_network(self):
        self.tgt_net.load_state_dict(self.net.state_dict())

    def save_model(self):
        self.net.save_checkpoint(self.agent_name + "_" + str(int((self.env_steps // 250000))) + "M")

    def load_models(self, name):
        self.net.load_checkpoint(name)
        self.tgt_net.load_checkpoint(name)

    def learn(self):
        if self.replay_period != 1:
            if self.replay_period_cnt == 0:
                for i in range(self.spi):
                    self.learn_call()
            self.replay_period_cnt = (self.replay_period_cnt + 1) % self.replay_period
        else:
            for i in range(self.spi):
                self.learn_call()

    def learn_call(self):
        if self.env_steps < self.min_sampling_size:
            return

        self.reset_noise(self.tgt_net)

        if self.grad_steps % self.replace_target_cnt == 0:
            self.replace_target_network()

        idxs, states, actions, rewards, next_states, dones, weights = self.memory.sample(self.batch_size)

        # use this code to check your states are correct!
        # If you apply BTR to a custom env and don't check your states first, you are killing both
        # trees and your own time

        # plt.imshow(states[0][0].unsqueeze(dim=0).cpu().permute(1, 2, 0))
        # plt.show()
        #
        # plt.imshow(states[0][1].unsqueeze(dim=0).cpu().permute(1, 2, 0))
        # plt.show()
        #
        # plt.imshow(states[0][2].unsqueeze(dim=0).cpu().permute(1, 2, 0))
        # plt.show()
        #
        # plt.imshow(states[1][0].unsqueeze(dim=0).cpu().permute(1, 2, 0))
        # plt.show()
        #
        # plt.imshow(states[2][0].unsqueeze(dim=0).cpu().permute(1, 2, 0))
        # plt.show()

        self.optimizer.zero_grad()

        # Perform a single forward pass with gradients
        q_k, taus = self.net(states)  # q_k: (batch_size, num_tau, n_actions)

        # Detach q_k for target computation to avoid tracking gradients
        q_k_detached = q_k.detach()

        with torch.no_grad():

            Q_targets_next, _ = self.tgt_net(next_states)

            # (batch, num_tau, actions)
            q_t_n = Q_targets_next.mean(dim=1)

            actions = actions.unsqueeze(1)
            rewards = rewards.unsqueeze(1)
            dones = dones.unsqueeze(1)
            weights = weights.unsqueeze(1)

            # calculate log-pi
            logsum = torch.logsumexp(
                (q_t_n - q_t_n.max(1)[0].unsqueeze(-1)) / self.entropy_tau, 1).unsqueeze(-1)  # logsum trick
            # assert logsum.shape == (self.batch_size, 1), "log pi next has wrong shape: {}".format(logsum.shape)
            tau_log_pi_next = (q_t_n - q_t_n.max(1)[0].unsqueeze(-1) - self.entropy_tau * logsum).unsqueeze(1)

            pi_target = F.softmax(q_t_n / self.entropy_tau, dim=1).unsqueeze(1)

            Q_target = (self.gamma ** self.n * (
                    pi_target * (Q_targets_next - tau_log_pi_next) * (~dones.unsqueeze(-1))).sum(2)).unsqueeze(1)

            # assert Q_target.shape == (self.batch_size, 1, self.num_tau)

            q_k_target = q_k_detached.mean(dim=1)  # (batch_size, n_actions)
            v_k_target = q_k_target.max(1)[0].unsqueeze(-1)
            tau_log_pik = q_k_target - v_k_target - self.entropy_tau * torch.logsumexp(
                (q_k_target - v_k_target) / self.entropy_tau, 1).unsqueeze(-1)

            # assert tau_log_pik.shape == (self.batch_size, self.n_actions), "shape instead is {}".format(
            # tau_log_pik.shape)
            munchausen_addon = tau_log_pik.gather(1, actions)

            # calc munchausen reward:
            munchausen_reward = (
                    rewards + self.alpha * torch.clamp(munchausen_addon, min=self.lo, max=0)).unsqueeze(-1)
            # assert munchausen_reward.shape == (self.batch_size, 1, 1)
            # Compute Q targets for current states
            Q_targets = munchausen_reward + Q_target

        # Get expected Q values from local model
        Q_expected = q_k.gather(2, actions.unsqueeze(-1).expand(self.batch_size, self.num_tau, 1))
        # assert Q_expected.shape == (self.batch_size, self.num_tau, 1)

        # Quantile Huber loss
        td_error = Q_targets - Q_expected
        loss_v = torch.abs(td_error).sum(dim=1).mean(dim=1).detach()
        # assert td_error.shape == (self.batch_size, self.num_tau, self.num_tau), "wrong td error shape"
        huber_l = calculate_huber_loss(td_error, 1.0, self.num_tau)
        quantil_l = abs(taus - (td_error.detach() < 0).float()) * huber_l / 1.0

        loss = quantil_l.sum(dim=1).mean(dim=1, keepdim=True)  # , keepdim=True if per weights get multipl

        # PER weights
        loss = loss * weights.to(self.net.device)

        loss = loss.mean()

        # update PER prios
        self.memory.update_priorities(idxs, loss_v.cpu().detach().numpy())

        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip)
        self.optimizer.step()

        self.grad_steps += 1
        if self.grad_steps % 10000 == 0:
            print("Completed " + str(self.grad_steps) + " gradient steps")


def calculate_huber_loss(td_errors, k=1.0, taus=8):
    """
    Calculate huber loss element-wisely depending on kappa k.
    """
    loss = torch.where(td_errors.abs() <= k, 0.5 * td_errors.pow(2), k * (td_errors.abs() - 0.5 * k))
    assert loss.shape == (td_errors.shape[0], taus, taus), "huber loss has wrong shape"
    return loss


def huber_loss(td_errors, k=1.0):
    """
    Calculate huber loss element-wisely depending on kappa k.
    """
    loss = torch.where(td_errors.abs() <= k, 0.5 * td_errors.pow(2), k * (td_errors.abs() - 0.5 * k))
    return loss


def non_default_args(args, parser):
    result = []
    for arg in vars(args):
        user_val = getattr(args, arg)
        default_val = parser.get_default(arg)
        if user_val != default_val and default_val != "NameThisGame" and arg != "include_evals" and arg != "eval_envs"\
                and arg != "num_eval_episodes" and arg != "analy":

            result.append(f"{arg}={user_val}")
    return ', '.join(result)


def format_arguments(arg_string):
    arg_string = arg_string.replace('=', '')
    arg_string = arg_string.replace('True', '1')
    arg_string = arg_string.replace('False', '0')
    arg_string = arg_string.replace(', ', '_')
    return arg_string


def main():
    parser = argparse.ArgumentParser()

    # environment setup
    parser.add_argument('--game', type=str, default="MarioKart")

    parser.add_argument('--envs', type=int, default=4)
    parser.add_argument('--frames', type=int, default=2000000000)
    parser.add_argument('--eval_envs', type=int, default=10)

    parser.add_argument('--bs', type=int, default=256)

    parser.add_argument('--repeat', type=int, default=0)

    parser.add_argument('--framestack', type=int, default=4)

    # agent setup
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--nstep', type=int, default=3)
    parser.add_argument('--maxpool_size', type=int, default=6)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--testing', type=bool, default=False)
    parser.add_argument('--munch_alpha', type=float, default=0.9)
    parser.add_argument('--grad_clip', type=int, default=10)

    parser.add_argument('--spectral', type=int, default=1)
    parser.add_argument('--discount', type=float, default=0.997)
    parser.add_argument('--taus', type=int, default=8)
    parser.add_argument('--c', type=int, default=500)
    parser.add_argument('--linear_size', type=int, default=512)
    parser.add_argument('--model_size', type=float, default=2)

    parser.add_argument('--ncos', type=int, default=64)
    parser.add_argument('--per_alpha', type=float, default=0.2)
    parser.add_argument('--per_beta_anneal', type=int, default=0)
    parser.add_argument('--layer_norm', type=int, default=0)
    parser.add_argument('--eps_steps', type=int, default=2000000)
    parser.add_argument('--eps_disable', type=int, default=1)

    args = parser.parse_args()

    arg_string = non_default_args(args, parser)
    formatted_string = format_arguments(arg_string)
    print(formatted_string)


    game = args.game
    envs = args.envs
    bs = args.bs
    c = args.c
    lr = args.lr
    framestack = args.framestack
    device_name = args.device
    nstep = args.nstep
    maxpool_size = args.maxpool_size
    munch_alpha = args.munch_alpha
    grad_clip = args.grad_clip
    spectral = args.spectral
    discount = args.discount
    linear_size = args.linear_size
    taus = args.taus
    model_size = args.model_size
    frames = args.frames // 4
    ncos = args.ncos
    per_alpha = args.per_alpha
    eps_steps = args.eps_steps
    eps_disable = args.eps_disable
    layer_norm = args.layer_norm

    frame_name = str(int(args.frames / 1000000)) + "M"

    agent_name = "BTR_" + game + frame_name

    replay_period = 64 / envs
    spi = 1

    if len(formatted_string) > 2:
        agent_name += '_' + formatted_string

    print("Agent Name:" + str(agent_name))
    testing = args.testing

    # creates new directory for results and models
    if not testing:
        counter = 0
        while True:
            if counter == 0:
                new_dir_name = agent_name
            else:
                new_dir_name = f"{agent_name}_{counter}"
            if not os.path.exists(new_dir_name):
                break
            counter += 1
        os.mkdir(new_dir_name)
        print(f"Created directory: {new_dir_name}")
        os.chdir(new_dir_name)

    if testing:
        # goes easy on the PC when debugging
        envs = 2
        num_envs = 2
        eval_every = 11580000
        n_steps = 11560000
        bs = 32
    else:
        num_envs = envs
        n_steps = frames
        eval_every = 250000
    next_eval = eval_every

    print("Currently Playing Game: " + str(game))

    if(device_name == None):
        gpu = "0"
        device = torch.device('cuda:' + gpu if torch.cuda.is_available() else 'cpu')
        print(f"\nDevice: {device}. If this does not say (cuda), you should be worried."
            f" Running this on CPU is extremely slow\n")
    else:
        device = torch.device(device_name)
        print(f"\nDevice: {device}")
        if device.type == 'mps':
            _patch_spectral_norm_for_mps()
            print("Applied MPS spectral norm patch (vdot workaround)")

    env = DolphinEnv(envs)
    print(env.observation_space)
    print(env.action_space[0])

    agent = Agent(n_actions=env.action_space[0].n, input_dims=[framestack, 75, 140], device=device, num_envs=num_envs,
                  agent_name=agent_name, total_frames=n_steps, testing=testing, batch_size=bs, lr=lr,
                  maxpool_size=maxpool_size, target_replace=c, spectral=spectral, discount=discount, taus=taus,
                  model_size=model_size, linear_size=linear_size, ncos=ncos, replay_period=replay_period,
                  framestack=framestack, per_alpha=per_alpha, layer_norm=layer_norm,
                  eps_steps=eps_steps, eps_disable=eps_disable, n=nstep,
                  munch_alpha=munch_alpha, grad_clip=grad_clip, imagex=140, imagey=75, spi=spi)

    scores_temp = []
    steps = 0
    last_steps = 0
    last_time = time.time()
    episodes = 0
    current_eval = 0
    scores_count = [0 for _ in range(num_envs)]
    scores = []
    observation, info = env.reset()
    processes = []

    if testing:
        from torchsummary import summary
        summary(agent.net, (framestack, 75, 140))

    while steps < n_steps:
        steps += num_envs
        try:
            action = agent.choose_action(observation)
        except Exception as e:
            print(f"Error: {e}")
            print(f"Observation: {observation}")
            raise Exception("Stop! Error Occurred")

        env.step_async(action)
        agent.learn()
        observation_, reward, done_, trun_, info = env.step_wait()

        for i in range(num_envs):
            scores_count[i] += reward[i]
            if done_[i] or trun_[i]:
                episodes += 1
                scores.append([scores_count[i], steps])
                scores_temp.append(scores_count[i])
                scores_count[i] = 0

        # no clipping, be careful with using large rewards!
        #reward = np.clip(reward, -1., 1.)

        for stream in range(num_envs):

            if info["Ignore"][stream]:
                continue

            if info["First"][stream]:
                observation[stream] = observation_[stream]

            next_obs = observation_[stream] if not trun_[stream] else np.array(info["final_observation"][stream])
            agent.store_transition(observation[stream], action[stream], reward[stream], next_obs,
                                   done_[stream], trun_[stream], stream=stream)

        observation = observation_

        if steps % 600 == 0 and len(scores) > 0:
            avg_score = np.mean(scores_temp[-50:])
            if episodes % 1 == 0:
                print('{} avg score {:.2f} total_timesteps {:.0f} fps {:.2f} games {}'
                      .format(agent_name, avg_score, steps,
                              (steps - last_steps) / (time.time() - last_time), episodes), flush=True)
                last_steps = steps
                last_time = time.time()

        # Evaluation
        if steps >= next_eval or steps >= n_steps:

            # save all models here
            print("Saving Model...")
            agent.save_model()

            fname = agent_name + "Experiment.npy"
            if not testing:
                np.save(fname, np.array(scores))

            # Parameters
            window = 200  # Smoothing window size

            # Extract scores and steps
            episode_scores = np.array([s[0] for s in scores])
            episode_steps = np.array([s[1] for s in scores])

            if len(scores) < window:
                print(f"Not enough episodes for a window size of {window}, reducing to {len(scores)}")
                window = len(scores)

            # Compute moving average
            cumsum = np.cumsum(np.insert(episode_scores, 0, 0))
            moving_avg = (cumsum[window:] - cumsum[:-window]) / window

            # X axis: use the 'steps' from the center of each window
            avg_steps = episode_steps[window - 1:]

            # Plot
            plt.figure(figsize=(8, 5))
            plt.plot(avg_steps, moving_avg, label=f'{window}-episode moving average')
            plt.xlabel('Steps')
            plt.ylabel('Score (smoothed)')
            plt.title('Smoothed Episode Scores Over Time')
            plt.grid(True)
            plt.tight_layout()
            plt.legend()

            plt.savefig('scores_over_time_smoothed.png')
            plt.close()

            current_eval += 1

            next_eval += eval_every

    # wait for our evaluations to finish before we quit the program
    for process in processes:
        process.join()

    print("Evaluations finished, job completed successfully!")


if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()