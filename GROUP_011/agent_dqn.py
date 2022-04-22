import numpy as np
import random
import torch
from torch import nn
import torch.nn.functional as F
import os.path as osp
from copy import deepcopy

class DQN(nn.Module):
  """Our CNN architecture used to approximate the action value function.

  Attributes:
    self.conv1 (nn.Module): First convolutional layer.
    self.conv2 (nn.Module): Second convolutional layer.
    self.pool (nn.Module): Pooling layer.
    self.fc1 (nn.Module): First fully-connected layer.
    self.fc2 (nn.Module): Second fully-connected layer.
  """
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(in_channels=4, out_channels=6, kernel_size=3)
    self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3)
    self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
    self.fc1 = nn.Linear(in_features=64, out_features=32)
    self.fc2 = nn.Linear(in_features=32, out_features=4)

  def forward(self, x):
    """Forward pass of the network.

    Args:
      x (torch.tensor): Input to network.

    Returns:
      The estimated value function, i.e., a tensor with four entries (one for each action).
    """
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = torch.flatten(x,1)
    x = F.relu(self.fc1(x))
    out = self.fc2(x)
    return out

class State():
  """Wrapper for a state. Helpful for the replay buffer.

  Attributes:
    scent (torch.tensor): Scent input.
    vision (torch.tensor): Vision input.
    features (torch.tensor): TA-provided features.

  Args:
    obs (list): Observation from the environment.
  """
  def __init__(self, obs):
    self.scent = torch.tensor(obs[0])
    self.vision = torch.tensor(obs[1])
    self.features = torch.tensor(obs[2])

  def get_scent(self):
    return self.scent

  def get_vision(self):
    return self.vision

  def get_features(self):
    return self.features

  def encode(self):
    """Encode features to be fed into CNN."""
    return self.features.reshape((15, 15, 4)).transpose(0, 2).unsqueeze(0).float()

class ReplayBuffer(object):
  """The replay buffer. Inspired by PyTorch tutorial: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

  Attributes:
    buffer (list): The replay buffer/memory.
    size (int): Number of entries in the buffer.
    capacity (int): Capacity of the buffer.

  Args:
    capacity (int): Capacity of the replay buffer.
  """

  def __init__(self, capacity):
    self.buffer = []
    self.size = 0
    self.capacity = capacity

  def add(self, curr_obs, action, reward, next_obs):
    """Save a transition in the buffer and remove the oldest one if no space.
    
    Args:
      curr_obs (list): Current observation.
      action (int): Action taken.
      reward (int): Reward received.
      next_obs (list): Next observation.
    """
    self.buffer.append([curr_obs, action, reward, next_obs])
    if self.size == self.capacity:
        self.buffer.pop(0)
    else:
        self.size += 1

  def sample(self, batch_size):
    """Sample a batch of transitions from the buffer.

    Args:
      batch_size (int): The size of the batch to return.

    Returns:
      A batch of transitions (current observation, action, reward, next observation).
    
    """
    sample = random.sample(self.buffer, batch_size)
    curr_obs_batch = torch.cat([a[0].encode() for a in sample])
    action_batch = torch.stack([torch.tensor(a[1]) for a in sample])
    reward_batch = torch.stack([torch.tensor(a[2]) for a in sample])
    next_obs_batch = torch.cat([a[3].encode() for a in sample])

    return curr_obs_batch, action_batch, reward_batch, next_obs_batch

  def __len__(self):
    return self.size

class Agent():
  '''DQN agent.

  Attributes:
    env_specs (dictionary): Information about the environment.
    lr (float): Learning rate.
    gamma (float): Discount factor.
    initial_eps (float): Starting value for epsilon.
    eps (float): Parameter for epsilon-greedy action selection.
    final_eps (float): Final value for epsilon (after annealing).
    eval_eps (float): Value of epsilon to use at evaluation time.
    eps_anneal_steps (int): Timespan over which to decay epsilon.
    buffer_capacity (int): Capacity of the buffer.
    buffer (ReplayBuffer): The replay buffer itself.
    target_update_freq: Frequency at which target network is updated.
    batch_size: Batch size for replay buffer sampling.
    num_actions (int): Number of actions that can be taken.
    model (nn.Module): The function approximator being used.
    optimizer (torch.optim object): The optimizer used during training.
    criterion (nn loss object): The loss being used.

  Args:
    env_specs (dictionary): Information about the environment.
  '''

  def __init__(self, env_specs):
    self.env_specs = env_specs
    self.lr = 0.00025
    self.gamma = 0.9
    self.initial_eps = 1
    self.eps = self.initial_eps
    self.final_eps = 0.1
    self.eval_eps = 0.05
    self.eps_anneal_steps = 1e+4
    self.buffer_capacity = 1000
    self.buffer = ReplayBuffer(self.buffer_capacity)
    self.target_update_freq = 1000
    self.batch_size = 16
    self.num_actions = 4

    self.model = DQN()
    self.model.train()

    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
    self.criterion = nn.MSELoss()

  def make_target_model(self):
    """Making the target model as a copy of the model."""
    self.target_model = deepcopy(self.model)
    #The target model should not contribute to the gradient
    for param in self.target_model.parameters():
        param.requires_grad = False

  def load_weights(self, root_path="./"):
    """Loading the weights of the function approximator.

    Args:
      root_path (str): Specification of path to weights
    """
    # Add root_path in front of the path of the saved network parameters
    # For example if you have weights.pth in the GROUP_MJ1, do `root_path+"weights.pth"` while loading the parameters    
    full_path = root_path + "weights.pth"
    if osp.exists(full_path):
      self.model.load_state_dict(torch.load(full_path))

  def save_weights(self, root_path="./"):
    """Saving the weights of the function approximator.
    
    Args:
      root_path (str): Destination path for saved weights.
    """
    full_path = root_path + "weights.pth"
    torch.save(self.model.state_dict(), full_path)

  def act(self, curr_obs, mode='eval'):
    """Given an observation from the environment, select an action.

    Args:
      curr_obs (list): Observation from the environment.
      mode (str): Indicates whether we are in training or evaluation mode.
    """
    if mode == 'train':
      eps = self.eps
    elif mode == 'eval':
      eps = self.eval_eps

    #Take random action with probability epsilon
    rand_action = np.random.binomial(1, eps)
    if rand_action:
      return self.env_specs['action_space'].sample()

    #Otherwise, take the greedy action
    feats = State(curr_obs).encode()
    q = self.model(feats)
    return torch.argmax(q)

  def update(self, curr_obs, action, reward, next_obs, done, timestep):
    """Given a transition, perform a Q-learning update.
    
    Args:
      curr_obs (list): The current observation from the environment.
      action (int): The action taken.
      reward (float): The reward received.
      next_obs (list): The next observation from the environment.
      done (bool): Whether or not we have reached the end of the episode.
      timestep (int): The current timestep.
    """
    self.optimizer.zero_grad()
    if not done:
      #Add observation to replay buffer
      self.buffer.add(State(curr_obs), action, reward, State(next_obs))

    if (timestep % self.target_update_freq) == 0:
        #Update target network
        self.make_target_model()

    if timestep < self.buffer_capacity:
        #No learning until buffer is full
        return
    elif timestep <= self.eps_anneal_steps:
        #Annealing epsilon
        self.eps = self.initial_eps - (self.initial_eps - self.final_eps)/(self.eps_anneal_steps - self.buffer_capacity) * (timestep - self.buffer_capacity)

    #Sample a batch
    curr_obs, actions, rewards, next_obs = self.buffer.sample(self.batch_size)

    #Compute loss and gradient before performing Q-learning update
    preds = self.model(curr_obs)
    estimates = preds.gather(1, actions.view(-1,1)).flatten()
    targets = rewards + self.gamma * torch.max(self.target_model(next_obs), dim=1)[0]
    loss = self.criterion(estimates, targets)
    loss.backward()
    self.optimizer.step()