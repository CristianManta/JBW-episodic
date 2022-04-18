from ssl import ALERT_DESCRIPTION_BAD_CERTIFICATE_HASH_VALUE
import numpy as np
import random
import torch
from torch import nn
import torch.nn.functional as F
import os.path as osp
from copy import deepcopy

class DQN(nn.Module): # TODO: See if can process an entire batch in one pass
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(in_channels=4, out_channels=6, kernel_size=3)
    self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3)
    self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
    self.fc1 = nn.Linear(in_features=64, out_features=32)
    self.fc2 = nn.Linear(in_features=32, out_features=4)

  def forward(self, x):
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = torch.flatten(x,1)
    x = F.relu(self.fc1(x))
    out = self.fc2(x)
    return out

class State():
  def __init__(self, obs):
    self.vision = torch.tensor(obs[0])
    self.scent = torch.tensor(obs[1])
    self.features = torch.tensor(obs[2])

  def get_vision(self):
    return self.vision

  def get_scent(self):
    return self.scent

  def get_features(self):
    return self.features

  def encode(self):
    return self.features.reshape((15, 15, 4)).transpose(0, 2).unsqueeze(0).float()

#Inspired by PyTorch tutorial: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
class ReplayBuffer(object):

    def __init__(self, capacity):
        self.buffer = []
        self.size = 0
        self.capacity = capacity

    def add(self, curr_obs, action, reward, next_obs):
        """Save a transition"""
        self.buffer.append([curr_obs, action, reward, next_obs])
        if self.size == self.capacity:
            self.buffer.pop(0)
        else:
            self.size += 1

    def sample(self, batch_size):
        sample = random.sample(self.buffer, batch_size)
        curr_obs_batch = torch.cat([a[0].encode() for a in sample])
        action_batch = torch.stack([torch.tensor(a[1]) for a in sample])
        reward_batch = torch.stack([torch.tensor(a[2]) for a in sample])
        next_obs_batch = torch.cat([a[3].encode() for a in sample])

        return curr_obs_batch, action_batch, reward_batch, next_obs_batch

    def __len__(self):
        return self.size

class Agent():
  '''The agent class that is to be filled.
     You are allowed to add any method you
     want to this class.
  '''

  def __init__(self, env_specs):
    self.env_specs = env_specs
    self.lr = 0.00025
    self.gamma = 0.9
    self.initial_eps = 1
    self.eps = self.initial_eps
    self.final_eps = 0.1
    self.eval_eps = 0.05
    self.eps_anneal_steps = 1e+5 #Timespan over which to decay epsilon
    self.buffer_capacity = 10000
    self.buffer = ReplayBuffer(self.buffer_capacity)
    self.target_update_freq = 1000
    self.batch_size = 32
    self.num_actions = 4

    self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    self.model = DQN()
    self.model.to(self.device)
    self.model.train()

    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
    self.criterion = nn.MSELoss()

  def make_target_model(self):
    self.target_model = deepcopy(self.model)
    for param in self.target_model.parameters():
        param.requires_grad = False
    self.target_model.to(self.device)

  def load_weights(self, root_path="./"):
    # Add root_path in front of the path of the saved network parameters
    # For example if you have weights.pth in the GROUP_MJ1, do `root_path+"weights.pth"` while loading the parameters    
    full_path = root_path + "weights.pth"
    if osp.exists(full_path):
      self.model.load_state_dict(torch.load(full_path))

  def save_weights(self, root_path="./"):
    full_path = root_path + "weights.pth"
    torch.save(self.model.state_dict(), full_path)

  def act(self, curr_obs, mode='eval'):
    if mode == 'train':
      eps = self.eps
    elif mode == 'eval':
      eps = self.eval_eps

    rand_action = np.random.binomial(1, eps)
    if rand_action:
      return self.env_specs['action_space'].sample()

    feats = State(curr_obs).encode().to(self.device)
    q = self.model(feats)
    return torch.argmax(q)

  def update(self, curr_obs, action, reward, next_obs, done, timestep):
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

    actions = actions.to(self.device)
    rewards = rewards.to(self.device)
    curr_obs = curr_obs.to(self.device)
    preds = self.model(curr_obs)
    estimates = preds.gather(1, actions.view(-1,1)).flatten()
    next_obs = next_obs.to(self.device)
    targets = rewards + self.gamma * torch.max(self.model(next_obs), dim=1)[0]

    loss = self.criterion(estimates, targets)
    loss.backward()
    self.optimizer.step()