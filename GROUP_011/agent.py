import numpy as np
import random
import torch
from torch import nn
import torch.nn.functional as F
import os.path as osp
from copy import deepcopy

class DQN(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(in_channels=4, out_channels=6, kernel_size=3)
    self.pool = nn.AvgPool2d(kernel_size=2, stride=2) # TODO: This may be removed at some point
    self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3)
    self.fc1 = nn.Linear(in_features=64, out_features=32)
    self.fc2 = nn.Linear(in_features=32, out_features=4)

  def forward(self, scent, feats):
    feats = self.pool(F.relu(self.conv1(feats)))
    feats = self.pool(F.relu(self.conv2(feats)))
    feats = torch.flatten(feats,1)
    feats = F.relu(self.fc1(feats))
    combined = torch.cat((feats, scent), dim=1)
    out = self.fc2(combined)
    return out

class State():
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
    return [self.scent.unsqueeze(0), self.features.reshape((15, 15, 4)).transpose(0, 2).unsqueeze(0).float()]

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
        curr_scent_batch = torch.cat([a[0].encode()[0] for a in sample])
        curr_feats_batch = torch.cat([a[0].encode()[1] for a in sample])
        action_batch = torch.stack([torch.tensor(a[1]) for a in sample])
        reward_batch = torch.stack([torch.tensor(a[2]) for a in sample])
        next_scent_batch = torch.cat([a[3].encode()[0] for a in sample])
        next_feats_batch = torch.cat([a[3].encode()[1] for a in sample])

        return curr_scent_batch, curr_feats_batch, action_batch, reward_batch, next_scent_batch, next_feats_batch

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
    self.batch_size = 32
    self.target_update_freq = 1000
    self.num_actions = 4

    self.model1 = DQN()
    self.model2 = DQN()

    self.model1.train()
    self.model2.train()

    self.optimizer = torch.optim.Adam(list(self.model1.parameters()) + list(self.model2.parameters()), lr=self.lr) # TODO: Maybe SGD is better    
    self.criterion = nn.MSELoss()

  def make_target_models(self):
    self.target_model1 = deepcopy(self.model1)
    self.target_model2 = deepcopy(self.model2)
    for param in self.target_model1.parameters():
        param.requires_grad = False
    for param in self.target_model2.parameters():
        param.requires_grad = False


  def load_weights(self, root_path="./"):
    # Add root_path in front of the path of the saved network parameters
    # For example if you have weights.pth in the GROUP_MJ1, do `root_path+"weights.pth"` while loading the parameters    
    full_path1 = root_path + "weights1.pth"
    full_path2 = root_path + "weights2.pth"
    if osp.exists(full_path1) and osp.exists(full_path2):
      self.model1.load_state_dict(torch.load(full_path1))
      self.model2.load_state_dict(torch.load(full_path2))
    

  def save_weights(self, root_path="./"):
    full_path1 = root_path + "weights1.pth"
    full_path2 = root_path + "weights2.pth"
    torch.save(self.model1.state_dict(), full_path1)
    torch.save(self.model2.state_dict(), full_path2)


  def act(self, curr_obs, mode='eval'):
    if mode == 'train':
      eps = self.eps
    elif mode == 'eval':
      eps = self.eval_eps

    rand_action = np.random.binomial(1, eps)
    if rand_action:
      return self.env_specs['action_space'].sample()

    inputs = State(curr_obs).encode()
    q1, q2 = self.model1(inputs[0], inputs[1]), self.model2(inputs[0], inputs[1])
    return torch.argmax(q1 + q2)


  def update(self, curr_obs, action, reward, next_obs, done, timestep):
    self.optimizer.zero_grad()
    if not done:
      #Add observation to replay buffer
      self.buffer.add(State(curr_obs), action, reward, State(next_obs))

    if (timestep % self.target_update_freq) == 0:
        #Update target network
        self.make_target_models()

    if timestep < self.buffer_capacity:
        #No learning until buffer is full
        return
    elif timestep <= self.eps_anneal_steps:
        #Annealing epsilon
        self.eps = self.initial_eps - (self.initial_eps - self.final_eps)/(self.eps_anneal_steps - self.buffer_capacity) * (timestep - self.buffer_capacity)

    #Sample a batch
    curr_scent, curr_feats, actions, rewards, next_scent, next_feats = self.buffer.sample(self.batch_size)
    update_model1 = np.random.binomial(1, 0.5)
    if update_model1:
      online_model = self.model1
      target_model = self.target_model2
    else:
      online_model = self.model2
      target_model = self.target_model1
    
    preds = online_model(curr_scent, curr_feats)
    estimates = preds.gather(1, actions.view(-1, 1)).flatten()
    targets = rewards + self.gamma * torch.max(target_model(next_scent, next_feats), dim=1)[0]

    loss = self.criterion(estimates, targets)
    loss.backward()
    self.optimizer.step()
