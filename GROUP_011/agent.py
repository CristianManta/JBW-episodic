import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import os.path as osp
from copy import deepcopy

class DQN(nn.Module): # TODO: See if can process an entire batch in one pass
  def __init__(self):
    super().__init__()
    scent_out_features = 4
    self.conv1 = nn.Conv2d(in_channels=4, out_channels=6, kernel_size=3)
    self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
    self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3)
    self.fc1 = nn.Linear(in_features=64, out_features=32)
    self.scent_fc = nn.Linear(in_features=3, out_features=scent_out_features)
    self.fc2 = nn.Linear(in_features=32 + scent_out_features, out_features=4)    

  def forward(self, inputs):
    scent = inputs[0]
    grid = inputs[1]

    scent = self.scent_fc(scent)
    grid = self.pool(F.relu(self.conv1(grid)))
    grid = self.pool(F.relu(self.conv2(grid)))
    grid = torch.flatten(grid,1)
    grid = F.relu(self.fc1(grid))

    combined = torch.cat((grid, scent), dim=1)
    grid = self.fc2(combined)
    return torch.flatten(grid)

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
        sample = []
        for _ in range(batch_size):
          index = np.random.randint(low=0, high=self.size)
          sample.append(self.buffer[index])
        return sample

    def __len__(self):
        return self.size

def make_target_model(model):
    target_model = deepcopy(model)
    for param in target_model.parameters():
        param.requires_grad = False

    return target_model

class Agent():
  '''The agent class that is to be filled.
     You are allowed to add any method you
     want to this class.
  '''

  def __init__(self, env_specs):
    self.env_specs = env_specs
    self.encode_features = self.encode_features_grid
    self.lr = 0.00025
    self.gamma = 0.9
    self.eps = 1
    self.final_eps = 0.1
    self.eval_eps = 0.05
    self.eps_anneal_steps = 1e+5 #Timespan over which to decay epsilon
    self.buffer_capacity = 10000
    self.buffer = ReplayBuffer(self.buffer_capacity)
    self.target_update_freq = 1000
    self.batch_size = 32
    self.num_actions = 4

    self.model = DQN()
    self.target_model = make_target_model(self.model)
    self.model.train()

    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
    self.criterion = nn.MSELoss()

  def load_weights(self, root_path="./"):
    # Add root_path in front of the path of the saved network parameters
    # For example if you have weights.pth in the GROUP_MJ1, do `root_path+"weights.pth"` while loading the parameters    
    full_path = root_path + "weights.pth"
    if osp.exists(full_path):
      self.model.load_state_dict(torch.load(full_path))
    

  def save_weights(self, root_path="./"):
    full_path = root_path + "weights.pth"
    torch.save(self.model.state_dict(), full_path)

  def encode_features_grid(self, curr_obs):
    scent = torch.from_numpy(curr_obs[0])
    grid = torch.from_numpy(curr_obs[2])
    return (scent.unsqueeze(0).float(), grid.reshape((15, 15, 4)).transpose(0, 2).unsqueeze(0).float())

  def act(self, curr_obs, mode='eval'):
    if mode == 'train':
      eps = self.eps
    elif mode == 'eval':
      eps = self.eval_eps

    rand_action = np.random.binomial(1, eps)
    if rand_action:
      return self.env_specs['action_space'].sample()

    feats = self.encode_features(curr_obs)
    q = self.model(feats)
    return torch.argmax(q)


  def update(self, curr_obs, action, reward, next_obs, done, timestep):
    self.optimizer.zero_grad()
    if done:
        next_obs = None
    #Add observation to replay buffer
    self.buffer.add(curr_obs, action, reward, next_obs)

    if (timestep % self.target_update_freq) == 0:
        #Update target network
        self.target_model = make_target_model(self.model)

    if timestep < self.buffer_capacity:
        #No learning until buffer is full
        return
    elif timestep <= self.eps_anneal_steps:
        #Annealing epsilon
        self.eps = 1.1 - 0.9/(self.eps_anneal_steps - self.buffer_capacity) * timestep

    #Sample a batch
    batch = self.buffer.sample(self.batch_size)
    estimates = torch.zeros(self.batch_size)
    targets = torch.zeros(self.batch_size)
    for i, (curr_obs, action, reward, next_obs) in enumerate(batch):
        curr_feats = self.encode_features(curr_obs)
        cur_q = self.model(curr_feats)[action]
        estimates[i] = cur_q
        if next_obs is None:
            targets[i] = torch.as_tensor(reward)
        else:
            next_feats = self.encode_features(next_obs)
            next_q = torch.max(self.target_model(next_feats))
            targets[i] = reward + self.gamma * next_q

    loss = self.criterion(estimates, targets)
    loss.backward()
    self.optimizer.step()