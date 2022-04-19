import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import os.path as osp

class CNN(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(in_channels=4, out_channels=6, kernel_size=3)
    self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
    self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3)
    self.fc1 = nn.Linear(in_features=64, out_features=32)
    self.fc2 = nn.Linear(in_features=32, out_features=4)

  def forward(self, x):
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = torch.flatten(x,1)
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return torch.flatten(x)

class LinearModel(nn.Module):
  def __init__(self, input_size):
    super().__init__()
    self.w = nn.Linear(in_features=input_size, out_features=4)

  def forward(self, x):
    return torch.flatten(self.w(x))

class Agent():
  '''The agent class that is to be filled.
     You are allowed to add any method you
     want to this class.
  '''

  def __init__(self, env_specs, model_str='linear'):
    self.env_specs = env_specs
    self.lr = 0.001
    self.gamma = 0.9
    self.eps = 0.1
    self.num_actions = 4

    if model_str == 'cnn':
      self.model = CNN()
      self.encode_features = self.encode_features_grid
    elif model_str == 'linear':
      self.model = LinearModel(self.input_size)
      self.encode_features = self.encode_features_sparse

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

  def encode_features_sparse(self, curr_obs):
    feats = torch(curr_obs[2]).flatten()

    return feats.float()

  def encode_features_grid(self, curr_obs):
    curr_obs = torch.from_numpy(curr_obs[2])
    return curr_obs.reshape((15, 15, 4)).transpose(0, 2).unsqueeze(0).float()

  def act(self, curr_obs, mode='eval'):
    rand_action = np.random.binomial(1, self.eps)
    if rand_action:
      return self.env_specs['action_space'].sample()

    feats = self.encode_features(curr_obs)
    q = self.model(feats)
    return torch.argmax(q)


  def update(self, curr_obs, action, reward, next_obs, done, timestep):
    self.optimizer.zero_grad()
    curr_feats = self.encode_features(curr_obs)
    cur_q = self.model(curr_feats)[action]
    if done:
      reward = torch.as_tensor(reward)
      loss = self.criterion(cur_q, reward)
    else:
      next_feats = self.encode_features(next_obs)
      next_q = torch.max(self.model(next_feats))
      loss = self.criterion(cur_q, reward + self.gamma * next_q)

    loss.backward()
    self.optimizer.step()
