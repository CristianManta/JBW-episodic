import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import os.path as osp

class CNN(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(in_channels=4, out_channels=6, kernel_size=3)
    self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
    self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3)
    self.fc1 = nn.Linear(in_features=64, out_features=32)
    self.fc_policy = nn.Linear(in_features=32, out_features=4)
    self.fc_value = nn.Linear(in_features=32, out_features=1)

  def forward(self, x):
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = torch.flatten(x,1)
    x = F.relu(self.fc1(x))
    policy = F.softmax(self.fc_policy(x).flatten(), dim=-1)
    value = self.fc_value(x)
    return policy, value

class Agent():
  '''The agent class that is to be filled.
     You are allowed to add any method you
     want to this class.
  '''

  def __init__(self, env_specs, model_str='cnn'):
    self.env_specs = env_specs
    self.lr = 0.001
    self.gamma = 0.9
    self.eps = 0.1
    self.num_actions = 4
    self.n = 20 #Number of timesteps in loss and gradient computation
    self.c = 1

    self.values = torch.zeros(self.n)
    self.rewards = torch.zeros(self.n)
    self.policies = torch.zeros(self.n)

    if model_str == 'cnn':
      self.model = CNN()
      self.encode_features = self.encode_features_grid

    self.model.train()

    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

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
    curr_obs = torch.from_numpy(curr_obs[2])
    return curr_obs.reshape((15, 15, 4)).transpose(0, 2).unsqueeze(0).float()

  def act(self, curr_obs, mode='eval'):
    rand_action = np.random.binomial(1, self.eps)
    if rand_action:
      return self.env_specs['action_space'].sample()

    feats = self.encode_features(curr_obs)
    policy, value = self.model(feats)
    action_sampler = Categorical(policy)
    return action_sampler.sample()


  def update(self, curr_obs, action, reward, next_obs, done, timestep):
    print(timestep)
    torch.autograd.set_detect_anomaly(True)
    t = timestep % self.n
    
    curr_feats = self.encode_features(curr_obs)
    policy, self.values[t] = self.model(curr_feats)
    self.policies[t] = policy[action]
    self.rewards[t] = reward * self.gamma**t

    if t == self.n - 1:
      advantages = torch.zeros(self.n)
      rets = torch.zeros(self.n)
      next_feats = self.encode_features(next_obs)
      _, final_value = self.model(next_feats)
      for i in range(self.n):
        rets[i] = torch.sum(self.rewards[i:]/(self.gamma**i)) + self.gamma**(self.n - i) * final_value
      advantages = rets - self.values

      self.optimizer.zero_grad()
      for param in self.model.fc_value.parameters():
        param.requires_grad = False
      policy_loss = torch.sum(-torch.log(self.policies) * advantages) 
      policy_loss.backward(retain_graph=True)
      for param in self.model.fc_value.parameters():
        param.requires_grad = True
      value_loss = torch.sum(advantages)**2
      value_loss.backward(retain_graph=True)
      self.optimizer.step()
