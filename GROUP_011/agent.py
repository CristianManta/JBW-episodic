import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import os
import os.path as osp

class LeNet(nn.Module):
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

  def __init__(self, env_specs, model_str='lenet', encoding_method='grid'):
    self.env_specs = env_specs
    self.alpha = 0.001
    self.gamma = 0.9
    self.eps = 0.25
    self.num_actions = 4

    if encoding_method == 'dense':      
      self.input_size = 15 * 15      
      self.encode_features = self.encode_features_dense
      self.action_position_embeds = [(7, 8), (6, 7), (8, 7), (7, 6)]
    elif encoding_method == 'sparse':      
      self.input_size = 900      
      self.encode_features = self.encode_features_sparse
    elif encoding_method == 'grid':
      self.encode_features = self.encode_features_grid

    if model_str == 'lenet':
      self.model = LeNet()
    elif model_str == 'linear':
      self.model = LinearModel(self.input_size)

    self.model.train()

    self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.alpha, momentum=0.9, weight_decay=5e-4)
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

  
  def encode_features_dense(self, curr_obs):
    """
    In curr_obs[2].reshape((15, 15, 4)), the last dimension encodes the following:
    Jelly bean -> [0, 0, 1, 0]
    Truffle -> [0, 0, 0, 1]
    Banana -> [0, 1, 0, 0]
    Apple -> [1, 0, 0, 0]

    In our encoding, we give them, respectively, the values below:
    """    
    jb_val = -2
    truffle_val = -1.1
    banana_val = 1
    apple_val = 2
    empty_val = 0 # This might be too naive

    feats = curr_obs[2].reshape((15, 15, 4))
    feats = np.concatenate((feats, np.zeros((15, 15, 1), dtype=int)), axis=2) # Need to add fifth indicator with a "1" iff there's no object there
    for i in range(15): # If you see a more compact (and fast) way to do this operation, please modify it or let me know
      for j in range(15):
        if sum(feats[i][j]) == 0:
          feats[i][j][-1] = 1    

    feats = feats.astype(bool)
    values = np.broadcast_to(np.array([apple_val, banana_val, jb_val, truffle_val, empty_val]), (15, 15, 5))
    feats = values[feats].reshape((15, 15))
    feats = torch.from_numpy(feats.flatten())
    return feats.float()

  def encode_features_sparse(self, curr_obs):
    feats = torch.from_numpy(curr_obs[2]).flatten()
    #feats = torch.concatenate((curr_obs[0].flatten(), curr_obs[1].flatten(), curr_obs[2].flatten()))

    return feats.float()

  def encode_features_grid(self, curr_obs):
    curr_obs = torch.from_numpy(curr_obs[2])
    return curr_obs.reshape((15, 15, 4)).transpose(0, 2).unsqueeze(0).float()

  def act(self, curr_obs, mode='eval'):
    if mode == 'train':
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
      next_action = self.act(next_obs, mode='train')
      next_q = self.model(next_feats)[next_action]
      loss = self.criterion(cur_q, reward + self.gamma * next_q)

    loss.backward()
    self.optimizer.step()
