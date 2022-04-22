import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import os.path as osp

class CNN(nn.Module):
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
    self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
    self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3)
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
    x = self.fc2(x)
    return torch.flatten(x)

class LinearModel(nn.Module):
  """Linear function approximator used to approximate the action value function.
  
  Attributes:
    w (nn.Module): Linear layer.
  """
  def __init__(self, input_size):
    super().__init__()
    self.w = nn.Linear(in_features=input_size, out_features=4)

  def forward(self, x):
    """Forward pass of the network.
    
    Args:
      x: Input to the network.

    Returns:
      The estimated value function, i.e., a tensor with four entries (one for each action).
    """
    return torch.flatten(self.w(x))

class Agent():
  '''Sarsa agent.

  Attributes:
    env_specs (dictionary): Information about the environment.
    lr (float): Learning rate.
    gamma (float): Discount factor
    eps (float): Parameter for epsilon-greedy action selection.
    num_actions (int): Number of actions that can be taken.
    model (nn.Module): The function approximator being used.
    encode_features (function): Specifies how to construct state representations.
    optimizer (torch.optim object): The optimizer used during training.
    criterion (nn loss object): The loss being used.

  Args:
    env_specs (dictionary): Information about the environment.
    model_str (str): String indicated which function approximator will be used (cnn or linear).
  '''

  def __init__(self, env_specs, model_str='cnn'):
    self.env_specs = env_specs
    self.lr = 0.001
    self.gamma = 0.9
    self.eps = 0.1
    self.num_actions = 4

    if model_str == 'cnn':
      self.model = CNN()
      self.encode_features = self.encode_features_grid
    elif model_str == 'linear':
      self.input_size = 900
      self.model = LinearModel(self.input_size)
      self.encode_features = self.encode_features_sparse

    self.model.train()

    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
    self.criterion = nn.MSELoss()

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

  def encode_features_sparse(self, curr_obs):
    """Sparse feature encoding (i.e. flattening the grid into a vector) for the linear model.

    Args:
      curr_obs (list): Current observation from the environment.

    Returns:
      A flattened tensor corresponding to the data in the grid.
    """
    feats = torch.tensor(curr_obs[2]).flatten()

    return feats.float()

  def encode_features_grid(self, curr_obs):
    """Grid encoding of the observation for the CNN model.
    
    Args:
      curr_obs (list): Current observation from the environment.

    Returns:
      A tensor ready to be fed into the CNN.
    """
    curr_obs = torch.from_numpy(curr_obs[2])
    return curr_obs.reshape((15, 15, 4)).transpose(0, 2).unsqueeze(0).float()

  def act(self, curr_obs, mode='eval'):
    """Given an observation from the environment, select an action.

    Args:
      curr_obs (list): Observation from the environment.
      mode (str): Indicates whether we are in training or evaluation mode.
    """
    #Select random action with probability eps
    rand_action = np.random.binomial(1, self.eps)
    if rand_action:
      return self.env_specs['action_space'].sample()

    #Otherwise, take the greedy action
    feats = self.encode_features(curr_obs)
    q = self.model(feats)
    return torch.argmax(q)


  def update(self, curr_obs, action, reward, next_obs, done, timestep):
    """Given a transition, perform a Sarsa update.
    
    Args:
      curr_obs (list): The current observation from the environment.
      action (int): The action taken.
      reward (float): The reward received.
      next_obs (list): The next observation from the environment.
      done (bool): Whether or not we have reached the end of the episode.
      timestep (int): The current timestep.
    """
    self.optimizer.zero_grad()
    curr_feats = self.encode_features(curr_obs)
    cur_q = self.model(curr_feats)[action]
    if done:
      reward = torch.as_tensor(reward)
      loss = self.criterion(cur_q, reward)
    else:
      with torch.no_grad():
        next_feats = self.encode_features(next_obs)
        next_action = self.act(next_obs)
        next_q = self.model(next_feats)[next_action]
      loss = self.criterion(cur_q, reward + self.gamma * next_q)

    loss.backward()
    self.optimizer.step()