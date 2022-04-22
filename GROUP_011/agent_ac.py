import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
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
    self.fc_policy = nn.Linear(in_features=32, out_features=4)
    self.fc_value = nn.Linear(in_features=32, out_features=1)

  def forward(self, x, head):
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
    if head == 'policy':
      x = F.softmax(self.fc_policy(x), dim=-1)
    if head == 'value':
      x = self.fc_value(x)
    return x

class Agent():
  '''A2C agent.

  Attributes:
    env_specs (dictionary): Information about the environment.
    lr (float): Learning rate.
    gamma (float): Discount factor
    eps (float): Parameter for epsilon-greedy action selection.
    num_actions (int): Number of actions that can be taken.
    n (int): Number of steps to consider when computing approximate returns.
    states (torch.tensor): Keeps a log of recently visited states.
    actions (torch.tensor): Keeps a log of recently taken actions.
    rewards (torch.tensor): Keeps a log of recently received rewards.
    model (nn.Module): The function approximator being used.
    encode_features (function): Specifies how to construct state representations.
    optimizer (torch.optim object): The optimizer used during training.
    criterion (nn loss object): The loss being used.

  Args:
    env_specs (dictionary): Information about the environment.
    model_str (str): String indicated which function approximator will be used (cnn or linear).
  '''

  def __init__(self, env_specs):
    self.env_specs = env_specs
    self.lr = 0.001
    self.gamma = 0.9
    self.num_actions = 4
    self.n = 5
    self.states = torch.zeros((self.n, 4, 15, 15))
    self.actions = torch.zeros(self.n, dtype=torch.int64)
    self.rewards = torch.zeros(self.n)

    self.model = CNN()
    self.encode_features = self.encode_features_grid

    self.model.train()

    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

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
    #Sample an action according to the probability distribution specified by the policy.
    feats = self.encode_features(curr_obs)
    policy = self.model(feats, 'policy')
    action_sampler = Categorical(policy)
    return action_sampler.sample()

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
    t = timestep % self.n
    
    #Keep a running log of states, actions, and rewards encountered
    curr_feats = self.encode_features(curr_obs)
    self.states[t] = curr_feats
    self.actions[t] = torch.tensor(action)
    self.rewards[t] = reward * self.gamma**t

    #Only update every n steps
    if t == self.n - 1:
      #Compute policy loss
      self.optimizer.zero_grad()
      rets = torch.zeros(self.n)
      policies = self.model(self.states, 'policy')
      with torch.no_grad():
        values = self.model(self.states, 'value')
      policies = policies.gather(1, self.actions.view(-1,1)).flatten()
      next_feats = self.encode_features(next_obs)
      with torch.no_grad():
        final_value = self.model(next_feats, 'value')
      for i in range(self.n):
        rets[i] = torch.sum(self.rewards[i:]/(self.gamma**i)) + self.gamma**(self.n - i) * final_value
      advantages = rets - values

      policy_loss = torch.sum(-torch.log(policies) * advantages) 

      #Compute value loss
      values = self.model(self.states, 'value')
      advantages = rets - values
      value_loss = torch.sum(torch.square(advantages))

      #Compute total loss and gradient before updating
      loss = policy_loss + value_loss
      loss.backward()
      self.optimizer.step()
