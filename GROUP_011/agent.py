import numpy as np

class Agent():
  '''The agent class that is to be filled.
     You are allowed to add any method you
     want to this class.
  '''

  def __init__(self, env_specs):
    self.env_specs = env_specs
    self.alpha = 0.01
    self.gamma = 0.99
    self.eps = 0.1
    self.num_actions = 4
    self.input_size = 1582
    self.w = np.random.randn(self.input_size)

  def load_weights(self, root_path):
    # Add root_path in front of the path of the saved network parameters
    # For example if you have weights.pth in the GROUP_MJ1, do `root_path+"weights.pth"` while loading the parameters
    pass

  def encode_features(self, curr_obs, a):
    feats = np.concatenate((curr_obs[0].flatten(), curr_obs[1].flatten(), curr_obs[2].flatten()))
    action = np.zeros(self.num_actions)
    action[a] = 1
    feats = np.concatenate((feats, action))

    return feats

  def act(self, curr_obs, mode='eval'):
    if mode == 'train':
      rand_action = np.random.binomial(1, self.eps)
      if rand_action:
        return self.env_specs['action_space'].sample()

    q = np.zeros(self.num_actions)
    for a in range(self.num_actions):
      feats = self.encode_features(curr_obs, a)
      q[a] = np.dot(self.w, feats)
      
    return np.argmax(q)


  def update(self, curr_obs, action, reward, next_obs, done, timestep):
    cur_feats = self.encode_features(curr_obs, action)
    cur_q = np.dot(self.w, cur_feats)
    if done:
      self.w = self.w + self.alpha * (reward - cur_q) * cur_feats
    else:
      next_action = self.act(next_obs)
      next_feats = self.encode_features(next_obs, next_action)
      next_q = np.dot(self.w, next_feats)
      self.w = self.w + self.alpha * (reward + self.gamma * next_q - cur_q) * cur_feats

