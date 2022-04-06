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
    self.input_size = 15 * 15
    self.w = np.random.randn(self.input_size)

    self.action_position_embeds = [(7, 8), (6, 7), (8, 7), (7, 6)]

  def load_weights(self, root_path):
    # Add root_path in front of the path of the saved network parameters
    # For example if you have weights.pth in the GROUP_MJ1, do `root_path+"weights.pth"` while loading the parameters
    pass
  
  def encode_features(self, curr_obs, a):
    """
    In curr_obs[2].reshape((15, 15, 4)), the last dimension encodes the following:
    Jelly bean -> [0, 0, 1, 0]
    Truffle -> [0, 0, 0, 1]
    Banana -> [0, 1, 0, 0]
    Apple -> [1, 0, 0, 0]

    In our encoding, we give them, respectively, the values below:
    """    
    jb_val = 1
    truffle_val = 2
    banana_val = 3
    apple_val = 4
    empty_val = 0 # This might be too naive

    feats = curr_obs[2].reshape((15, 15, 4))
    feats = np.concatenate((feats, np.zeros((15, 15, 1), dtype=int)), axis=2) # Need to add fifth indicator with a "1" iff there's no object there
    for i in range(15):
      for j in range(15):
        if sum(feats[i][j]) == 0:
          feats[i][j][-1] = 1    

    feats = feats.astype(bool)
    values = np.broadcast_to(np.array([apple_val, banana_val, jb_val, truffle_val, empty_val]), (15, 15, 5))
    feats = values[feats].reshape((15, 15))
    feats[self.action_position_embeds[a][0], self.action_position_embeds[a][1]] += 1 # Adding the action embedding

    return feats.flatten()

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

