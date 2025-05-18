import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class policy_network(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=64):
        super(policy_network, self).__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))  # learnable

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        std = torch.exp(self.log_std)
        return mean, std
    
    def get_action(self, obs):
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float32)
        mean, std = self.forward(obs)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action.detach().numpy(), log_prob.item()
    
policy_net = policy_network(obs_dim=67, act_dim=21)
policy_net.load_state_dict(torch.load("ppo_policy_77000.pth"))

# Do not modify the input of the 'act' function and the '__init__' function. 
class Agent(object):
    """Agent that acts randomly."""
    def __init__(self):
        self.action_space = gym.spaces.Box(-1.0, 1.0, (21,), np.float64)

    def act(self, observation):
        action, prob = policy_net.get_action(observation)
        return action
