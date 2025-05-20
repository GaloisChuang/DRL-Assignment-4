import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from dmc import make_dmc_env

LOG_STD_MIN = -20
LOG_STD_MAX = 2

def make_env():
    env_name = "humanoid-walk"
    seed = np.random.randint(0, 1e6)
    return make_dmc_env(env_name, seed, flatten=True, use_pixels=False)

class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, env):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU()
        )
        self.mean = nn.Linear(256, act_dim)
        self.log_std = nn.Linear(256, act_dim)

        init_w = 3e-3
        self.mean.weight.data.uniform_(-init_w, init_w)
        self.mean.bias.data.uniform_(-init_w, init_w)
        self.log_std.weight.data.uniform_(-init_w, init_w)
        self.log_std.bias.data.uniform_(-init_w, init_w)
        self.register_buffer(
            "action_scale",
            torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias",
            torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32)
        )

    def forward(self, state):
        x = self.fc(state)
        mean = self.mean(x)
        log_std = torch.tanh(self.log_std(x))
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t) - torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        return action, log_prob.sum(-1, keepdim=True), torch.tanh(mean) * self.action_scale + self.action_bias
    
env = make_env()
policy = GaussianPolicy(67, 21, env)
chechpoint = torch.load('New_750.pth')
policy.load_state_dict(chechpoint['policy'])

# Do not modify the input of the 'act' function and the '__init__' function. 
class Agent(object):
    """Agent that acts randomly."""
    def __init__(self):
        self.action_space = gym.spaces.Box(-1.0, 1.0, (21,), np.float64)

    def act(self, observation):
        state = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)
        action, _, _ = policy.sample(state)
        return action[0].cpu().detach().numpy()
