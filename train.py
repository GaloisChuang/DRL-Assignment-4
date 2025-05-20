import os
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import gymnasium as gym
import numpy as np
from dmc import make_dmc_env
import matplotlib.pyplot as plt

LOG_STD_MIN = -20
LOG_STD_MAX = 2

# Environment Setup
def make_env():
    env_name = "humanoid-walk"
    seed = np.random.randint(0, 1e6)
    return make_dmc_env(env_name, seed, flatten=True, use_pixels=False)

# Replay Buffer
class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, capacity=int(1e6)):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0
        self.states      = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions     = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rewards     = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.masks       = np.zeros((capacity, 1), dtype=np.float32)

    def add(self, state, action, reward, next_state, done):
        self.states[self.ptr]      = state
        self.actions[self.ptr]     = action
        self.rewards[self.ptr]     = reward
        self.next_states[self.ptr] = next_state
        self.masks[self.ptr]       = 0.0 if done else 1.0
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = np.random.choice(self.size, batch_size, replace=False)
        return (
            torch.tensor(self.states[idx],      device=device),
            torch.tensor(self.actions[idx],     device=device),
            torch.tensor(self.rewards[idx],     device=device),
            torch.tensor(self.next_states[idx], device=device),
            torch.tensor(self.masks[idx],       device=device)
        )

# Networks
class QNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, s, a):
        return self.net(torch.cat([s, a], dim=-1))

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

# SAC Agent
class SAC:
    def __init__(self, obs_dim, act_dim, env, lr=3e-4, gamma=0.99, polyak=0.995):
        self.q1 = QNetwork(obs_dim, act_dim).to(device)
        self.q2 = QNetwork(obs_dim, act_dim).to(device)
        self.q1_target = QNetwork(obs_dim, act_dim).to(device)
        self.q2_target = QNetwork(obs_dim, act_dim).to(device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        # single optimizer for both Q-networks
        self.q_optimizer = optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr
        )
        self.policy = GaussianPolicy(obs_dim, act_dim, env).to(device)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr)
        self.target_entropy = -act_dim
        self.gamma = gamma
        self.polyak = polyak

    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, state, eval=False):
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        if eval:
            with torch.no_grad():
                _, _, a = self.policy.sample(state)
                return a[0].cpu().numpy()
        a, _, _ = self.policy.sample(state)
        return a[0].cpu().detach().numpy()

    def update(self, buffer, batch_size=256):
        s, a, r, s2, m = buffer.sample(batch_size)
        # Critic update
        with torch.no_grad():
            a2, log_pi2, _ = self.policy.sample(s2)
            q1_t = self.q1_target(s2, a2)
            q2_t = self.q2_target(s2, a2)
            min_q_t = torch.min(q1_t, q2_t) - self.alpha() * log_pi2
            target = r + self.gamma * m * min_q_t

        q1_loss = F.mse_loss(self.q1(s, a), target)
        q2_loss = F.mse_loss(self.q2(s, a), target)
        self.q_optimizer.zero_grad()
        (q1_loss + q2_loss).backward()
        self.q_optimizer.step()

        # Policy update
        a_new, log_pi, _ = self.policy.sample(s)
        q1_pi = self.q1(s, a_new)
        q2_pi = self.q2(s, a_new)
        min_q_pi = torch.min(q1_pi, q2_pi)
        policy_loss = (self.alpha().detach() * log_pi - min_q_pi).mean()
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # Alpha update
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # Target networks update
        with torch.no_grad():
            for p, tp in zip(self.q1.parameters(), self.q1_target.parameters()):
                tp.data.mul_(self.polyak).add_(p.data * (1 - self.polyak))
            for p, tp in zip(self.q2.parameters(), self.q2_target.parameters()):
                tp.data.mul_(self.polyak).add_(p.data * (1 - self.polyak))

        return q1_loss.item(), q2_loss.item(), policy_loss.item(), alpha_loss.item(), self.alpha().item()

# Hyperparameters and setup
env = make_env()
obs, _ = env.reset()
obs_dim = obs.shape[0]
act_dim = env.action_space.shape[0]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize
agent = SAC(obs_dim, act_dim, env)
buffer = ReplayBuffer(obs_dim, act_dim)
episodes = 5000

# Training loop
global_step = 0
for ep in range(1, episodes + 1):
    obs, _ = env.reset()
    done = False
    total_reward = 0
    q1_l, q2_l, p_l, a_l, alpha = None, None, None, None, None
    while not done:
        a = agent.select_action(obs)
        next_obs, r, term, trunc, _ = env.step(a)
        done = term or trunc
        buffer.add(obs, a, r, next_obs, term)
        obs = next_obs
        total_reward += r
        global_step += 1

        if buffer.size > 10000:
            q1_l, q2_l, p_l, a_l, alpha = agent.update(buffer)

    # Logging
    if q1_l is not None:
        print(f"Episode: {ep}, Reward: {total_reward:.2f}, Q1 Loss: {q1_l:.4f}, Q2 Loss: {q2_l:.4f}, Policy Loss: {p_l:.4f}, Alpha Loss: {a_l:.4f}, Alpha: {alpha:.4f}")
    else:
        print(f"Episode: {ep}, Reward: {total_reward:.2f}, Waiting for buffer to fill...")

    if ep % 250 == 0:
        torch.save({
            'policy': agent.policy.state_dict(),
            'q1':     agent.q1.state_dict(),
            'q2':     agent.q2.state_dict(),
            'alpha':  agent.log_alpha
        }, f"New_{ep}.pth")
