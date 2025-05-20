import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from dmc import make_dmc_env

# Environment Setup
def make_env():
    env_name = "humanoid-walk"
    seed = np.random.randint(0, 1e6)
    return make_dmc_env(env_name, seed, flatten=True, use_pixels=False)

class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, capacity=1_000_000, device="cpu"):
        self.capacity = capacity
        self.device = device
        self.ptr = 0
        self.size = 0

        # Preallocate memory
        self.states      = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions     = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rewards     = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones       = np.zeros((capacity, 1), dtype=np.float32)

    def add(self, state, action, reward, next_state, done):
        self.states[self.ptr]      = state
        self.actions[self.ptr]     = action
        self.rewards[self.ptr]     = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr]       = done

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idxs = np.random.choice(self.size, batch_size, replace=False)
        states      = torch.from_numpy(self.states[idxs]).to(self.device)
        actions     = torch.from_numpy(self.actions[idxs]).to(self.device)
        rewards     = torch.from_numpy(self.rewards[idxs]).to(self.device)
        next_states = torch.from_numpy(self.next_states[idxs]).to(self.device)
        dones       = torch.from_numpy(self.dones[idxs]).to(self.device)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return self.size

class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim=256):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim=256, action_space=None,
                 log_std_min=-20, log_std_max=2, eps=1e-6):
        super(PolicyNetwork, self).__init__()
        assert action_space is not None, "pass env.action_space here"
        self.fc1 = nn.Linear(num_inputs, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, num_actions)
        self.log_std = nn.Linear(hidden_dim, num_actions)  # Changed to learned function of state
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.eps = eps

        high = torch.tensor(action_space.high, dtype=torch.float32)
        low  = torch.tensor(action_space.low,  dtype=torch.float32)
        self.register_buffer('action_scale', (high - low) / 2.0)
        self.register_buffer('action_bias',  (high + low) / 2.0)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = log_std.clamp(self.log_std_min, self.log_std_max)
        std = log_std.exp()
        return mean, std, log_std

    def sample(self, state):
        mean, std, _ = self.forward(state)
        dist = Normal(mean, std)
        raw_action = dist.rsample()
        log_prob = dist.log_prob(raw_action).sum(-1, keepdim=True)

        # Apply squashing and compute log_prob correction
        tanh_action = torch.tanh(raw_action)
        log_prob -= torch.sum(
            torch.log(1 - tanh_action.pow(2) + self.eps), 
            dim=-1, 
            keepdim=True
        )

        action = tanh_action * self.action_scale + self.action_bias
        return action, log_prob, dist

class AlphaTuner(nn.Module):
    def __init__(self, init_log_alpha=-2.0, target_entropy=None, act_dim=None, lr=3e-4, device="cpu"):
        super(AlphaTuner, self).__init__()
        self.log_alpha = nn.Parameter(torch.tensor(init_log_alpha, dtype=torch.float32))
        # Set target entropy to -dim(A) if not specified
        self.target_entropy = target_entropy if target_entropy is not None else -act_dim
        self.optimizer = optim.Adam([self.log_alpha], lr=lr)
        self.device = device
        self.to(device)

    def get_alpha(self):
        return self.log_alpha.exp()

    def update(self, log_pi):
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        self.optimizer.zero_grad()
        alpha_loss.backward()
        self.optimizer.step()
        return alpha_loss.item(), self.get_alpha().item()

class SACAgent:
    def __init__(self, env, hidden_dim=256, buffer_capacity=1_000_000,
                 batch_size=256, gamma=0.99, tau=0.005, lr=3e-4, 
                 start_steps=10000, update_after=1000, update_every=50, 
                 max_grad_norm=5):
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]
        
        # Training hyperparameters
        self.start_steps = start_steps  # Random exploration steps
        self.update_after = update_after  # Wait before starting updates
        self.update_every = update_every  # Policy update frequency
        self.max_grad_norm = max_grad_norm  # For gradient clipping

        self.policy_net = PolicyNetwork(self.obs_dim, self.act_dim,
                                        hidden_dim, action_space=env.action_space).to(self.device)
        self.q_net1 = QNetwork(self.obs_dim, self.act_dim, hidden_dim).to(self.device)
        self.q_net2 = QNetwork(self.obs_dim, self.act_dim, hidden_dim).to(self.device)
        self.target_q_net1 = QNetwork(self.obs_dim, self.act_dim, hidden_dim).to(self.device)
        self.target_q_net2 = QNetwork(self.obs_dim, self.act_dim, hidden_dim).to(self.device)

        # Initialize target networks with source network params
        self.hard_update(self.target_q_net1, self.q_net1)
        self.hard_update(self.target_q_net2, self.q_net2)

        # Initialize alpha tuner with proper target entropy
        self.alpha_tuner = AlphaTuner(device=self.device, act_dim=self.act_dim)
        self.buffer = ReplayBuffer(self.obs_dim, self.act_dim, buffer_capacity, device=self.device)
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.lr = lr

        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.q_optimizer1 = optim.Adam(self.q_net1.parameters(), lr=self.lr)
        self.q_optimizer2 = optim.Adam(self.q_net2.parameters(), lr=self.lr)
        
        self.total_steps = 0

    def hard_update(self, target, source):
        """Copy source network parameters to target"""
        for t_param, s_param in zip(target.parameters(), source.parameters()):
            t_param.data.copy_(s_param.data)

    def select_action(self, state, deterministic=False):
        # Random exploration
        if self.total_steps < self.start_steps:
            return self.env.action_space.sample()
            
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if deterministic:
            mean, _, _ = self.policy_net(state)
            tanh_mean = torch.tanh(mean)
            action = tanh_mean * self.policy_net.action_scale + self.policy_net.action_bias
            return action.detach().cpu().numpy()[0]
            
        action, _, _ = self.policy_net.sample(state)
        return action.detach().cpu().numpy()[0]

    def update(self):
        if len(self.buffer) < self.batch_size:
            return None, None, None, None, None

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        # Keep consistent dimensions
        rewards = rewards.view(-1, 1)
        dones = dones.view(-1, 1)

        with torch.no_grad():
            next_actions, next_log_probs, _ = self.policy_net.sample(next_states)
            t_q1 = self.target_q_net1(next_states, next_actions)
            t_q2 = self.target_q_net2(next_states, next_actions)
            t_q = torch.min(t_q1, t_q2) - self.alpha_tuner.get_alpha() * next_log_probs
            target_q = rewards + (1 - dones) * self.gamma * t_q

        # Q-function updates
        q1 = self.q_net1(states, actions)
        q2 = self.q_net2(states, actions)
        q_loss1 = F.mse_loss(q1, target_q)
        q_loss2 = F.mse_loss(q2, target_q)

        self.q_optimizer1.zero_grad()
        q_loss1.backward()
        # Apply gradient clipping
        torch.nn.utils.clip_grad_norm_(self.q_net1.parameters(), self.max_grad_norm)
        self.q_optimizer1.step()
        
        self.q_optimizer2.zero_grad()
        q_loss2.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net2.parameters(), self.max_grad_norm)
        self.q_optimizer2.step()

        # Policy update
        new_actions, log_probs, _ = self.policy_net.sample(states)
        q_new1 = self.q_net1(states, new_actions)
        q_new2 = self.q_net2(states, new_actions)
        q_new = torch.min(q_new1, q_new2)
        policy_loss = (self.alpha_tuner.get_alpha() * log_probs - q_new).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
        self.policy_optimizer.step()

        # Alpha update
        alpha_loss, alpha_value = self.alpha_tuner.update(log_probs.detach())

        # Soft target updates
        for t, p in zip(self.target_q_net1.parameters(), self.q_net1.parameters()):
            t.data.copy_(self.tau * p.data + (1 - self.tau) * t.data)
        for t, p in zip(self.target_q_net2.parameters(), self.q_net2.parameters()):
            t.data.copy_(self.tau * p.data + (1 - self.tau) * t.data)

        return q_loss1.item(), q_loss2.item(), policy_loss.item(), alpha_loss, alpha_value

def train(num_episodes=1000, log_interval=10, save_interval=250):
    env = make_env()
    agent = SACAgent(env)
    
    total_steps = 0
    avg_rewards = []  # For tracking performance
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        episode_steps = 0
        terminated = False
        q1_losses, q2_losses, p_losses, a_losses, alphas = [], [], [], [], []
        
        while not terminated:
            # Select action
            action = agent.select_action(state)
            
            # Execute action
            next_state, reward, done, truncated, _ = env.step(action)
            terminated = done or truncated
            
            # Store experience
            agent.buffer.add(state, action, reward, next_state, float(done))
            
            # Update agent
            if total_steps >= agent.update_after and total_steps % agent.update_every == 0:
                for _ in range(agent.update_every):  # Multiple updates per step
                    update_result = agent.update()
                    if update_result[0] is not None:
                        q1, q2, policy_loss, alpha_loss, alpha = update_result
                        q1_losses.append(q1)
                        q2_losses.append(q2)
                        p_losses.append(policy_loss)
                        a_losses.append(alpha_loss)
                        alphas.append(alpha)
            
            # Update state
            state = next_state
            episode_reward += reward
            episode_steps += 1
            total_steps += 1
            
        # Logging
        if q1_losses:
            avg_q1_loss = np.mean(q1_losses)
            avg_q2_loss = np.mean(q2_losses)
            avg_p_loss = np.mean(p_losses)
            avg_a_loss = np.mean(a_losses)
            avg_alpha = np.mean(alphas)
            print(f"Episode {episode + 1}: Reward: {episode_reward:.2f}, Steps: {episode_steps}, "
                  f"Q1 Loss: {avg_q1_loss:.4f}, Q2 Loss: {avg_q2_loss:.4f}, "
                  f"Policy Loss: {avg_p_loss:.4f}, Alpha Loss: {avg_a_loss:.4f}, Alpha: {avg_alpha:.4f}")
        else:
            print(f"Episode {episode + 1}: Reward: {episode_reward:.2f}, Steps: {episode_steps}, Filling buffer...")
        
        avg_rewards.append(episode_reward)
        
        # Periodic evaluation & saving
        if (episode + 1) % save_interval == 0:
            torch.save({
                'policy_state_dict': agent.policy_net.state_dict(),
                'q1_state_dict': agent.q_net1.state_dict(),
                'q2_state_dict': agent.q_net2.state_dict(),
                'target_q1_state_dict': agent.target_q_net1.state_dict(),
                'target_q2_state_dict': agent.target_q_net2.state_dict(),
                'alpha_tuner_state_dict': agent.alpha_tuner.state_dict()
            }, f"checkpoint_sac_{episode + 1}.pth")
            
            # Optional: Evaluate with deterministic policy
            eval_reward = evaluate(agent, env, num_episodes=5)
            print(f"Evaluation after {episode + 1} episodes: {eval_reward:.2f}")

def evaluate(agent, env, num_episodes=5):
    """Evaluate the agent with a deterministic policy"""
    total_rewards = []
    
    for _ in range(num_episodes):
        state, _ = env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state, deterministic=True)
            next_state, reward, done, truncated, _ = env.step(action)
            done = done or truncated
            episode_reward += reward
            state = next_state
            
        total_rewards.append(episode_reward)
        
    return np.mean(total_rewards)

if __name__ == "__main__":
    train()