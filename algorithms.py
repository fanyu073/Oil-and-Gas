import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
from typing import Dict, List, Any

class Actor(nn.Module):
    """Actor网络，用于生成动作"""
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, action_dim)
        
        # 初始化权重
        nn.init.orthogonal_(self.fc1.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc2.weight, gain=np.sqrt(2))
        nn.init.orthogonal_(self.fc3.weight, gain=0.01)
        
        # 初始化偏置
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.constant_(self.fc3.bias, 0)
        
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        # 添加梯度裁剪
        x = torch.clamp(x, -1, 1)
        return x

class Critic(nn.Module):
    """Critic网络，用于评估状态-动作对的价值"""
    def __init__(self, state_dim, action_dim, num_agents=1):
        super(Critic, self).__init__()
        # 输入包含状态和动作
        input_dim = state_dim + action_dim * num_agents
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        
    def forward(self, state, action):
        # 连接状态和动作
        x = torch.cat([state, action], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DDPG:
    """DDPG算法实现"""
    def __init__(self, state_dim, action_dim, device):
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Actor网络
        self.actor = Actor(state_dim, action_dim).to(device)
        self.target_actor = Actor(state_dim, action_dim).to(device)
        self.target_actor.load_state_dict(self.actor.state_dict())
        
        # Critic网络
        self.critic = Critic(state_dim, action_dim).to(device)
        self.target_critic = Critic(state_dim, action_dim).to(device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        
        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.001)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.002)
        
        # 经验回放缓冲区
        self.replay_buffer = deque(maxlen=100000)
        
    def select_action(self, state, noise=0.0):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state).cpu().numpy().squeeze()
            action = action + noise * np.random.randn(*action.shape)
            action = np.clip(action, -1, 1)
        return action
    
    def update(self, batch_size=128):
        if len(self.replay_buffer) < batch_size:
            return 0.0, 0.0
            
        batch = random.sample(self.replay_buffer, batch_size)
        states = torch.FloatTensor(np.array([exp['state'] for exp in batch])).to(self.device)
        actions = torch.FloatTensor(np.array([exp['action'] for exp in batch])).to(self.device)
        rewards = torch.FloatTensor(np.array([exp['reward'] for exp in batch])).to(self.device)
        next_states = torch.FloatTensor(np.array([exp['next_state'] for exp in batch])).to(self.device)
        dones = torch.FloatTensor(np.array([exp['done'] for exp in batch])).to(self.device)
        
        # 更新Critic
        self.critic.train()
        self.critic_optimizer.zero_grad()
        
        with torch.no_grad():
            next_actions = self.target_actor(next_states)
            target_q = self.target_critic(next_states, next_actions)
            target_q = rewards + 0.99 * (1 - dones) * target_q
            
        current_q = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_q, target_q)
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # 更新Actor
        self.actor.train()
        self.actor_optimizer.zero_grad()
        
        actor_loss = -self.critic(states, self.actor(states)).mean()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # 软更新目标网络
        self._soft_update(self.target_actor, self.actor)
        self._soft_update(self.target_critic, self.critic)
        
        return critic_loss.item(), actor_loss.item()
    
    def _soft_update(self, target, source, tau=0.01):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

class PPO:
    """PPO算法实现"""
    def __init__(self, state_dim, action_dim, device):
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Actor网络
        self.actor = Actor(state_dim, action_dim).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.0003)
        
        # Critic网络
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.0003)
        
        # PPO参数
        self.clip_epsilon = 0.2
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.c1 = 1.0  # 价值函数系数
        self.c2 = 0.01  # 熵正则化系数
        self.min_std = 0.01  # 最小标准差
        
        # 经验缓冲区
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        
    def select_action(self, state, noise=0.0):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_mean = self.actor(state)
            # 确保标准差大于0且有限
            action_std = torch.ones_like(action_mean) * max(noise, self.min_std)
            action_std = torch.clamp(action_std, self.min_std, 1.0)
            
            # 检查并处理NaN值
            if torch.isnan(action_mean).any():
                action_mean = torch.zeros_like(action_mean)
            
            dist = torch.distributions.Normal(action_mean, action_std)
            action = dist.sample()
            action = torch.clamp(action, -1, 1)
            return action.cpu().numpy().squeeze()
    
    def update(self, batch_size=128):
        if len(self.states) < batch_size:
            return 0.0, 0.0
            
        # 转换为tensor
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.FloatTensor(np.array(self.actions)).to(self.device)
        rewards = torch.FloatTensor(np.array(self.rewards)).to(self.device)
        values = torch.FloatTensor(np.array(self.values)).to(self.device)
        
        # 计算优势函数
        advantages = self._compute_gae(rewards, values)
        
        # 计算新的动作概率和价值
        action_means = self.actor(states)
        # 确保标准差大于0且有限
        action_stds = torch.ones_like(action_means) * max(0.1, self.min_std)
        action_stds = torch.clamp(action_stds, self.min_std, 1.0)
        
        # 检查并处理NaN值
        if torch.isnan(action_means).any():
            action_means = torch.zeros_like(action_means)
            
        dist = torch.distributions.Normal(action_means, action_stds)
        new_log_probs = dist.log_prob(actions).sum(dim=1)
        new_values = self.critic(states, actions).squeeze()
        
        # 计算比率
        old_log_probs = torch.log(torch.exp(-0.5 * (actions - action_means).pow(2) / action_stds.pow(2))).sum(dim=1)
        ratio = torch.exp(new_log_probs - old_log_probs)
        
        # 计算PPO损失
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()
        
        # 计算价值损失
        value_pred = self.critic(states, actions).squeeze()
        value_targets = rewards + self.gamma * values
        critic_loss = nn.MSELoss()(value_pred, value_targets)
        
        # 更新网络
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        # 添加梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
        self.actor_optimizer.step()
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # 添加梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
        self.critic_optimizer.step()
        
        # 清空缓冲区
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        
        return critic_loss.item(), actor_loss.item()
    
    def _compute_gae(self, rewards, values):
        advantages = torch.zeros_like(rewards)
        gae = 0
        
        for t in reversed(range(len(rewards) - 1)):
            delta = rewards[t] + self.gamma * values[t + 1] - values[t]
            gae = delta + self.gamma * self.gae_lambda * gae
            advantages[t] = gae
            
        return advantages

class SAC:
    """SAC算法实现"""
    def __init__(self, state_dim, action_dim, device):
        self.device = device
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Actor网络
        self.actor = Actor(state_dim, action_dim).to(device)
        
        # 双Q网络
        self.q1 = Critic(state_dim, action_dim).to(device)
        self.q2 = Critic(state_dim, action_dim).to(device)
        
        # 目标Q网络
        self.target_q1 = Critic(state_dim, action_dim).to(device)
        self.target_q2 = Critic(state_dim, action_dim).to(device)
        self.target_q1.load_state_dict(self.q1.state_dict())
        self.target_q2.load_state_dict(self.q2.state_dict())
        
        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.0003)
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=0.0003)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=0.0003)
        
        # 经验回放缓冲区
        self.replay_buffer = deque(maxlen=100000)
        
    def select_action(self, state, noise=0.0):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state).cpu().numpy().squeeze()
            action = action + noise * np.random.randn(*action.shape)
            action = np.clip(action, -1, 1)
        return action
    
    def update(self, batch_size=128):
        if len(self.replay_buffer) < batch_size:
            return 0.0, 0.0
            
        batch = random.sample(self.replay_buffer, batch_size)
        states = torch.FloatTensor(np.array([exp['state'] for exp in batch])).to(self.device)
        actions = torch.FloatTensor(np.array([exp['action'] for exp in batch])).to(self.device)
        rewards = torch.FloatTensor(np.array([exp['reward'] for exp in batch])).to(self.device)
        next_states = torch.FloatTensor(np.array([exp['next_state'] for exp in batch])).to(self.device)
        dones = torch.FloatTensor(np.array([exp['done'] for exp in batch])).to(self.device)
        
        # 更新Q网络
        self.q1.train()
        self.q2.train()
        self.q1_optimizer.zero_grad()
        self.q2_optimizer.zero_grad()
        
        with torch.no_grad():
            next_actions = self.actor(next_states)
            next_q1 = self.target_q1(next_states, next_actions)
            next_q2 = self.target_q2(next_states, next_actions)
            next_q = torch.min(next_q1, next_q2)
            target_q = rewards + 0.99 * (1 - dones) * next_q
            
        current_q1 = self.q1(states, actions)
        current_q2 = self.q2(states, actions)
        q1_loss = nn.MSELoss()(current_q1, target_q)
        q2_loss = nn.MSELoss()(current_q2, target_q)
        
        q1_loss.backward()
        q2_loss.backward()
        self.q1_optimizer.step()
        self.q2_optimizer.step()
        
        # 更新Actor
        self.actor.train()
        self.actor_optimizer.zero_grad()
        
        current_actions = self.actor(states)
        current_q1 = self.q1(states, current_actions)
        current_q2 = self.q2(states, current_actions)
        current_q = torch.min(current_q1, current_q2)
        
        actor_loss = -current_q.mean()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # 软更新目标网络
        self._soft_update(self.target_q1, self.q1)
        self._soft_update(self.target_q2, self.q2)
        
        return (q1_loss.item() + q2_loss.item()) / 2, actor_loss.item()
    
    def _soft_update(self, target, source, tau=0.01):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data) 