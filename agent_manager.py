import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Any
import numpy as np
from collections import deque
import random

class Actor(nn.Module):
    """Actor网络，用于生成动作"""
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, action_dim)
        
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))  # 输出范围[-1,1]

class Critic(nn.Module):
    """Critic网络，用于评估状态-动作对的价值"""
    def __init__(self, state_dim, action_dim, num_agents):
        super(Critic, self).__init__()
        # 输入包含状态和所有智能体的动作
        input_dim = state_dim + action_dim * num_agents
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        
    def forward(self, state, actions):
        # 连接状态和动作
        x = torch.cat([state, actions], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class MegaAgent:
    """MegaAgent框架下的MADDPG实现"""
    def __init__(self, state_dim, action_dim, num_agents=3):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # 创建Actor网络
        self.actors = {
            'coordinator': Actor(state_dim, action_dim).to(self.device),
            'pressure': Actor(state_dim, action_dim).to(self.device),
            'extraction': Actor(state_dim, action_dim).to(self.device)
        }
        
        # 创建Critic网络
        self.critics = {
            'coordinator': Critic(state_dim, action_dim, num_agents).to(self.device),
            'pressure': Critic(state_dim, action_dim, num_agents).to(self.device),
            'extraction': Critic(state_dim, action_dim, num_agents).to(self.device)
        }
        
        # 创建目标网络
        self.target_actors = {
            name: Actor(state_dim, action_dim).to(self.device)
            for name in self.actors.keys()
        }
        self.target_critics = {
            name: Critic(state_dim, action_dim, num_agents).to(self.device)
            for name in self.critics.keys()
        }
        
        # 复制参数到目标网络
        for name in self.actors.keys():
            self.target_actors[name].load_state_dict(self.actors[name].state_dict())
            self.target_critics[name].load_state_dict(self.critics[name].state_dict())
        
        # 创建优化器
        self.actor_optimizers = {
            name: optim.Adam(actor.parameters(), lr=0.001)
            for name, actor in self.actors.items()
        }
        self.critic_optimizers = {
            name: optim.Adam(critic.parameters(), lr=0.002)
            for name, critic in self.critics.items()
        }
        
        # 经验回放缓冲区
        self.replay_buffer = deque(maxlen=100000)
        
    def select_action(self, state, noise=0.0):
        """选择动作"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        actions = {}
        
        for name, actor in self.actors.items():
            actor.eval()
            with torch.no_grad():
                action = actor(state).cpu().numpy().squeeze()
                # 添加探索噪声
                action = action + noise * np.random.randn(*action.shape)
                action = np.clip(action, -1, 1)
                actions[name] = action
        
        return actions
    
    def update(self, batch_size=128):
        """更新网络"""
        if len(self.replay_buffer) < batch_size:
            return 0.0, 0.0
            
        # 采样批次数据
        batch = random.sample(self.replay_buffer, batch_size)
        states = torch.FloatTensor(np.array([exp['state'] for exp in batch])).to(self.device)
        actions = torch.FloatTensor(np.array([exp['action'] for exp in batch])).to(self.device)
        rewards = torch.FloatTensor(np.array([exp['reward'] for exp in batch])).to(self.device)
        next_states = torch.FloatTensor(np.array([exp['next_state'] for exp in batch])).to(self.device)
        dones = torch.FloatTensor(np.array([exp['done'] for exp in batch])).to(self.device)
        
        total_critic_loss = 0.0
        total_actor_loss = 0.0
        
        # 更新每个智能体
        for name in self.actors.keys():
            # 更新Critic
            self.critics[name].train()
            self.critic_optimizers[name].zero_grad()
            
            # 计算目标Q值
            with torch.no_grad():
                next_actions = []
                for agent_name in self.actors.keys():
                    next_agent_action = self.target_actors[agent_name](next_states)
                    next_actions.append(next_agent_action)
                next_actions = torch.cat(next_actions, dim=1)
                
                target_q = self.target_critics[name](next_states, next_actions)
                target_q = rewards + 0.99 * (1 - dones) * target_q
            
            # 计算当前Q值
            current_q = self.critics[name](states, actions)
            critic_loss = nn.MSELoss()(current_q, target_q)
            critic_loss.backward()
            self.critic_optimizers[name].step()
            total_critic_loss += critic_loss.item()
            
            # 更新Actor
            self.actors[name].train()
            self.actor_optimizers[name].zero_grad()
            
            # 计算策略梯度
            current_actions = []
            for agent_name in self.actors.keys():
                if agent_name == name:
                    current_agent_action = self.actors[agent_name](states)
                else:
                    current_agent_action = self.actors[agent_name](states).detach()
                current_actions.append(current_agent_action)
            
            current_actions = torch.cat(current_actions, dim=1)
            actor_loss = -self.critics[name](states, current_actions).mean()
            actor_loss.backward()
            self.actor_optimizers[name].step()
            total_actor_loss += actor_loss.item()
            
            # 软更新目标网络
            self._soft_update(self.target_actors[name], self.actors[name])
            self._soft_update(self.target_critics[name], self.critics[name])
        
        return total_critic_loss / self.num_agents, total_actor_loss / self.num_agents
    
    def _soft_update(self, target, source, tau=0.01):
        """软更新目标网络参数"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

class MonitorModule:
    """监控模块"""
    def __init__(self):
        self.metrics = {
            'pressure': [],
            'wear': [],
            'oil_production': [],
            'efficiency': []
        }
        
    def update(self, state: Dict[str, Any]):
        """更新监控指标"""
        self.metrics['pressure'].append(state['pressure'])
        self.metrics['wear'].append(state['wear'])
        self.metrics['oil_production'].append(state['oil_production'])
        self.metrics['efficiency'].append(state['efficiency'])
        
    def get_report(self) -> Dict[str, Any]:
        """获取监控报告"""
        return {
            'avg_pressure': np.mean(self.metrics['pressure']),
            'max_wear': max(self.metrics['wear']),
            'total_oil': sum(self.metrics['oil_production']),
            'avg_efficiency': np.mean(self.metrics['efficiency'])
        }

class StorageModule:
    """数据存储模块"""
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.buffer = []
        self.training_logs = []
        self.evaluation_results = []
        
    def store_experience(self, experience: Dict[str, Any]):
        """存储经验数据"""
        if len(self.buffer) >= self.max_size:
            self.buffer.pop(0)
        self.buffer.append(experience)
        
    def sample_batch(self, batch_size: int) -> List[Dict[str, Any]]:
        """采样批次数据"""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]
        
    def log_training(self, log_data: Dict[str, Any]):
        """记录训练日志"""
        self.training_logs.append(log_data)
        
    def log_evaluation(self, eval_data: Dict[str, Any]):
        """记录评估结果"""
        self.evaluation_results.append(eval_data) 