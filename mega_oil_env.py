import numpy as np
import gymnasium as gym
from gymnasium.spaces import Box, MultiDiscrete
from typing import Dict, List, Tuple, Any

class MegaOilEnv(gym.Env):
    def __init__(self, num_agents: int = 3):
        super(MegaOilEnv, self).__init__()
        
        # 环境参数
        self.num_agents = num_agents
        self.max_pressure = 5000
        self.optimal_pressure = 3000
        self.max_wear = 100
        self.max_oil = 10000
        self.max_steps = 200
        
        # 定义智能体角色
        self.agents = {
            "coordinator": 0,  # 主控智能体
            "pressure": 1,     # 压力控制智能体
            "extraction": 2    # 喷油智能体
        }
        
        # 观察空间
        self.observation_space = Box(
            low=np.array([0, 0, 0, 0, 0]),  # [压力, 磨损, 剩余油量, 喷油速率, 时间步]
            high=np.array([self.max_pressure, self.max_wear, self.max_oil, 500, self.max_steps]),
            dtype=np.float32
        )
        
        # 动作空间
        self.action_space = Box(
            low=np.array([-1.0, -1.0]),  # [压力控制, 喷油控制]
            high=np.array([1.0, 1.0]),
            dtype=np.float32
        )
        
        # 初始化环境
        self.reset()
        
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        """重置环境状态"""
        super().reset(seed=seed)
        
        # 初始化状态
        self.state = {
            "pressure": 2500,      # 初始压力
            "wear": 0,             # 设备磨损
            "oil_left": self.max_oil,  # 剩余油量
            "oil_rate": 100,       # 喷油速率
            "step": 0              # 当前步数
        }
        
        # 初始化监控指标
        self.metrics = {
            "pressure_history": [],
            "wear_history": [],
            "oil_production": [],
            "efficiency": []
        }
        
        return self._get_observation(), {}
    
    def _get_observation(self) -> np.ndarray:
        """获取当前观察状态"""
        return np.array([
            self.state["pressure"],
            self.state["wear"],
            self.state["oil_left"],
            self.state["oil_rate"],
            self.state["step"]
        ], dtype=np.float32)
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """执行一步环境交互"""
        # 处理不同算法的动作格式
        if isinstance(action, dict):
            # MADDPG算法的动作格式
            action = action['coordinator']
        elif len(action.shape) == 1:
            # 其他算法的动作格式
            action = action.reshape(-1)
            
        # 确保动作维度正确
        if len(action) != 2:
            raise ValueError(f"动作维度错误：期望2，实际{len(action)}")
            
        pressure_action, oil_action = action
        
        # 更新压力（将连续动作映射到实际压力变化）
        pressure_change = pressure_action * 50
        self.state["pressure"] += pressure_change
            
        # 更新喷油速率（将连续动作映射到实际喷油速率变化）
        oil_rate_change = oil_action * 20
        self.state["oil_rate"] += oil_rate_change
        
        # 限制状态范围
        self.state["pressure"] = np.clip(self.state["pressure"], 0, self.max_pressure)
        self.state["oil_rate"] = np.clip(self.state["oil_rate"], 0, 500)
        
        # 计算采油量
        extracted_oil = min(self.state["oil_left"], self.state["oil_rate"])
        self.state["oil_left"] -= extracted_oil
        
        # 更新设备磨损
        self.state["wear"] += abs(oil_action - 1) * 5
        self.state["wear"] = np.clip(self.state["wear"], 0, self.max_wear)
        
        # 更新时间步
        self.state["step"] += 1
        
        # 计算奖励
        reward = self._calculate_reward(extracted_oil)
        
        # 检查是否结束
        done = self._is_done()
        
        # 更新监控指标
        self._update_metrics()
        
        return self._get_observation(), reward, done, False, {}
    
    def _calculate_reward(self, extracted_oil: float) -> float:
        """计算奖励"""
        # 采油奖励
        reward_oil = extracted_oil / 100
        
        # 压力控制奖励
        reward_pressure = -abs(self.state["pressure"] - self.optimal_pressure) / 1000
        
        # 设备磨损惩罚
        reward_wear = -self.state["wear"] / 100
        
        # 综合奖励
        return 0.5 * reward_oil + 0.3 * reward_pressure + 0.2 * reward_wear
    
    def _is_done(self) -> bool:
        """检查是否结束"""
        return (self.state["oil_left"] <= 0 or 
                self.state["wear"] >= self.max_wear or 
                self.state["step"] >= self.max_steps)
    
    def _update_metrics(self):
        """更新监控指标"""
        self.metrics["pressure_history"].append(self.state["pressure"])
        self.metrics["wear_history"].append(self.state["wear"])
        self.metrics["oil_production"].append(self.max_oil - self.state["oil_left"])
        self.metrics["efficiency"].append(
            (self.max_oil - self.state["oil_left"]) / (self.state["wear"] + 1)
        )
    
    def render(self):
        """渲染环境状态"""
        print(f"Step: {self.state['step']}")
        print(f"Pressure: {self.state['pressure']:.2f}")
        print(f"Wear: {self.state['wear']:.2f}")
        print(f"Oil Left: {self.state['oil_left']:.2f}")
        print(f"Oil Rate: {self.state['oil_rate']:.2f}")
        print("---") 