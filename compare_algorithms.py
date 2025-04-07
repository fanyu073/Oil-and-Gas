import torch
import numpy as np
from mega_oil_env import MegaOilEnv
from agent_manager import MegaAgent
from algorithms import DDPG, PPO, SAC
import matplotlib.pyplot as plt
import time
import pickle
from typing import Dict, List, Any

def train_algorithm(env, algorithm, name: str, num_episodes: int = 1000, max_steps: int = 200,
                   batch_size: int = 128, eval_freq: int = 100) -> Dict[str, List[float]]:
    """训练指定的算法并返回训练指标"""
    metrics = {
        'episode_rewards': [],
        'eval_rewards': [],
        'pressure_history': [],
        'wear_history': [],
        'oil_production': [],
        'efficiency': [],
        'critic_losses': [],
        'actor_losses': [],
        'training_times': []
    }
    
    print(f"\n开始训练 {name} 算法...")
    
    for episode in range(num_episodes):
        episode_start = time.time()
        obs, _ = env.reset()
        episode_reward = 0
        noise = max(0.05, 0.999 ** episode)
        
        for step in range(max_steps):
            # 选择动作
            if name == 'MADDPG':
                actions = algorithm.select_action(obs, noise)
                action = actions['coordinator']
            else:
                action = algorithm.select_action(obs, noise)
            
            # 执行动作
            next_obs, reward, done, _, info = env.step(action)
            
            # 存储经验
            if isinstance(algorithm, (DDPG, SAC)):
                algorithm.replay_buffer.append({
                    'state': obs,
                    'action': action,
                    'reward': reward,
                    'next_state': next_obs,
                    'done': done
                })
            elif isinstance(algorithm, PPO):
                algorithm.states.append(obs)
                algorithm.actions.append(action)
                algorithm.rewards.append(reward)
                algorithm.values.append(algorithm.critic(torch.FloatTensor(obs).unsqueeze(0).to(algorithm.device), 
                                                      torch.FloatTensor(action).unsqueeze(0).to(algorithm.device)).item())
            
            # 更新算法
            if (isinstance(algorithm, (DDPG, SAC)) and len(algorithm.replay_buffer) >= batch_size) or \
               (isinstance(algorithm, PPO) and len(algorithm.states) >= batch_size):
                critic_loss, actor_loss = algorithm.update(batch_size)
                metrics['critic_losses'].append(critic_loss)
                metrics['actor_losses'].append(actor_loss)
            
            # 更新状态和记录
            obs = next_obs
            episode_reward += reward
            
            if done:
                break
        
        # 记录训练数据
        episode_time = time.time() - episode_start
        metrics['episode_rewards'].append(episode_reward)
        metrics['training_times'].append(episode_time)
        
        # 记录环境指标
        metrics['pressure_history'].append(env.state['pressure'])
        metrics['wear_history'].append(env.state['wear'])
        metrics['oil_production'].append(env.max_oil - env.state['oil_left'])
        metrics['efficiency'].append((env.max_oil - env.state['oil_left']) / (env.state['wear'] + 1))
        
        # 显示训练进度
        print(f"\n{name} - Episode {episode}")
        print(f"Reward: {episode_reward:.2f}")
        print(f"Pressure: {env.state['pressure']:.2f}")
        print(f"Wear: {env.state['wear']:.2f}")
        print(f"Oil Production: {env.max_oil - env.state['oil_left']:.2f}")
        print(f"Efficiency: {(env.max_oil - env.state['oil_left']) / (env.state['wear'] + 1):.2f}")
        print(f"Time: {episode_time:.2f}s")
        
        # 定期评估
        if episode % eval_freq == 0:
            eval_metrics = evaluate(env, algorithm, name)
            metrics['eval_rewards'].append(eval_metrics['mean_reward'])
            print(f"\n{name} - Evaluation Results:")
            print(f"Mean Reward: {eval_metrics['mean_reward']:.2f}")
            print(f"Std Reward: {eval_metrics['std_reward']:.2f}")
            print(f"Mean Pressure: {eval_metrics['mean_pressure']:.2f}")
            print(f"Mean Efficiency: {eval_metrics['mean_efficiency']:.2f}")
    
    return metrics

def evaluate(env, algorithm, name: str, num_episodes: int = 5) -> Dict[str, float]:
    """评估算法性能"""
    eval_metrics = {
        'rewards': [],
        'pressures': [],
        'efficiencies': []
    }
    
    for _ in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        
        while True:
            # 选择动作
            if name == 'MADDPG':
                actions = algorithm.select_action(obs, noise=0.0)
                action = actions['coordinator']
            else:
                action = algorithm.select_action(obs, noise=0.0)
                
            next_obs, reward, done, _, _ = env.step(action)
            episode_reward += reward
            
            if done:
                eval_metrics['rewards'].append(episode_reward)
                eval_metrics['pressures'].append(env.state['pressure'])
                eval_metrics['efficiencies'].append(
                    (env.max_oil - env.state['oil_left']) / (env.state['wear'] + 1)
                )
                break
                
            obs = next_obs
    
    return {
        'mean_reward': np.mean(eval_metrics['rewards']),
        'std_reward': np.std(eval_metrics['rewards']),
        'mean_pressure': np.mean(eval_metrics['pressures']),
        'mean_efficiency': np.mean(eval_metrics['efficiencies'])
    }

def plot_comparison(metrics_dict: Dict[str, Dict[str, List[float]]]):
    """绘制算法对比图"""
    plt.figure(figsize=(15, 10))
    
    # 绘制训练奖励曲线
    plt.subplot(2, 2, 1)
    for name, metrics in metrics_dict.items():
        plt.plot(metrics['episode_rewards'], label=name)
    plt.title("Training Rewards Comparison")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    
    # 绘制评估奖励曲线
    plt.subplot(2, 2, 2)
    for name, metrics in metrics_dict.items():
        plt.plot(metrics['eval_rewards'], label=name)
    plt.title("Evaluation Rewards Comparison")
    plt.xlabel("Evaluation Episode")
    plt.ylabel("Reward")
    plt.legend()
    
    # 绘制压力和磨损曲线
    plt.subplot(2, 2, 3)
    for name, metrics in metrics_dict.items():
        plt.plot(metrics['pressure_history'], label=f"{name} Pressure")
        plt.plot(metrics['wear_history'], label=f"{name} Wear", linestyle='--')
    plt.title("Pressure and Wear Comparison")
    plt.xlabel("Episode")
    plt.ylabel("Value")
    plt.legend()
    
    # 绘制效率和产量曲线
    plt.subplot(2, 2, 4)
    for name, metrics in metrics_dict.items():
        plt.plot(metrics['efficiency'], label=f"{name} Efficiency")
        plt.plot(metrics['oil_production'], label=f"{name} Production", linestyle='--')
    plt.title("Efficiency and Production Comparison")
    plt.xlabel("Episode")
    plt.ylabel("Value")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("algorithm_comparison.png")
    plt.close()

def main():
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 初始化环境
    env = MegaOilEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 初始化算法
    algorithms = {
        'MADDPG': MegaAgent(state_dim, action_dim),
        'DDPG': DDPG(state_dim, action_dim, device),
        'PPO': PPO(state_dim, action_dim, device),
        'SAC': SAC(state_dim, action_dim, device)
    }
    
    # 训练和评估每个算法
    metrics_dict = {}
    for name, algorithm in algorithms.items():
        metrics = train_algorithm(env, algorithm, name)
        metrics_dict[name] = metrics
        
        # 保存每个算法的指标
        with open(f"{name.lower()}_metrics.pkl", 'wb') as f:
            pickle.dump(metrics, f)
    
    # 绘制对比图
    plot_comparison(metrics_dict)
    
    print("\n所有算法训练完成！结果已保存。")

if __name__ == "__main__":
    main() 