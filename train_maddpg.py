import torch
import numpy as np
from mega_oil_env import MegaOilEnv
from agent_manager import MegaAgent
import matplotlib.pyplot as plt
import time

def train():
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 初始化环境和MegaAgent
    env = MegaOilEnv()
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    mega_agent = MegaAgent(state_dim, action_dim)
    
    # 训练参数
    num_episodes = 1000
    max_steps = 200
    batch_size = 128
    eval_freq = 100
    save_freq = 500
    
    # 训练记录
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
    
    # 训练循环
    for episode in range(num_episodes):
        episode_start = time.time()
        obs, _ = env.reset()
        episode_reward = 0
        noise = max(0.05, 0.999 ** episode)  # 噪声逐渐衰减
        
        for step in range(max_steps):
            # 选择动作
            actions = mega_agent.select_action(obs, noise)
            action = actions['coordinator']
            
            # 执行动作
            next_obs, reward, done, _, info = env.step(action)
            
            # 存储经验
            mega_agent.replay_buffer.append({
                'state': obs,
                'action': np.concatenate([actions[name] for name in mega_agent.actors.keys()]),
                'reward': reward,
                'next_state': next_obs,
                'done': done
            })
            
            # 更新MegaAgent
            if len(mega_agent.replay_buffer) >= batch_size:
                critic_loss, actor_loss = mega_agent.update(batch_size)
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
        
        # 每次训练都显示结果
        print(f"Episode {episode}")
        print(f"Reward: {episode_reward:.2f}")
        print(f"Pressure: {env.state['pressure']:.2f}")
        print(f"Wear: {env.state['wear']:.2f}")
        print(f"Oil Production: {env.max_oil - env.state['oil_left']:.2f}")
        print(f"Efficiency: {(env.max_oil - env.state['oil_left']) / (env.state['wear'] + 1):.2f}")
        print(f"Time: {episode_time:.2f}s")
        print("---")
        
        # 定期评估
        if episode % eval_freq == 0:
            eval_metrics = evaluate(env, mega_agent)
            metrics['eval_rewards'].append(eval_metrics['mean_reward'])
            print(f"Evaluation Results:")
            print(f"Mean Reward: {eval_metrics['mean_reward']:.2f}")
            print(f"Std Reward: {eval_metrics['std_reward']:.2f}")
            print(f"Mean Pressure: {eval_metrics['mean_pressure']:.2f}")
            print(f"Mean Efficiency: {eval_metrics['mean_efficiency']:.2f}")
            print("---")
        
        # 定期保存模型和指标
        if episode % save_freq == 0:
            save_model(mega_agent, f"mega_agent_episode_{episode}.pth")
            save_metrics(metrics, f"metrics_episode_{episode}.pkl")
            print(f"Model and metrics saved at episode {episode}")
    
    # 保存最终模型和指标
    save_model(mega_agent, "mega_agent_final.pth")
    save_metrics(metrics, "metrics_final.pkl")
    print("Training completed. Final model and metrics saved.")
    
    # 绘制训练曲线
    plot_training_curves(metrics)

def evaluate(env, mega_agent, num_episodes=5):
    eval_metrics = {
        'rewards': [],
        'pressures': [],
        'efficiencies': []
    }
    
    for _ in range(num_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        
        while True:
            actions = mega_agent.select_action(obs, noise=0.0)
            action = actions['coordinator']
            
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

def save_model(mega_agent, filename):
    torch.save({
        'actors_state_dict': {name: actor.state_dict() for name, actor in mega_agent.actors.items()},
        'critics_state_dict': {name: critic.state_dict() for name, critic in mega_agent.critics.items()},
        'target_actors_state_dict': {name: actor.state_dict() for name, actor in mega_agent.target_actors.items()},
        'target_critics_state_dict': {name: critic.state_dict() for name, critic in mega_agent.target_critics.items()},
        'actor_optimizers_state_dict': {name: optimizer.state_dict() for name, optimizer in mega_agent.actor_optimizers.items()},
        'critic_optimizers_state_dict': {name: optimizer.state_dict() for name, optimizer in mega_agent.critic_optimizers.items()}
    }, filename)

def save_metrics(metrics, filename):
    import pickle
    with open(filename, 'wb') as f:
        pickle.dump(metrics, f)

def plot_training_curves(metrics):
    plt.figure(figsize=(15, 10))
    
    # 绘制训练奖励曲线
    plt.subplot(2, 2, 1)
    plt.plot(metrics['episode_rewards'])
    plt.title("Training Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    
    # 绘制评估奖励曲线
    plt.subplot(2, 2, 2)
    plt.plot(metrics['eval_rewards'])
    plt.title("Evaluation Rewards")
    plt.xlabel("Evaluation Episode")
    plt.ylabel("Reward")
    
    # 绘制压力和磨损曲线
    plt.subplot(2, 2, 3)
    plt.plot(metrics['pressure_history'], label='Pressure')
    plt.plot(metrics['wear_history'], label='Wear')
    plt.title("Pressure and Wear History")
    plt.xlabel("Episode")
    plt.ylabel("Value")
    plt.legend()
    
    # 绘制效率和产量曲线
    plt.subplot(2, 2, 4)
    plt.plot(metrics['efficiency'], label='Efficiency')
    plt.plot(metrics['oil_production'], label='Oil Production')
    plt.title("Efficiency and Oil Production")
    plt.xlabel("Episode")
    plt.ylabel("Value")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig("training_analysis.png")
    plt.close()

if __name__ == "__main__":
    train() 