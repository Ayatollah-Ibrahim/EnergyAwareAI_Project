"""
PPO Training for Energy Harvesting Camera System - Jetson Nano
Part 5: Training Loop and Curriculum Learning

This module implements:
- Curriculum training pipeline
- Single stage training
- Evaluation functions
- Training visualization
- Progress tracking

Author: Revised Implementation
Date: 2026-02-06
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from collections import deque
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Configuration
HISTORY_MAXLEN = 10000  # Maximum history to keep in memory

# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================

def evaluate_policy(
    env,
    agent,  # PPOAgent
    n_episodes: int = 10,
    deterministic: bool = True
) -> float:
    """
    Evaluate policy performance.
    
    Args:
        env: Environment
        agent: PPO agent
        n_episodes: Number of episodes
        deterministic: Use greedy action selection
    
    Returns:
        Mean episode reward
    """
    episode_rewards = []
    
    for _ in range(n_episodes):
        state, _ = env.reset()
        episode_reward = 0.0
        done = False
        
        while not done:
            action, _, _ = agent.select_action(state, deterministic)
            
            mode, capture, process, tx = action
            flat_action = mode * 18 + capture * 6 + process * 2 + tx
            
            next_state, reward, terminated, truncated, _ = env.step(flat_action)
            done = terminated or truncated
            
            episode_reward += reward
            state = next_state
        
        episode_rewards.append(episode_reward)
    
    return float(np.mean(episode_rewards))

# ============================================================================
# SINGLE STAGE TRAINING
# ============================================================================

def train_ppo_stage(
    env,
    agent,  # PPOAgent
    n_episodes: int,
    stage_name: str,
    save_dir: Path,
    eval_interval: int = 100
) -> Dict[str, List[float]]:
    """
    Train PPO agent for one curriculum stage.
    
    Args:
        env: Environment
        agent: PPO agent
        n_episodes: Number of episodes
        stage_name: Stage identifier
        save_dir: Directory for checkpoints
        eval_interval: Episodes between evaluations
    
    Returns:
        Training history dictionary
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    
    history = {
        'episode_rewards': deque(maxlen=HISTORY_MAXLEN),
        'episode_lengths': deque(maxlen=HISTORY_MAXLEN),
        'delivery_rates': deque(maxlen=HISTORY_MAXLEN),
        'energy_efficiency': deque(maxlen=HISTORY_MAXLEN),
        'policy_loss': deque(maxlen=HISTORY_MAXLEN),
        'value_loss': deque(maxlen=HISTORY_MAXLEN),
        'entropy': deque(maxlen=HISTORY_MAXLEN),
        'explained_variance': deque(maxlen=HISTORY_MAXLEN),
        'eval_rewards': deque(maxlen=HISTORY_MAXLEN // 10),
        'delivery_reward_pct': deque(maxlen=HISTORY_MAXLEN),
        'quality_reward_pct': deque(maxlen=HISTORY_MAXLEN),
        'shaping_reward_pct': deque(maxlen=HISTORY_MAXLEN),
        'penalty_pct': deque(maxlen=HISTORY_MAXLEN)
    }
    
    episode_rewards = deque(maxlen=100)
    episode_deliveries = deque(maxlen=100)
    episode = 0
    
    while episode < n_episodes:
        state, info = env.reset()
        episode_reward = 0.0
        episode_length = 0
        episode_done = False
        
        # Track reward components
        episode_reward_components = {
            'R_delivery_raw': 0.0,
            'R_quality_raw': 0.0,
            'R_shaping_raw': 0.0,
            'P_total': 0.0
        }
        
        while not episode_done:
            action, log_prob, value = agent.select_action(state)
            
            mode, capture, process, tx = action
            flat_action = mode * 18 + capture * 6 + process * 2 + tx
            
            next_state, reward, terminated, truncated, info = env.step(flat_action)
            done = terminated or truncated
            
            # Accumulate reward components
            if 'reward_breakdown' in info:
                rb = info['reward_breakdown']
                episode_reward_components['R_delivery_raw'] += rb.get('R_delivery_raw', 0)
                episode_reward_components['R_quality_raw'] += (
                    rb.get('R_quality_raw', 0) + rb.get('R_redundancy_raw', 0)
                )
                episode_reward_components['R_shaping_raw'] += rb.get('R_shaping_raw', 0)
                episode_reward_components['P_total'] += (
                    rb.get('P_missed_raw', 0) + 
                    rb.get('P_overflow_raw', 0) + 
                    rb.get('P_battery_raw', 0)
                )
            
            agent.store_transition(state, action, log_prob, reward, value, done)
            
            episode_reward += reward
            episode_length += 1
            agent.total_timesteps += 1
            
            state = next_state
            
            # Update policy when buffer is full
            if agent.buffer.is_full():
                if not done:
                    with torch.no_grad():
                        state_tensor = torch.as_tensor(
                            state, 
                            dtype=torch.float32, 
                            device=agent.config.device
                        )
                        last_value = agent.critic(state_tensor.unsqueeze(0)).item()
                else:
                    last_value = 0.0
                
                agent.finish_episode(last_value=last_value)
                stats = agent.update()
                
                if stats:
                    history['policy_loss'].append(stats['policy_loss'])
                    history['value_loss'].append(stats['value_loss'])
                    history['entropy'].append(stats['entropy'])
                    history['explained_variance'].append(
                        stats.get('explained_variance', 0)
                    )
            
            if done or episode_length >= env.params.max_episode_steps:
                if not agent.buffer.is_full() and agent.buffer.ptr > 0:
                    agent.finish_episode(last_value=0.0)
                episode_done = True
        
        # Episode complete
        episode_rewards.append(episode_reward)
        history['episode_rewards'].append(episode_reward)
        history['episode_lengths'].append(episode_length)
        
        event_stats = env.buffer.get_event_delivery_stats()
        delivery_rate = event_stats['delivery_rate']
        episode_deliveries.append(delivery_rate)
        
        energy_eff = (
            env.episode_stats['energy_harvested'] / 
            max(env.episode_stats['energy_consumed'], 1.0)
        )
        
        history['delivery_rates'].append(delivery_rate)
        history['energy_efficiency'].append(energy_eff)
        
        # Track reward component percentages
        total_positive = (
            episode_reward_components['R_delivery_raw'] +
            episode_reward_components['R_quality_raw'] +
            episode_reward_components['R_shaping_raw']
        )
        if total_positive > 0:
            history['delivery_reward_pct'].append(
                100 * episode_reward_components['R_delivery_raw'] / total_positive
            )
            history['quality_reward_pct'].append(
                100 * episode_reward_components['R_quality_raw'] / total_positive
            )
            history['shaping_reward_pct'].append(
                100 * episode_reward_components['R_shaping_raw'] / total_positive
            )
        else:
            history['delivery_reward_pct'].append(0)
            history['quality_reward_pct'].append(0)
            history['shaping_reward_pct'].append(0)
        
        history['penalty_pct'].append(episode_reward_components['P_total'])
        
        # Logging
        if (episode + 1) % 20 == 0:
            mean_reward = np.mean(episode_rewards)
            mean_delivery = np.mean(episode_deliveries)
            
            recent_delivery_pct = np.mean(
                list(history['delivery_reward_pct'])[-20:]
            )
            recent_quality_pct = np.mean(
                list(history['quality_reward_pct'])[-20:]
            )
            recent_explained_var = np.mean(
                list(history['explained_variance'])[-20:]
            ) if history['explained_variance'] else 0
            
            logger.info(
                f"[{stage_name}] Ep {episode + 1}/{n_episodes} | "
                f"R: {episode_reward:.2f} | "
                f"Mean(100): {mean_reward:.2f} | "
                f"Del: {mean_delivery:.1%} | "
                f"DeliveryR%: {recent_delivery_pct:.0f}% | "
                f"QualityR%: {recent_quality_pct:.0f}% | "
                f"ExpVar: {recent_explained_var:.2f}"
            )
            
            # Warning if delivery reward too low
            if recent_delivery_pct < 50 and episode > 100:
                logger.warning(
                    f"⚠️  Delivery reward is only {recent_delivery_pct:.0f}% of total! "
                    "Agent may be gaming the reward function."
                )
        
        # Evaluation
        if (episode + 1) % eval_interval == 0:
            eval_reward = evaluate_policy(env, agent, n_episodes=5)
            history['eval_rewards'].append(eval_reward)
            logger.info(f"[{stage_name}] Evaluation: {eval_reward:.2f}")
        
        episode += 1
    
    return {k: list(v) for k, v in history.items()}

# ============================================================================
# CURRICULUM TRAINING
# ============================================================================

def train_curriculum(
    data_path: str = "NREL.csv",
    stages: Optional[List] = None,  # List[CurriculumStage]
    save_dir: str = "experiments_Jetson/curriculum",
    eval_interval: int = 100
):
    """
    Train agent using curriculum learning.
    
    Args:
        data_path: Path to GHI data
        stages: Curriculum stages (None for default)
        save_dir: Directory for saving
        eval_interval: Episodes between evaluations
    
    Returns:
        Tuple of (agent, history)
    """
    logger.info("=" * 80)
    logger.info("CURRICULUM LEARNING TRAINING")
    logger.info("=" * 80)
    
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # Load curriculum
    if stages is None:
        from src.env.core import create_curriculum_stages
        stages = create_curriculum_stages()
    
    logger.info("\nCurriculum Overview:")
    logger.info("=" * 80)
    for i, stage in enumerate(stages, 1):
        logger.info(f"\nStage {i}/{len(stages)}:")
        logger.info(stage.get_description())
    logger.info("=" * 80)
    
    # Initialize environment and agent
    from src.env.core import SystemParameters
    from src.env.environment import EnergyHarvestingCameraEnv
    from src.agents.ppo import PPOConfig, PPOAgent
    
    base_params = SystemParameters()
    env = EnergyHarvestingCameraEnv(data_path, base_params)
    
    config = PPOConfig(
        state_dim=env.observation_space.shape[0],
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    agent = PPOAgent(config)
    
    # Training history
    full_history = {
        'stage_names': deque(maxlen=HISTORY_MAXLEN),
        'stage_episodes': deque(maxlen=HISTORY_MAXLEN),
        'episode_rewards': deque(maxlen=HISTORY_MAXLEN),
        'episode_lengths': deque(maxlen=HISTORY_MAXLEN),
        'delivery_rates': deque(maxlen=HISTORY_MAXLEN),
        'energy_efficiency': deque(maxlen=HISTORY_MAXLEN),
        'policy_loss': deque(maxlen=HISTORY_MAXLEN),
        'value_loss': deque(maxlen=HISTORY_MAXLEN),
        'entropy': deque(maxlen=HISTORY_MAXLEN),
        'eval_rewards': deque(maxlen=HISTORY_MAXLEN),
        'stage_boundaries': []
    }
    
    total_episodes = 0
    
    # Train each stage
    for stage_idx, stage in enumerate(stages, 1):
        logger.info("\n" + "=" * 80)
        logger.info(f"STARTING STAGE {stage_idx}/{len(stages)}: {stage.name}")
        logger.info("=" * 80)
        
        # Apply stage parameters
        stage_params = stage.apply_to_params(SystemParameters())
        stage_params.save(Path(save_dir) / f"stage_{stage_idx}_params.json")
        
        # Create environment with stage parameters
        env = EnergyHarvestingCameraEnv(
            data_path,
            stage_params,
            event_timeout=stage.event_timeout
        )
        
        # Update agent hyperparameters
        agent.config.update_for_stage(stage)
        agent.update_learning_rate(stage.learning_rate)
        
        # Train stage
        logger.info(f"\nTraining for {stage.n_episodes} episodes...")
        stage_history = train_ppo_stage(
            env=env,
            agent=agent,
            n_episodes=stage.n_episodes,
            stage_name=stage.name,
            save_dir=Path(save_dir) / f"stage_{stage_idx}",
            eval_interval=eval_interval
        )
        
        # Accumulate history
        full_history['stage_names'].extend(
            [stage.name] * len(stage_history['episode_rewards'])
        )
        full_history['stage_episodes'].extend(
            range(total_episodes, total_episodes + len(stage_history['episode_rewards']))
        )
        full_history['episode_rewards'].extend(stage_history['episode_rewards'])
        full_history['episode_lengths'].extend(stage_history['episode_lengths'])
        full_history['delivery_rates'].extend(stage_history['delivery_rates'])
        full_history['energy_efficiency'].extend(stage_history['energy_efficiency'])
        full_history['policy_loss'].extend(stage_history['policy_loss'])
        full_history['value_loss'].extend(stage_history['value_loss'])
        full_history['entropy'].extend(stage_history['entropy'])
        full_history['eval_rewards'].extend(stage_history['eval_rewards'])
        full_history['stage_boundaries'].append(total_episodes)
        
        total_episodes += stage.n_episodes
        
        # Save stage checkpoint
        agent.save(Path(save_dir) / f"stage_{stage_idx}_final.pt")
        
        logger.info(f"\n✅ Stage {stage_idx} completed!")
        logger.info(
            f"   Final mean reward (last 100): "
            f"{np.mean(stage_history['episode_rewards'][-100:]):.2f}"
        )
        logger.info(
            f"   Final delivery rate (last 100): "
            f"{np.mean(stage_history['delivery_rates'][-100:]):.1%}"
        )
    
    # Save final agent
    agent.save(Path(save_dir) / "final_agent.pt")
    
    # Save full history
    with open(Path(save_dir) / "full_history.json", 'w') as f:
        serializable_history = {}
        for k, v in full_history.items():
            v_list = list(v)
            serializable_history[k] = [
                float(x) if isinstance(x, (int, float, np.number)) else x
                for x in v_list
            ]
        json.dump(serializable_history, f, indent=2)
    
    # Plot results
    plot_curriculum_results(
        full_history, 
        save_path=str(Path(save_dir) / "curriculum_training.png")
    )
    
    logger.info("\n" + "=" * 80)
    logger.info("CURRICULUM TRAINING COMPLETED!")
    logger.info(f"Total episodes: {total_episodes}")
    logger.info(f"Results saved to: {save_dir}")
    logger.info("=" * 80)
    
    return agent, full_history

# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_curriculum_results(
    history: Dict, 
    save_path: Optional[str] = None
):
    """
    Plot curriculum learning results.
    
    Args:
        history: Training history
        save_path: Path to save figure
    """
    # Convert deques to lists
    if isinstance(history['stage_episodes'], deque):
        history = {
            k: list(v) if isinstance(v, deque) else v
            for k, v in history.items()
        }
    
    fig, axes = plt.subplots(3, 2, figsize=(18, 14))
    fig.suptitle(
        'Curriculum Learning Progress', 
        fontsize=16, 
        fontweight='bold'
    )
    
    episodes = history['stage_episodes']
    stage_boundaries = history['stage_boundaries']
    
    # 1. Episode rewards
    ax = axes[0, 0]
    ax.plot(episodes, history['episode_rewards'], alpha=0.3, label='Episode Reward')
    
    if len(history['episode_rewards']) >= 50:
        window = 50
        ma = np.convolve(
            history['episode_rewards'], 
            np.ones(window)/window, 
            mode='valid'
        )
        ax.plot(
            episodes[window-1:], ma, 
            linewidth=2, color='red', label=f'MA({window})'
        )
    
    for boundary in stage_boundaries[1:]:
        ax.axvline(x=boundary, color='black', linestyle='--', alpha=0.5)
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title('Episode Rewards')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Delivery rate
    ax = axes[0, 1]
    ax.plot(episodes, history['delivery_rates'], alpha=0.6, color='green')
    
    for boundary in stage_boundaries[1:]:
        ax.axvline(x=boundary, color='black', linestyle='--', alpha=0.5)
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Delivery Rate')
    ax.set_title('Event Delivery Rate')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    
    # 3. Energy efficiency
    ax = axes[1, 0]
    ax.plot(episodes, history['energy_efficiency'], alpha=0.6, color='orange')
    ax.axhline(y=1.0, color='r', linestyle='--', label='Neutral')
    
    for boundary in stage_boundaries[1:]:
        ax.axvline(x=boundary, color='black', linestyle='--', alpha=0.5)
    
    ax.set_xlabel('Episode')
    ax.set_ylabel('Efficiency')
    ax.set_title('Energy Efficiency')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Policy loss
    ax = axes[1, 1]
    if history['policy_loss']:
        ax.plot(history['policy_loss'], alpha=0.6)
        ax.set_xlabel('Update')
        ax.set_ylabel('Loss')
        ax.set_title('Policy Loss')
        ax.grid(True, alpha=0.3)
    
    # 5. Value loss
    ax = axes[2, 0]
    if history['value_loss']:
        ax.plot(history['value_loss'], alpha=0.6)
        ax.set_xlabel('Update')
        ax.set_ylabel('Loss')
        ax.set_title('Value Loss')
        ax.grid(True, alpha=0.3)
    
    # 6. Entropy
    ax = axes[2, 1]
    if history['entropy']:
        ax.plot(history['entropy'], alpha=0.6)
        ax.set_xlabel('Update')
        ax.set_ylabel('Entropy')
        ax.set_title('Policy Entropy')
        ax.grid(True, alpha=0.3)
    
    # Stage names
    stage_names = []
    for name in history['stage_names']:
        if name not in stage_names:
            stage_names.append(name)
    
    fig.text(
        0.5, 0.02, 
        f'Stages: {" → ".join(stage_names)}',
        ha='center', fontsize=12, style='italic'
    )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Curriculum plot saved to {save_path}")
    
    plt.show()

logger.info("✅ Part 5: Training Loop and Curriculum loaded successfully")
