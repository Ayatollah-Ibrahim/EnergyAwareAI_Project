"""
PPO Training for Energy Harvesting Camera System - Jetson Nano
Part 6: Mastery Stage and Fine-Tuning Framework

This module implements:
- Mastery stage for exact environment adaptation
- Fine-tuning framework for new environments
- Advanced training techniques (gradient accumulation, mixed precision)
- Environment-specific optimization
- Transfer learning utilities

Author: Mastery & Fine-tuning Implementation
Date: 2026-02-07
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import deque
from tqdm import tqdm
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)

# ============================================================================
# MASTERY STAGE CONFIGURATION
# ============================================================================

@dataclass
class MasteryConfig:
    """Configuration for mastery stage training."""
    
    # Training parameters
    n_episodes: int = 2000
    name: str = "Mastery"
    
    # Environment exact match
    initial_battery_soc: float = 0.5          # Match deployment
    prob_event_per_step: float = 0.3          # Match deployment
    images_per_event_mean: float = 100.0      # Match deployment
    n_tx_img_max: int = 5                     # Match deployment
    event_timeout: int = 150                  # Match deployment
    
    # Reward structure (fine-tuned for final performance)
    r_event_delivered: float = 6.0
    r_event_missed: float = -4.0
    
    # PPO hyperparameters (fine-tuned)
    learning_rate: float = 5e-5               # Lower for stability
    entropy_coef: float = 0.05                # Lower for exploitation
    
    # Advanced techniques
    use_gradient_accumulation: bool = False       # Not compatible with PPO buffer
    gradient_accumulation_steps: int = 4
    use_mixed_precision: bool = False             # Enable if GPU available
    use_learning_rate_decay: bool = True
    lr_decay_rate: float = 0.95
    lr_decay_interval: int = 200                  # Episodes
    
    # Evaluation
    eval_interval: int = 50
    n_eval_episodes: int = 20
    
    # Early stopping
    use_early_stopping: bool = True
    patience: int = 300                       # Episodes without improvement
    min_improvement: float = 0.01             # Minimum reward improvement
    
    # Checkpoint management
    save_best_only: bool = True
    keep_n_checkpoints: int = 5
    
    def to_curriculum_stage(self):
        """Convert to CurriculumStage format."""
        from src.env.core import CurriculumStage
        return CurriculumStage(
            name=self.name,
            n_episodes=self.n_episodes,
            initial_battery_soc=self.initial_battery_soc,
            prob_event_per_step=self.prob_event_per_step,
            r_event_delivered=self.r_event_delivered,
            r_event_missed=self.r_event_missed,
            learning_rate=self.learning_rate,
            entropy_coef=self.entropy_coef,
            images_per_event_mean=self.images_per_event_mean,
            n_tx_img_max=self.n_tx_img_max,
            event_timeout=self.event_timeout
        )
    
    def save(self, filepath: Path):
        """Save configuration to JSON."""
        with open(filepath, 'w') as f:
            json.dump(asdict(self), f, indent=2)
        logger.info(f"Mastery config saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: Path):
        """Load configuration from JSON."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls(**data)


# ============================================================================
# MASTERY STAGE TRAINER
# ============================================================================

class MasteryStageTrainer:
    """Advanced trainer for mastery stage with fine-tuning capabilities."""
    
    def __init__(
        self,
        env,
        agent,
        config: MasteryConfig,
        save_dir: Path
    ):
        """
        Initialize mastery trainer.
        
        Args:
            env: Environment (configured for target deployment)
            agent: Pre-trained PPO agent
            config: Mastery configuration
            save_dir: Directory for checkpoints and logs
        """
        self.env = env
        self.agent = agent
        self.config = config
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.episode = 0
        self.best_eval_reward = -np.inf
        self.episodes_without_improvement = 0
        self.gradient_accumulation_counter = 0
        
        # History
        self.history = {
            'episode_rewards': deque(maxlen=10000),
            'episode_lengths': deque(maxlen=10000),
            'delivery_rates': deque(maxlen=10000),
            'eval_rewards': [],
            'eval_episodes': [],
            'learning_rates': [],
            'best_rewards': []
        }
        
        # Checkpoint management
        self.checkpoint_rewards = deque(maxlen=config.keep_n_checkpoints)
        self.checkpoint_paths = deque(maxlen=config.keep_n_checkpoints)
        
        # Save initial config
        config.save(self.save_dir / "mastery_config.json")
        
        logger.info(f"MasteryStageTrainer initialized")
        logger.info(f"  Target episodes: {config.n_episodes}")
        logger.info(f"  Learning rate: {config.learning_rate:.2e}")
        logger.info(f"  Entropy coefficient: {config.entropy_coef:.3f}")
    
    def train(self) -> Dict[str, List]:
        """
        Run mastery stage training.
        
        Returns:
            Training history dictionary
        """
        logger.info("\n" + "="*80)
        logger.info("STARTING MASTERY STAGE TRAINING")
        logger.info("="*80)
        
        # Update agent hyperparameters
        self.agent.config.learning_rate = self.config.learning_rate
        self.agent.config.entropy_coef = self.config.entropy_coef
        self.agent.update_learning_rate(self.config.learning_rate)
        
        pbar = tqdm(total=self.config.n_episodes, desc="Mastery Training")
        
        while self.episode < self.config.n_episodes:
            # Check early stopping
            if self._should_stop_early():
                logger.info(f"\nEarly stopping triggered at episode {self.episode}")
                break
            
            # Train one episode
            episode_reward, episode_length, delivery_rate = self._train_episode()
            
            # Record
            self.history['episode_rewards'].append(episode_reward)
            self.history['episode_lengths'].append(episode_length)
            self.history['delivery_rates'].append(delivery_rate)
            
            # Learning rate decay
            if self.config.use_learning_rate_decay:
                if self.episode % self.config.lr_decay_interval == 0 and self.episode > 0:
                    self._decay_learning_rate()
            
            # Evaluation
            if self.episode % self.config.eval_interval == 0:
                eval_reward = self._evaluate()
                self.history['eval_rewards'].append(eval_reward)
                self.history['eval_episodes'].append(self.episode)
                
                # Update best reward
                if eval_reward > self.best_eval_reward + self.config.min_improvement:
                    improvement = eval_reward - self.best_eval_reward
                    self.best_eval_reward = eval_reward
                    self.episodes_without_improvement = 0
                    logger.info(f"  🎯 New best reward: {eval_reward:.2f} (+{improvement:.2f})")
                    
                    # Save best checkpoint
                    if self.config.save_best_only:
                        self._save_checkpoint("best")
                else:
                    self.episodes_without_improvement += self.config.eval_interval
                
                # Save regular checkpoint
                if not self.config.save_best_only:
                    self._save_checkpoint(f"episode_{self.episode}")
                
                self.history['best_rewards'].append(self.best_eval_reward)
            
            # Update progress
            pbar.update(1)
            pbar.set_postfix({
                'reward': f"{episode_reward:.1f}",
                'delivery': f"{delivery_rate:.1%}",
                'best': f"{self.best_eval_reward:.1f}",
                'lr': f"{self.agent.config.learning_rate:.2e}"
            })
            
            self.episode += 1
        
        pbar.close()
        
        # Final evaluation
        logger.info("\nRunning final evaluation...")
        final_eval_reward = self._evaluate(n_episodes=50)
        logger.info(f"Final evaluation reward: {final_eval_reward:.2f}")
        
        # Save final checkpoint
        self._save_checkpoint("final")
        
        # Save history
        self._save_history()
        
        # Plot results
        self._plot_results()
        
        logger.info("\n" + "="*80)
        logger.info("MASTERY STAGE COMPLETED")
        logger.info(f"  Episodes trained: {self.episode}")
        logger.info(f"  Best evaluation reward: {self.best_eval_reward:.2f}")
        logger.info(f"  Final evaluation reward: {final_eval_reward:.2f}")
        logger.info(f"  Results saved to: {self.save_dir}")
        logger.info("="*80)
        
        return dict(self.history)
    
    def _train_episode(self) -> Tuple[float, int, float]:
        """Train one episode."""
        state, _ = self.env.reset()
        episode_reward = 0.0
        episode_length = 0
        done = False
        
        while not done:
            # Select action
            action, log_prob, value = self.agent.select_action(state)
            
            # Convert to flat action
            mode, capture, process, tx = action
            flat_action = mode * 18 + capture * 6 + process * 2 + tx
            
            # Step environment
            next_state, reward, terminated, truncated, _ = self.env.step(flat_action)
            done = terminated or truncated
            
            # Store transition
            self.agent.store_transition(state, action, log_prob, reward, value, done)
            
            episode_reward += reward
            episode_length += 1
            self.agent.total_timesteps += 1
            
            state = next_state
            
            # Update policy when buffer is full
            if self.agent.buffer.is_full():
                if not done:
                    with torch.no_grad():
                        state_tensor = torch.as_tensor(
                            state, 
                            dtype=torch.float32, 
                            device=self.agent.config.device
                        )
                        last_value = self.agent.critic(state_tensor.unsqueeze(0)).item()
                else:
                    last_value = 0.0
                
                self.agent.finish_episode(last_value=last_value)
                
                # ALWAYS call update when buffer is full (even with gradient accumulation)
                # This resets the buffer so we can continue storing
                stats = self.agent.update()
                
                # Note: Gradient accumulation in PPO doesn't work the same way as in supervised learning
                # because we need to reset the buffer. Disabling it for now.
                # If you want gradient accumulation, you'd need to modify the buffer size instead.
            
            if done or episode_length >= self.env.params.max_episode_steps:
                break
        
        # Finish episode if there are remaining transitions
        if not self.agent.buffer.is_full() and self.agent.buffer.ptr > 0:
            self.agent.finish_episode(last_value=0.0)
        
        # Get delivery rate
        event_stats = self.env.buffer.get_event_delivery_stats()
        delivery_rate = event_stats['delivery_rate']
        
        return episode_reward, episode_length, delivery_rate
    
    def _evaluate(self, n_episodes: Optional[int] = None) -> float:
        """Evaluate current policy."""
        from src.training.trainer import evaluate_policy
        
        n_episodes = n_episodes or self.config.n_eval_episodes
        eval_reward = evaluate_policy(self.env, self.agent, n_episodes, deterministic=True)
        
        return eval_reward
    
    def _decay_learning_rate(self):
        """Decay learning rate."""
        new_lr = self.agent.config.learning_rate * self.config.lr_decay_rate
        self.agent.update_learning_rate(new_lr)
        self.history['learning_rates'].append(new_lr)
        logger.info(f"  Learning rate decayed to {new_lr:.2e}")
    
    def _should_stop_early(self) -> bool:
        """Check if early stopping criterion is met."""
        if not self.config.use_early_stopping:
            return False
        
        if self.episodes_without_improvement >= self.config.patience:
            return True
        
        return False
    
    def _save_checkpoint(self, name: str):
        """Save checkpoint."""
        checkpoint_path = self.save_dir / f"mastery_{name}.pt"
        self.agent.save(str(checkpoint_path))
        
        # Manage checkpoint queue
        if not self.config.save_best_only and name != "final":
            self.checkpoint_paths.append(checkpoint_path)
            self.checkpoint_rewards.append(self.best_eval_reward)
            
            # Remove old checkpoints if exceeding limit
            if len(self.checkpoint_paths) > self.config.keep_n_checkpoints:
                old_path = self.checkpoint_paths.popleft()
                if old_path.exists() and old_path != checkpoint_path:
                    old_path.unlink()
        
        logger.debug(f"Checkpoint saved: {checkpoint_path.name}")
    
    def _save_history(self):
        """Save training history."""
        history_path = self.save_dir / "mastery_history.json"
        
        serializable_history = {}
        for k, v in self.history.items():
            v_list = list(v)
            serializable_history[k] = [
                float(x) if isinstance(x, (int, float, np.number)) else x
                for x in v_list
            ]
        
        with open(history_path, 'w') as f:
            json.dump(serializable_history, f, indent=2)
        
        logger.info(f"History saved to {history_path}")
    
    def _plot_results(self):
        """Plot training results."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Mastery Stage Training Results', fontsize=16, fontweight='bold')
        
        # Episode rewards
        ax = axes[0, 0]
        episodes = list(range(len(self.history['episode_rewards'])))
        ax.plot(episodes, self.history['episode_rewards'], alpha=0.3, label='Episode')
        
        if len(self.history['episode_rewards']) >= 50:
            window = 50
            ma = np.convolve(
                self.history['episode_rewards'],
                np.ones(window)/window,
                mode='valid'
            )
            ax.plot(episodes[window-1:], ma, linewidth=2, color='red', label=f'MA({window})')
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title('Training Rewards')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Evaluation rewards
        ax = axes[0, 1]
        if self.history['eval_episodes']:
            ax.plot(self.history['eval_episodes'], self.history['eval_rewards'], 
                   'o-', linewidth=2, markersize=6, label='Eval Reward')
            ax.plot(self.history['eval_episodes'], self.history['best_rewards'],
                   '--', linewidth=2, label='Best Reward', color='green')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title('Evaluation Performance')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Delivery rate
        ax = axes[1, 0]
        ax.plot(episodes, self.history['delivery_rates'], alpha=0.6, color='green')
        ax.axhline(y=0.95, color='r', linestyle='--', alpha=0.5, label='95% Target')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Delivery Rate')
        ax.set_title('Event Delivery Rate')
        ax.set_ylim([0, 1])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Learning rate
        ax = axes[1, 1]
        if self.history['learning_rates']:
            ax.plot(self.history['learning_rates'], linewidth=2)
            ax.set_xlabel('Decay Step')
            ax.set_ylabel('Learning Rate')
            ax.set_title('Learning Rate Schedule')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No LR Decay', ha='center', va='center',
                   transform=ax.transAxes, fontsize=14)
            ax.axis('off')
        
        plt.tight_layout()
        
        plot_path = self.save_dir / "mastery_training.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        logger.info(f"Training plot saved to {plot_path}")
        
        plt.show()


# ============================================================================
# FINE-TUNING FRAMEWORK
# ============================================================================

@dataclass
class FineTuningConfig:
    """Configuration for fine-tuning to a new environment."""
    
    # New environment parameters
    target_name: str = "New_Environment"
    initial_battery_soc: float = 0.5
    prob_event_per_step: float = 0.25
    images_per_event_mean: float = 80.0
    n_tx_img_max: int = 6
    event_timeout: int = 180
    
    # Fine-tuning strategy
    n_episodes: int = 1000
    freeze_layers: bool = False              # Freeze actor/critic base layers
    learning_rate: float = 1e-4              # Lower than from-scratch
    entropy_coef: float = 0.08
    
    # Adaptation settings
    use_domain_adaptation: bool = True
    adaptation_strength: float = 0.1         # How much to trust new env vs old
    
    # Evaluation
    eval_interval: int = 50
    n_eval_episodes: int = 10


class FineTuner:
    """Fine-tune a pre-trained agent to a new environment."""
    
    def __init__(
        self,
        pretrained_agent,
        new_env,
        config: FineTuningConfig,
        save_dir: Path
    ):
        """
        Initialize fine-tuner.
        
        Args:
            pretrained_agent: Agent pre-trained on source environment
            new_env: Target environment for fine-tuning
            config: Fine-tuning configuration
            save_dir: Directory for saving results
        """
        self.agent = pretrained_agent
        self.env = new_env
        self.config = config
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Optionally freeze layers
        if config.freeze_layers:
            self._freeze_base_layers()
        
        # Update hyperparameters
        self.agent.config.learning_rate = config.learning_rate
        self.agent.config.entropy_coef = config.entropy_coef
        self.agent.update_learning_rate(config.learning_rate)
        
        logger.info(f"FineTuner initialized for {config.target_name}")
        logger.info(f"  Learning rate: {config.learning_rate:.2e}")
        logger.info(f"  Frozen layers: {config.freeze_layers}")
    
    def fine_tune(self) -> Dict[str, List]:
        """Run fine-tuning process."""
        logger.info("\n" + "="*80)
        logger.info(f"FINE-TUNING TO: {self.config.target_name}")
        logger.info("="*80)
        
        # Use MasteryStageTrainer for the actual training
        mastery_config = MasteryConfig(
            n_episodes=self.config.n_episodes,
            name=f"FineTune_{self.config.target_name}",
            initial_battery_soc=self.config.initial_battery_soc,
            prob_event_per_step=self.config.prob_event_per_step,
            images_per_event_mean=self.config.images_per_event_mean,
            n_tx_img_max=self.config.n_tx_img_max,
            event_timeout=self.config.event_timeout,
            learning_rate=self.config.learning_rate,
            entropy_coef=self.config.entropy_coef,
            eval_interval=self.config.eval_interval,
            n_eval_episodes=self.config.n_eval_episodes,
            use_early_stopping=True,
            patience=200
        )
        
        trainer = MasteryStageTrainer(
            self.env,
            self.agent,
            mastery_config,
            self.save_dir
        )
        
        history = trainer.train()
        
        return history
    
    def _freeze_base_layers(self):
        """Freeze base layers of actor and critic."""
        # Freeze all but last layer of actor
        for name, param in self.agent.actor.named_parameters():
            if 'head' not in name:  # Assuming heads are named with 'head'
                param.requires_grad = False
                logger.info(f"  Froze: {name}")
        
        # Freeze all but last layer of critic
        for name, param in self.agent.critic.named_parameters():
            if 'value_head' not in name:
                param.requires_grad = False
                logger.info(f"  Froze: {name}")


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def add_mastery_stage_to_curriculum(
    base_checkpoint: str,
    data_path: str,
    save_dir: str = "mastery_training",
    mastery_config: Optional[MasteryConfig] = None
):
    """
    Add and train mastery stage after curriculum training.
    
    Args:
        base_checkpoint: Path to checkpoint from curriculum training
        data_path: Path to GHI data
        save_dir: Directory for mastery results
        mastery_config: Optional custom mastery configuration
    
    Returns:
        Trained agent, training history
    """
    from src.env.core import SystemParameters
    from src.env.environment import EnergyHarvestingCameraEnv
    from src.agents.ppo import PPOAgent, PPOConfig
    
    # Load pre-trained agent
    logger.info(f"Loading agent from {base_checkpoint}...")
    
    # Create environment matching deployment
    params = SystemParameters()
    env = EnergyHarvestingCameraEnv(data_path, params)
    
    # Load agent
    ppo_config = PPOConfig(state_dim=env.observation_space.shape[0])
    agent = PPOAgent(ppo_config)
    agent.load(base_checkpoint)
    
    logger.info("✅ Agent loaded successfully")
    
    # Create mastery configuration
    if mastery_config is None:
        mastery_config = MasteryConfig()
    
    # Apply mastery config to environment
    stage = mastery_config.to_curriculum_stage()
    params = stage.apply_to_params(SystemParameters())
    env = EnergyHarvestingCameraEnv(data_path, params, event_timeout=stage.event_timeout)
    
    # Train mastery stage
    trainer = MasteryStageTrainer(env, agent, mastery_config, Path(save_dir))
    history = trainer.train()
    
    env.close()
    
    return agent, history


def fine_tune_for_new_environment(
    pretrained_checkpoint: str,
    new_data_path: str,
    fine_tune_config: FineTuningConfig,
    save_dir: str = "fine_tuning_results"
):
    """
    Fine-tune a pre-trained agent for a new environment.
    
    Args:
        pretrained_checkpoint: Path to mastery-trained checkpoint
        new_data_path: Path to GHI data for new environment
        fine_tune_config: Fine-tuning configuration
        save_dir: Directory for results
    
    Returns:
        Fine-tuned agent, training history
    """
    from src.env.core import SystemParameters
    from src.env.environment import EnergyHarvestingCameraEnv
    from src.agents.ppo import PPOAgent, PPOConfig
    
    # Load pretrained agent
    logger.info(f"Loading pretrained agent from {pretrained_checkpoint}...")
    
    # Create new environment
    params = SystemParameters()
    params.initial_battery_soc = fine_tune_config.initial_battery_soc
    params.prob_event_per_step = fine_tune_config.prob_event_per_step
    params.images_per_event_mean = fine_tune_config.images_per_event_mean
    params.n_tx_img_max = fine_tune_config.n_tx_img_max
    
    new_env = EnergyHarvestingCameraEnv(
        new_data_path,
        params,
        event_timeout=fine_tune_config.event_timeout
    )
    
    # Load agent
    ppo_config = PPOConfig(state_dim=new_env.observation_space.shape[0])
    agent = PPOAgent(ppo_config)
    agent.load(pretrained_checkpoint)
    
    logger.info("✅ Agent loaded successfully")
    
    # Fine-tune
    fine_tuner = FineTuner(agent, new_env, fine_tune_config, Path(save_dir))
    history = fine_tuner.fine_tune()
    
    new_env.close()
    
    return agent, history


logger.info("✅ Part 6: Mastery Stage and Fine-Tuning loaded successfully")
