"""
PPO Training for Energy Harvesting Camera System - Jetson Nano
Part 4: PPO Neural Networks and Agent

This module implements:
- Hierarchical Actor network
- Critic (value) network
- PPO Agent with training logic
- Rollout buffer for experience collection
- Advantage estimation (GAE)

Author: Revised Implementation
Date: 2026-02-06
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from typing import Tuple, Dict, Optional
from dataclasses import dataclass
from collections import deque
from pathlib import Path
import logging
import json

logger = logging.getLogger(__name__)

# ============================================================================
# PPO CONFIGURATION
# ============================================================================

@dataclass
class PPOConfig:
    """PPO hyperparameters with numerical stability."""
    
    # Network architecture
    state_dim: int = 44
    hidden_dim: int = 256
    n_hidden_layers: int = 2
    
    # Action dimensions
    n_modes: int = 3      # Sleep modes
    n_capture: int = 3    # Capture modes
    n_process: int = 3    # Processing modes
    n_tx: int = 2         # Transmission modes
    
    # Learning parameters
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    
    # PPO clipping
    clip_epsilon: float = 0.2
    clip_epsilon_vf: float = 0.2  # Value function clipping
    
    # Loss coefficients
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.1
    max_grad_norm: float = 0.5
    
    # Training parameters
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 5
    normalize_advantages: bool = True
    use_gae: bool = True
    
    # Reward normalization
    normalize_rewards: bool = True
    reward_norm_clip: float = 10.0
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Logging
    log_interval: int = 10
    save_interval: int = 100
    
    def validate(self) -> None:
        """Validate configuration."""
        assert self.state_dim > 0
        assert self.hidden_dim > 0
        assert 0 < self.gamma <= 1
        assert self.n_steps > 0
        logger.info(f"PPO configuration validated (device: {self.device})")
    
    def update_for_stage(self, stage) -> None:  # CurriculumStage
        """Update hyperparameters for curriculum stage."""
        self.learning_rate = stage.learning_rate
        self.entropy_coef = stage.entropy_coef
        logger.info(
            f"PPO config updated: LR={self.learning_rate:.1e}, "
            f"Entropy={self.entropy_coef:.2f}"
        )

# ============================================================================
# ROLLOUT BUFFER
# ============================================================================

class RolloutBuffer:
    """Experience replay buffer for PPO."""
    
    def __init__(self, buffer_size: int, state_dim: int, device: str = "cpu"):
        """
        Initialize rollout buffer.
        
        Args:
            buffer_size: Maximum buffer capacity
            state_dim: Observation dimension
            device: Torch device
        """
        self.buffer_size = buffer_size
        self.state_dim = state_dim
        self.device = device
        self.reset()
    
    def reset(self) -> None:
        """Reset buffer to empty state."""
        self.states = np.zeros((self.buffer_size, self.state_dim), dtype=np.float32)
        self.actions_mode = np.zeros(self.buffer_size, dtype=np.int64)
        self.actions_capture = np.zeros(self.buffer_size, dtype=np.int64)
        self.actions_process = np.zeros(self.buffer_size, dtype=np.int64)
        self.actions_tx = np.zeros(self.buffer_size, dtype=np.int64)
        self.log_probs = np.zeros(self.buffer_size, dtype=np.float32)
        self.rewards = np.zeros(self.buffer_size, dtype=np.float32)
        self.values = np.zeros(self.buffer_size, dtype=np.float32)
        self.dones = np.zeros(self.buffer_size, dtype=np.float32)
        self.advantages = np.zeros(self.buffer_size, dtype=np.float32)
        self.returns = np.zeros(self.buffer_size, dtype=np.float32)
        self.ptr = 0
        self.path_start_idx = 0
    
    def store(
        self,
        state: np.ndarray,
        action_mode: int,
        action_capture: int,
        action_process: int,
        action_tx: int,
        log_prob: float,
        reward: float,
        value: float,
        done: bool
    ) -> None:
        """Store one transition."""
        assert self.ptr < self.buffer_size
        
        self.states[self.ptr] = state
        self.actions_mode[self.ptr] = action_mode
        self.actions_capture[self.ptr] = action_capture
        self.actions_process[self.ptr] = action_process
        self.actions_tx[self.ptr] = action_tx
        self.log_probs[self.ptr] = log_prob
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.dones[self.ptr] = done
        self.ptr += 1
    
    def finish_path(
        self, 
        last_value: float = 0.0, 
        gamma: float = 0.99,
        gae_lambda: float = 0.95
    ) -> None:
        """Compute advantages and returns using GAE."""
        path_slice = slice(self.path_start_idx, self.ptr)
        rewards = np.append(self.rewards[path_slice], last_value)
        values = np.append(self.values[path_slice], last_value)
        
        # TD residuals
        deltas = rewards[:-1] + gamma * values[1:] - values[:-1]
        
        # GAE
        advantages = self._discount_cumsum(deltas, gamma * gae_lambda)
        returns = advantages + values[:-1]
        
        self.advantages[path_slice] = advantages
        self.returns[path_slice] = returns
        self.path_start_idx = self.ptr
    
    def _discount_cumsum(self, x, discount):
        """Compute discounted cumulative sum with numerical stability."""
        cumsum = np.zeros_like(x)
        cumsum[-1] = x[-1]
        for t in reversed(range(len(x) - 1)):
            cumsum[t] = x[t] + discount * cumsum[t + 1]
            cumsum[t] = np.clip(cumsum[t], -1e6, 1e6)  # Prevent explosion
        return cumsum
    
    def get(self) -> Dict[str, torch.Tensor]:
        """Get all buffer data as tensors with normalized advantages."""
        assert self.ptr == self.buffer_size
        
        # Normalize advantages
        adv_mean = self.advantages.mean()
        adv_std = self.advantages.std() + 1e-8
        advantages_normalized = (self.advantages - adv_mean) / adv_std
        
        data = {
            'states': torch.as_tensor(
                self.states, dtype=torch.float32, device=self.device
            ),
            'actions_mode': torch.as_tensor(
                self.actions_mode, dtype=torch.long, device=self.device
            ),
            'actions_capture': torch.as_tensor(
                self.actions_capture, dtype=torch.long, device=self.device
            ),
            'actions_process': torch.as_tensor(
                self.actions_process, dtype=torch.long, device=self.device
            ),
            'actions_tx': torch.as_tensor(
                self.actions_tx, dtype=torch.long, device=self.device
            ),
            'log_probs_old': torch.as_tensor(
                self.log_probs, dtype=torch.float32, device=self.device
            ),
            'advantages': torch.as_tensor(
                advantages_normalized, dtype=torch.float32, device=self.device
            ),
            'returns': torch.as_tensor(
                self.returns, dtype=torch.float32, device=self.device
            ),
            'values_old': torch.as_tensor(
                self.values, dtype=torch.float32, device=self.device
            )
        }
        
        return data
    
    def is_full(self) -> bool:
        """Check if buffer is full."""
        return self.ptr >= self.buffer_size

# ============================================================================
# ACTOR NETWORK
# ============================================================================

class HierarchicalActor(nn.Module):
    """
    Hierarchical actor network with separate heads.
    
    Architecture:
        Shared network → 4 independent heads (sleep, capture, process, tx)
    """
    
    def __init__(self, config: PPOConfig):
        """
        Initialize actor.
        
        Args:
            config: PPO configuration
        """
        super().__init__()
        self.config = config
        
        # Shared network
        layers = []
        input_dim = config.state_dim
        
        for _ in range(config.n_hidden_layers):
            layers.append(nn.Linear(input_dim, config.hidden_dim))
            layers.append(nn.ReLU())
            input_dim = config.hidden_dim
        
        self.shared_net = nn.Sequential(*layers)
        
        # Action heads
        self.mode_head = nn.Linear(config.hidden_dim, config.n_modes)
        self.capture_head = nn.Linear(config.hidden_dim, config.n_capture)
        self.process_head = nn.Linear(config.hidden_dim, config.n_process)
        self.tx_head = nn.Linear(config.hidden_dim, config.n_tx)
        
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass.
        
        Args:
            state: Observation tensor
        
        Returns:
            Tuple of (mode_logits, capture_logits, process_logits, tx_logits)
        """
        features = self.shared_net(state)
        
        mode_logits = self.mode_head(features)
        capture_logits = self.capture_head(features)
        process_logits = self.process_head(features)
        tx_logits = self.tx_head(features)
        
        return mode_logits, capture_logits, process_logits, tx_logits
    
    def get_action_and_log_prob(
        self, 
        state: torch.Tensor, 
        deterministic: bool = False
    ) -> Tuple:
        """
        Sample action from policy.
        
        Args:
            state: Observation tensor
            deterministic: Use greedy action selection
        
        Returns:
            Tuple of (action, log_prob)
        """
        state = state.unsqueeze(0)
        mode_logits, capture_logits, process_logits, tx_logits = self.forward(state)
        
        # Sample mode
        mode_dist = Categorical(logits=mode_logits)
        if deterministic:
            mode = mode_dist.probs.argmax(dim=-1)
        else:
            mode = mode_dist.sample()
        mode_log_prob = mode_dist.log_prob(mode)
        
        mode_val = mode.item()
        
        # If active mode, sample other actions
        if mode_val == 0:
            capture_dist = Categorical(logits=capture_logits)
            process_dist = Categorical(logits=process_logits)
            tx_dist = Categorical(logits=tx_logits)
            
            if deterministic:
                capture = capture_dist.probs.argmax(dim=-1)
                process = process_dist.probs.argmax(dim=-1)
                tx = tx_dist.probs.argmax(dim=-1)
            else:
                capture = capture_dist.sample()
                process = process_dist.sample()
                tx = tx_dist.sample()
            
            capture_log_prob = capture_dist.log_prob(capture)
            process_log_prob = process_dist.log_prob(process)
            tx_log_prob = tx_dist.log_prob(tx)
            
            total_log_prob = (
                mode_log_prob + capture_log_prob + 
                process_log_prob + tx_log_prob
            )
            
            capture_val = capture.item()
            process_val = process.item()
            tx_val = tx.item()
        else:
            # Sleep/Halt: no other actions
            capture_val = 0
            process_val = 0
            tx_val = 0
            total_log_prob = mode_log_prob
        
        action = (mode_val, capture_val, process_val, tx_val)
        return action, total_log_prob.item()
    
    def evaluate_actions(
        self,
        states: torch.Tensor,
        actions_mode: torch.Tensor,
        actions_capture: torch.Tensor,
        actions_process: torch.Tensor,
        actions_tx: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate actions for policy update.
        
        Args:
            states: Batch of observations
            actions_*: Batch of actions
        
        Returns:
            Tuple of (log_probs, entropy)
        """
        mode_logits, capture_logits, process_logits, tx_logits = self.forward(states)
        
        # Mode distribution
        mode_dist = Categorical(logits=mode_logits)
        mode_log_probs = mode_dist.log_prob(actions_mode)
        mode_entropy = mode_dist.entropy()
        
        # Active mask (only active mode uses other actions)
        active_mask = (actions_mode == 0).float()
        
        # Other distributions
        capture_dist = Categorical(logits=capture_logits)
        process_dist = Categorical(logits=process_logits)
        tx_dist = Categorical(logits=tx_logits)
        
        capture_log_probs = capture_dist.log_prob(actions_capture) * active_mask
        process_log_probs = process_dist.log_prob(actions_process) * active_mask
        tx_log_probs = tx_dist.log_prob(actions_tx) * active_mask
        
        capture_entropy = capture_dist.entropy() * active_mask
        process_entropy = process_dist.entropy() * active_mask
        tx_entropy = tx_dist.entropy() * active_mask
        
        # Total
        total_log_probs = (
            mode_log_probs + capture_log_probs + 
            process_log_probs + tx_log_probs
        )
        total_entropy = (
            mode_entropy + capture_entropy + 
            process_entropy + tx_entropy
        )
        
        return total_log_probs, total_entropy

# ============================================================================
# CRITIC NETWORK
# ============================================================================

class Critic(nn.Module):
    """Critic network for value estimation."""
    
    def __init__(self, config: PPOConfig):
        """
        Initialize critic.
        
        Args:
            config: PPO configuration
        """
        super().__init__()
        self.config = config
        
        # Value network
        layers = []
        input_dim = config.state_dim
        
        for _ in range(config.n_hidden_layers):
            layers.append(nn.Linear(input_dim, config.hidden_dim))
            layers.append(nn.ReLU())
            input_dim = config.hidden_dim
        
        layers.append(nn.Linear(config.hidden_dim, 1))
        
        self.value_net = nn.Sequential(*layers)
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            state: Observation tensor
        
        Returns:
            State value
        """
        return self.value_net(state)

# ============================================================================
# PPO AGENT
# ============================================================================

class PPOAgent:
    """PPO agent with numerical stability fixes."""
    
    def __init__(self, config: PPOConfig):
        """
        Initialize PPO agent.
        
        Args:
            config: PPO configuration
        """
        self.config = config
        config.validate()
        
        # Networks
        self.actor = HierarchicalActor(config).to(config.device)
        self.critic = Critic(config).to(config.device)
        
        # Optimizers
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(), lr=config.learning_rate
        )
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=config.learning_rate
        )
        
        # Rollout buffer
        self.buffer = RolloutBuffer(
            config.n_steps, config.state_dim, config.device
        )
        
        # Reward normalization
        self.reward_mean = 0.0
        self.reward_std = 1.0
        self.reward_history = deque(maxlen=1000)
        
        # Value function monitoring
        self.explained_variance_history = deque(maxlen=100)
        
        # Training state
        self.update_count = 0
        self.total_timesteps = 0
        
        logger.info("PPO agent initialized with numerical stability")
    
    def select_action(
        self, 
        state: np.ndarray, 
        deterministic: bool = False
    ) -> Tuple:
        """
        Select action given observation.
        
        Args:
            state: Observation
            deterministic: Use greedy selection
        
        Returns:
            Tuple of (action, log_prob, value)
        """
        with torch.no_grad():
            state_tensor = torch.as_tensor(
                state, dtype=torch.float32, device=self.config.device
            )
            action, log_prob = self.actor.get_action_and_log_prob(
                state_tensor, deterministic
            )
            value = self.critic(state_tensor.unsqueeze(0)).item()
        
        return action, log_prob, value
    
    def store_transition(
        self,
        state: np.ndarray,
        action: Tuple[int, int, int, int],
        log_prob: float,
        reward: float,
        value: float,
        done: bool
    ) -> None:
        """Store transition in buffer."""
        # Update reward statistics
        self.reward_history.append(reward)
        
        if len(self.reward_history) >= 100:
            self.reward_mean = np.mean(self.reward_history)
            self.reward_std = np.std(self.reward_history) + 1e-8
        
        # Normalize reward
        if self.config.normalize_rewards and self.reward_std > 0:
            reward_normalized = (reward - self.reward_mean) / self.reward_std
            reward_normalized = np.clip(
                reward_normalized, 
                -self.config.reward_norm_clip,
                self.config.reward_norm_clip
            )
        else:
            reward_normalized = reward
        
        mode, capture, process, tx = action
        self.buffer.store(
            state, mode, capture, process, tx, 
            log_prob, reward_normalized, value, done
        )
    
    def finish_episode(self, last_value: float = 0.0) -> None:
        """Finish episode and compute advantages."""
        self.buffer.finish_path(
            last_value, self.config.gamma, self.config.gae_lambda
        )
    
    def update(self) -> Dict[str, float]:
        """Perform PPO update."""
        if not self.buffer.is_full():
            return {}
        
        data = self.buffer.get()
        
        stats = {
            'policy_loss': 0.0,
            'value_loss': 0.0,
            'value_loss_clipped': 0.0,
            'entropy': 0.0,
            'approx_kl': 0.0,
            'clip_fraction': 0.0,
            'explained_variance': 0.0,
            'reward_mean': self.reward_mean,
            'reward_std': self.reward_std
        }
        
        for epoch in range(self.config.n_epochs):
            indices = np.arange(self.config.n_steps)
            np.random.shuffle(indices)
            
            for start in range(0, self.config.n_steps, self.config.batch_size):
                end = start + self.config.batch_size
                mb_indices = indices[start:end]
                
                # Get minibatch
                mb_states = data['states'][mb_indices]
                mb_actions_mode = data['actions_mode'][mb_indices]
                mb_actions_capture = data['actions_capture'][mb_indices]
                mb_actions_process = data['actions_process'][mb_indices]
                mb_actions_tx = data['actions_tx'][mb_indices]
                mb_log_probs_old = data['log_probs_old'][mb_indices]
                mb_advantages = data['advantages'][mb_indices]
                mb_returns = data['returns'][mb_indices]
                mb_values_old = data['values_old'][mb_indices]
                
                # Evaluate actions
                log_probs, entropy = self.actor.evaluate_actions(
                    mb_states, mb_actions_mode, mb_actions_capture,
                    mb_actions_process, mb_actions_tx
                )
                
                values = self.critic(mb_states).squeeze(-1)
                
                # Policy loss
                ratio = torch.exp(log_probs - mb_log_probs_old)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(
                    ratio, 
                    1 - self.config.clip_epsilon,
                    1 + self.config.clip_epsilon
                ) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss with clipping
                value_loss_unclipped = F.mse_loss(values, mb_returns)
                
                values_clipped = mb_values_old + torch.clamp(
                    values - mb_values_old,
                    -self.config.clip_epsilon_vf,
                    self.config.clip_epsilon_vf
                )
                value_loss_clipped = F.mse_loss(values_clipped, mb_returns)
                value_loss = torch.max(value_loss_unclipped, value_loss_clipped)
                
                # Entropy loss
                entropy_loss = -entropy.mean()
                
                # Total loss
                total_loss = (
                    policy_loss +
                    self.config.value_loss_coef * value_loss +
                    self.config.entropy_coef * entropy_loss
                )
                
                # Optimize
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                total_loss.backward()
                
                nn.utils.clip_grad_norm_(
                    self.actor.parameters(), self.config.max_grad_norm
                )
                nn.utils.clip_grad_norm_(
                    self.critic.parameters(), self.config.max_grad_norm
                )
                
                self.actor_optimizer.step()
                self.critic_optimizer.step()
                
                # Compute explained variance
                with torch.no_grad():
                    var_y = torch.var(mb_returns)
                    explained_var = 1 - torch.var(mb_returns - values) / (var_y + 1e-8)
                    self.explained_variance_history.append(explained_var.item())
                
                # Compute statistics
                with torch.no_grad():
                    approx_kl = (mb_log_probs_old - log_probs).mean().item()
                    clipped = (
                        (ratio > 1 + self.config.clip_epsilon) | 
                        (ratio < 1 - self.config.clip_epsilon)
                    )
                    clip_fraction = clipped.float().mean().item()
                
                stats['policy_loss'] += policy_loss.item()
                stats['value_loss'] += value_loss_unclipped.item()
                stats['value_loss_clipped'] += value_loss_clipped.item()
                stats['entropy'] += entropy.mean().item()
                stats['approx_kl'] += approx_kl
                stats['clip_fraction'] += clip_fraction
        
        # Average statistics
        n_updates = self.config.n_epochs * (self.config.n_steps // self.config.batch_size)
        for key in stats:
            if key not in ['reward_mean', 'reward_std']:
                stats[key] /= n_updates
        
        # Explained variance
        if len(self.explained_variance_history) > 0:
            stats['explained_variance'] = np.mean(self.explained_variance_history)
        
        self.buffer.reset()
        self.update_count += 1
        
        return stats
    
    def update_learning_rate(self, new_lr: float) -> None:
        """Update learning rate for both optimizers."""
        for param_group in self.actor_optimizer.param_groups:
            param_group['lr'] = new_lr
        for param_group in self.critic_optimizer.param_groups:
            param_group['lr'] = new_lr
        logger.info(f"Learning rate updated to {new_lr:.1e}")
    
    def save(self, filepath: str) -> None:
        """Save agent checkpoint."""
        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'update_count': self.update_count,
            'total_timesteps': self.total_timesteps,
            'reward_mean': self.reward_mean,
            'reward_std': self.reward_std,
            'reward_history': list(self.reward_history),
            'config': self.config
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, filepath)
        logger.info(f"Agent saved to {filepath}")
    
    def load(self, filepath: str) -> None:
        """Load agent checkpoint."""
        checkpoint = torch.load(
            filepath, 
            map_location=self.config.device,
            weights_only=False
        )
        
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_optimizer.load_state_dict(
            checkpoint['actor_optimizer_state_dict']
        )
        self.critic_optimizer.load_state_dict(
            checkpoint['critic_optimizer_state_dict']
        )
        self.update_count = checkpoint['update_count']
        self.total_timesteps = checkpoint['total_timesteps']
        
        self.reward_mean = checkpoint.get('reward_mean', 0.0)
        self.reward_std = checkpoint.get('reward_std', 1.0)
        if 'reward_history' in checkpoint:
            self.reward_history = deque(
                checkpoint['reward_history'], maxlen=1000
            )
        
        logger.info(f"Agent loaded from {filepath}")

logger.info("✅ Part 4: PPO Networks and Agent loaded successfully")
