# ============================================================================
# DEEP Q-NETWORK (DQN) BASELINE
# ============================================================================
"""
Deep Q-Network (DQN) Baseline for Energy Harvesting Camera Control

This is the STANDARD deep RL baseline for discrete action spaces.
DQN learns a Q-function Q(s, a) that estimates the value of taking
action a in state s.

Key Features:
- Experience replay buffer
- Target network for stability
- Epsilon-greedy exploration
- Double DQN variant
- Dueling DQN architecture option

Reference: Mnih et al., "Human-level control through deep RL" (Nature, 2015)

Author: [Your Name]
Date: 2024
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque, namedtuple
from typing import Tuple, List, Dict, Any, Optional
from dataclasses import dataclass
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class DQNConfig:
    """Configuration for Deep Q-Network."""
    
    # Network architecture
    state_dim: int = 44  # Match observation space
    hidden_dim: int = 256
    n_hidden_layers: int = 2
    
    # Use dueling architecture
    use_dueling: bool = True
    
    # Use double DQN
    use_double_dqn: bool = True
    
    # Learning parameters
    learning_rate: float = 3e-4
    gamma: float = 0.99  # Discount factor
    
    # Exploration
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: int = 10000  # Steps to decay from start to end
    
    # Experience replay
    buffer_size: int = 100000
    batch_size: int = 64
    min_buffer_size: int = 1000  # Start training after this many samples
    
    # Training
    target_update_frequency: int = 1000  # Update target network every N steps
    train_frequency: int = 4  # Train every N steps
    
    # Prioritized replay (optional)
    use_prioritized_replay: bool = False
    priority_alpha: float = 0.6
    priority_beta_start: float = 0.4
    priority_beta_frames: int = 100000
    
    # Gradient clipping
    max_grad_norm: float = 10.0
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Logging
    log_interval: int = 100
    
    def validate(self):
        """Validate configuration."""
        assert self.state_dim > 0
        assert self.hidden_dim > 0
        assert 0 < self.gamma <= 1
        assert 0 <= self.epsilon_end <= self.epsilon_start <= 1
        assert self.batch_size <= self.buffer_size
        logger.info(f"DQN config validated (device: {self.device})")



Transition = namedtuple(
    'Transition',
    ('state', 'action', 'reward', 'next_state', 'done')
)


class ReplayBuffer:
    """Experience replay buffer for DQN."""
    
    def __init__(self, capacity: int):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store
        """
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
    
    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """Add a transition to the buffer."""
        self.buffer.append(
            Transition(state, action, reward, next_state, done)
        )
    
    def sample(self, batch_size: int) -> List[Transition]:
        """Sample a batch of transitions."""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[idx] for idx in indices]
    
    def __len__(self) -> int:
        """Get current buffer size."""
        return len(self.buffer)


class PrioritizedReplayBuffer:
    """
    Prioritized experience replay buffer.
    
    Samples important transitions more frequently.
    Reference: Schaul et al., "Prioritized Experience Replay" (ICLR 2016)
    """
    
    def __init__(self, capacity: int, alpha: float = 0.6):
        """
        Initialize prioritized replay buffer.
        
        Args:
            capacity: Maximum buffer size
            alpha: Prioritization exponent (0 = uniform, 1 = full priority)
        """
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0
    
    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """Add transition with maximum priority."""
        max_priority = self.priorities.max() if self.size > 0 else 1.0
        
        if self.size < self.capacity:
            self.buffer.append(
                Transition(state, action, reward, next_state, done)
            )
            self.size += 1
        else:
            self.buffer[self.position] = \
                Transition(state, action, reward, next_state, done)
        
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
    
    def sample(
        self,
        batch_size: int,
        beta: float = 0.4
    ) -> Tuple[List[Transition], np.ndarray, np.ndarray]:
        """
        Sample batch with priorities.
        
        Args:
            batch_size: Number of samples
            beta: Importance sampling exponent
        
        Returns:
            Tuple of (transitions, weights, indices)
        """
        # Compute sampling probabilities
        priorities = self.priorities[:self.size]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        # Sample indices
        indices = np.random.choice(
            self.size, batch_size, replace=False, p=probabilities
        )
        
        # Get transitions
        transitions = [self.buffer[idx] for idx in indices]
        
        # Compute importance sampling weights
        weights = (self.size * probabilities[indices]) ** (-beta)
        weights /= weights.max()  # Normalize
        
        return transitions, weights, indices
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """Update priorities for sampled transitions."""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
    
    def __len__(self) -> int:
        """Get current buffer size."""
        return self.size



class QNetwork(nn.Module):
    """Standard Q-Network."""
    
    def __init__(self, config: DQNConfig, n_actions: int = 54):
        """
        Initialize Q-Network.
        
        Args:
            config: DQN configuration
            n_actions: Number of discrete actions
        """
        super().__init__()
        
        self.config = config
        self.n_actions = n_actions
        
        # Build network layers
        layers = []
        input_dim = config.state_dim
        
        for _ in range(config.n_hidden_layers):
            layers.append(nn.Linear(input_dim, config.hidden_dim))
            layers.append(nn.ReLU())
            input_dim = config.hidden_dim
        
        self.feature_net = nn.Sequential(*layers)
        self.q_head = nn.Linear(config.hidden_dim, n_actions)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            state: State tensor [batch_size, state_dim]
        
        Returns:
            Q-values [batch_size, n_actions]
        """
        features = self.feature_net(state)
        q_values = self.q_head(features)
        return q_values


class DuelingQNetwork(nn.Module):
    """
    Dueling Q-Network architecture.
    
    Splits Q(s,a) into V(s) + A(s,a) for better learning.
    Reference: Wang et al., "Dueling Network Architectures" (ICML 2016)
    """
    
    def __init__(self, config: DQNConfig, n_actions: int = 54):
        """Initialize Dueling Q-Network."""
        super().__init__()
        
        self.config = config
        self.n_actions = n_actions
        
        # Shared feature network
        layers = []
        input_dim = config.state_dim
        
        for _ in range(config.n_hidden_layers):
            layers.append(nn.Linear(input_dim, config.hidden_dim))
            layers.append(nn.ReLU())
            input_dim = config.hidden_dim
        
        self.feature_net = nn.Sequential(*layers)
        
        # Value stream: V(s)
        self.value_stream = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, 1)
        )
        
        # Advantage stream: A(s,a)
        self.advantage_stream = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, n_actions)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with dueling architecture.
        
        Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        """
        features = self.feature_net(state)
        
        # Value function
        value = self.value_stream(features)
        
        # Advantage function
        advantages = self.advantage_stream(features)
        
        # Combine: Q = V + (A - mean(A))
        # Subtracting mean makes the representation unique
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        
        return q_values



class DQNAgent:
    """
    Deep Q-Network agent.
    
    Implements:
    - Experience replay
    - Target network
    - Epsilon-greedy exploration
    - Double DQN (optional)
    - Dueling architecture (optional)
    - Prioritized replay (optional)
    """
    
    def __init__(self, config: DQNConfig, n_actions: int = 54):
        """
        Initialize DQN agent.
        
        Args:
            config: DQN configuration
            n_actions: Number of discrete actions
        """
        self.config = config
        self.n_actions = n_actions
        config.validate()
        
        # Create Q-networks
        NetworkClass = DuelingQNetwork if config.use_dueling else QNetwork
        
        self.q_network = NetworkClass(config, n_actions).to(config.device)
        self.target_network = NetworkClass(config, n_actions).to(config.device)
        
        # Initialize target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.q_network.parameters(),
            lr=config.learning_rate
        )
        
        # Replay buffer
        if config.use_prioritized_replay:
            self.replay_buffer = PrioritizedReplayBuffer(
                config.buffer_size,
                config.priority_alpha
            )
        else:
            self.replay_buffer = ReplayBuffer(config.buffer_size)
        
        # Training state
        self.steps = 0
        self.episodes = 0
        self.epsilon = config.epsilon_start
        
        # Statistics
        self.training_losses = []
        self.episode_rewards = []
        
        # For baseline compatibility
        self.name = "DQN"
        if config.use_dueling:
            self.name += " (Dueling)"
        if config.use_double_dqn:
            self.name += " (Double)"
        
        self.description = (
            f"Deep Q-Network with {config.n_hidden_layers}-layer "
            f"{config.hidden_dim}-dim network"
        )
        
        # For evaluate_policy compatibility
        self.action_counts = np.zeros(54, dtype=int)
        self.total_actions = 0
        
        logger.info(f"DQN agent initialized: {self.name}")
        logger.info(f"  Network: {sum(p.numel() for p in self.q_network.parameters())} parameters")
        logger.info(f"  Buffer: {config.buffer_size} capacity")
    
    def reset(self):
        """Reset statistics (for baseline compatibility)."""
        self.action_counts = np.zeros(54, dtype=int)
        self.total_actions = 0
    
    def _update_epsilon(self):
        """Update epsilon for exploration."""
        # Linear decay
        self.epsilon = max(
            self.config.epsilon_end,
            self.config.epsilon_start - 
            (self.config.epsilon_start - self.config.epsilon_end) * 
            self.steps / self.config.epsilon_decay
        )
    
    def select_action(
        self,
        state: np.ndarray,
        deterministic: bool = False
    ) -> Tuple[int, float, float]:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state
            deterministic: If True, always take greedy action
        
        Returns:
            Tuple of (action, log_prob=0.0, value=0.0)
            Note: log_prob and value are 0 for compatibility with PPO interface
        """
        
        # Epsilon-greedy exploration
        if not deterministic and np.random.random() < self.epsilon:
            action = np.random.randint(self.n_actions)
        else:
            # Greedy action
            with torch.no_grad():
                state_tensor = torch.as_tensor(
                    state, dtype=torch.float32, device=self.config.device
                ).unsqueeze(0)
                
                q_values = self.q_network(state_tensor)
                action = q_values.argmax(dim=1).item()
        
        # Record action
        self.action_counts[action] += 1
        self.total_actions += 1
        
        # Return in PPO-compatible format
        # (Convert flat action to tuple for environment)
        mode = action // 18
        capture = (action % 18) // 6
        process = (action % 6) // 2
        tx = action % 2
        
        return (mode, capture, process, tx), 0.0, 0.0
    
    def store_transition(
        self,
        state: np.ndarray,
        action: Tuple[int, int, int, int],
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """
        Store transition in replay buffer.
        
        Args:
            state: Current state
            action: Action tuple (mode, capture, process, tx)
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        """
        # Convert action tuple to flat index
        mode, capture, process, tx = action
        flat_action = mode * 18 + capture * 6 + process * 2 + tx
        
        self.replay_buffer.push(state, flat_action, reward, next_state, done)
    
    def train_step(self) -> Optional[float]:
        """
        Perform one training step.
        
        Returns:
            Loss value if training occurred, None otherwise
        """
        
        # Check if buffer has enough samples
        if len(self.replay_buffer) < self.config.min_buffer_size:
            return None
        
        # Sample batch
        if self.config.use_prioritized_replay:
            # Prioritized replay
            beta = min(
                1.0,
                self.config.priority_beta_start + 
                self.steps * (1.0 - self.config.priority_beta_start) / 
                self.config.priority_beta_frames
            )
            
            transitions, weights, indices = self.replay_buffer.sample(
                self.config.batch_size, beta
            )
            
            weights = torch.as_tensor(
                weights, dtype=torch.float32, device=self.config.device
            )
        else:
            # Uniform replay
            transitions = self.replay_buffer.sample(self.config.batch_size)
            weights = torch.ones(
                self.config.batch_size, device=self.config.device
            )
            indices = None
        
        # Unpack batch
        batch = Transition(*zip(*transitions))
        
        states = torch.as_tensor(
            np.array(batch.state), dtype=torch.float32, device=self.config.device
        )
        actions = torch.as_tensor(
            batch.action, dtype=torch.long, device=self.config.device
        ).unsqueeze(1)
        rewards = torch.as_tensor(
            batch.reward, dtype=torch.float32, device=self.config.device
        )
        next_states = torch.as_tensor(
            np.array(batch.next_state), dtype=torch.float32, device=self.config.device
        )
        dones = torch.as_tensor(
            batch.done, dtype=torch.float32, device=self.config.device
        )
        
        # Compute current Q-values
        current_q_values = self.q_network(states).gather(1, actions).squeeze(1)
        
        # Compute target Q-values
        with torch.no_grad():
            if self.config.use_double_dqn:
                # Double DQN: use online network to select action,
                # target network to evaluate
                next_actions = self.q_network(next_states).argmax(dim=1, keepdim=True)
                next_q_values = self.target_network(next_states).gather(1, next_actions).squeeze(1)
            else:
                # Standard DQN: use target network for both
                next_q_values = self.target_network(next_states).max(dim=1)[0]
            
            target_q_values = rewards + self.config.gamma * next_q_values * (1 - dones)
        
        # Compute TD errors
        td_errors = current_q_values - target_q_values
        
        # Compute loss (weighted MSE)
        loss = (weights * td_errors ** 2).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            self.q_network.parameters(),
            self.config.max_grad_norm
        )
        self.optimizer.step()
        
        # Update priorities
        if self.config.use_prioritized_replay:
            priorities = np.abs(td_errors.detach().cpu().numpy()) + 1e-6
            self.replay_buffer.update_priorities(indices, priorities)
        
        # Record loss
        loss_value = loss.item()
        self.training_losses.append(loss_value)
        
        return loss_value
    
    def update_target_network(self):
        """Update target network with current Q-network weights."""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def train(
        self,
        env,
        n_episodes: int,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Train DQN agent.
        
        Args:
            env: Environment
            n_episodes: Number of episodes to train
            verbose: Whether to print progress
        
        Returns:
            Training history
        """
        
        history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'losses': [],
            'epsilon': []
        }
        
        for episode in range(n_episodes):
            state, _ = env.reset()
            episode_reward = 0.0
            episode_length = 0
            done = False
            
            while not done:
                # Select action
                action_tuple, _, _ = self.select_action(state, deterministic=False)
                
                # Convert to flat action
                mode, capture, process, tx = action_tuple
                flat_action = mode * 18 + capture * 6 + process * 2 + tx
                
                # Take step
                next_state, reward, terminated, truncated, _ = env.step(flat_action)
                done = terminated or truncated
                
                # Store transition
                self.store_transition(state, action_tuple, reward, next_state, done)
                
                # Train
                if self.steps % self.config.train_frequency == 0:
                    loss = self.train_step()
                    if loss is not None:
                        history['losses'].append(loss)
                
                # Update target network
                if self.steps % self.config.target_update_frequency == 0:
                    self.update_target_network()
                
                # Update epsilon
                self._update_epsilon()
                
                episode_reward += reward
                episode_length += 1
                self.steps += 1
                state = next_state
            
            # Episode complete
            self.episodes += 1
            history['episode_rewards'].append(episode_reward)
            history['episode_lengths'].append(episode_length)
            history['epsilon'].append(self.epsilon)
            
            # Logging
            if verbose and (episode + 1) % self.config.log_interval == 0:
                recent_rewards = history['episode_rewards'][-100:]
                recent_losses = history['losses'][-100:] if history['losses'] else [0]
                
                logger.info(
                    f"Episode {episode + 1}/{n_episodes} | "
                    f"Reward: {episode_reward:.2f} | "
                    f"Mean(100): {np.mean(recent_rewards):.2f} | "
                    f"Loss: {np.mean(recent_losses):.4f} | "
                    f"Epsilon: {self.epsilon:.3f} | "
                    f"Buffer: {len(self.replay_buffer)}"
                )
        
        return history
    
    def save(self, filepath: str):
        """Save DQN agent."""
        checkpoint = {
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'steps': self.steps,
            'episodes': self.episodes,
            'epsilon': self.epsilon,
            'config': self.config
        }
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, filepath)
        logger.info(f"DQN agent saved to {filepath}")
    
    def load(self, filepath: str):
        """Load DQN agent."""
        checkpoint = torch.load(filepath, map_location=self.config.device, weights_only=False)
        
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.steps = checkpoint['steps']
        self.episodes = checkpoint['episodes']
        self.epsilon = checkpoint['epsilon']
        
        logger.info(f"DQN agent loaded from {filepath}")
    
    def get_action_distribution(self) -> np.ndarray:
        """Get normalized action distribution (for baseline compatibility)."""
        if self.total_actions == 0:
            return np.zeros(54)
        return self.action_counts / self.total_actions



def train_dqn_baseline(
    env,
    config: Optional[DQNConfig] = None,
    n_episodes: int = 5000,
    save_dir: str = "experiments/dqn_baseline",
    verbose: bool = True
) -> Tuple[DQNAgent, Dict]:
    """
    Train DQN baseline from scratch.
    
    Args:
        env: Environment
        config: DQN configuration (defaults to DQNConfig())
        n_episodes: Number of training episodes
        save_dir: Directory to save results
        verbose: Whether to print progress
    
    Returns:
        Tuple of (trained agent, training history)
    """
    
    logger.info("=" * 80)
    logger.info("TRAINING DQN BASELINE")
    logger.info("=" * 80)
    
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # Create agent
    if config is None:
        config = DQNConfig(state_dim=env.observation_space.shape[0])
    
    agent = DQNAgent(config)
    
    # Train
    logger.info(f"\nTraining for {n_episodes} episodes...")
    history = agent.train(env, n_episodes, verbose=verbose)
    
    # Save
    agent.save(Path(save_dir) / "dqn_final.pt")
    
    # Plot learning curve
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Episode rewards
    ax = axes[0, 0]
    ax.plot(history['episode_rewards'], alpha=0.3)
    if len(history['episode_rewards']) >= 100:
        window = 100
        ma = np.convolve(
            history['episode_rewards'],
            np.ones(window) / window,
            mode='valid'
        )
        ax.plot(range(window - 1, len(history['episode_rewards'])), ma,
               linewidth=2, color='red', label=f'MA({window})')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title('DQN Training: Episode Rewards')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Loss
    ax = axes[0, 1]
    if history['losses']:
        ax.plot(history['losses'], alpha=0.5)
        if len(history['losses']) >= 100:
            window = 100
            ma = np.convolve(
                history['losses'],
                np.ones(window) / window,
                mode='valid'
            )
            ax.plot(range(window - 1, len(history['losses'])), ma,
                   linewidth=2, color='red')
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss')
    ax.grid(alpha=0.3)
    
    # Epsilon
    ax = axes[1, 0]
    ax.plot(history['epsilon'])
    ax.set_xlabel('Episode')
    ax.set_ylabel('Epsilon')
    ax.set_title('Exploration Rate')
    ax.grid(alpha=0.3)
    
    # Episode length
    ax = axes[1, 1]
    ax.plot(history['episode_lengths'], alpha=0.3)
    if len(history['episode_lengths']) >= 100:
        window = 100
        ma = np.convolve(
            history['episode_lengths'],
            np.ones(window) / window,
            mode='valid'
        )
        ax.plot(range(window - 1, len(history['episode_lengths'])), ma,
               linewidth=2, color='red')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Length')
    ax.set_title('Episode Length')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(Path(save_dir) / "dqn_training.png", dpi=300, bbox_inches='tight')
    logger.info(f"Training plot saved to {save_dir}/dqn_training.png")
    
    logger.info("\n" + "=" * 80)
    logger.info("DQN TRAINING COMPLETED")
    logger.info(f"Final epsilon: {agent.epsilon:.3f}")
    logger.info(f"Final mean reward (last 100): {np.mean(history['episode_rewards'][-100:]):.2f}")
    logger.info("=" * 80)
    
    return agent, history



"""
EXAMPLE 1: Train standard DQN
------------------------------
env = EnergyHarvestingCameraEnv("NREL.csv", SystemParameters())

agent, history = train_dqn_baseline(
    env=env,
    n_episodes=5000,
    save_dir="experiments/dqn_standard"
)


EXAMPLE 2: Train Dueling Double DQN (best variant)
---------------------------------------------------
config = DQNConfig(
    state_dim=env.observation_space.shape[0],
    use_dueling=True,
    use_double_dqn=True,
    hidden_dim=256,
    learning_rate=3e-4
)

agent, history = train_dqn_baseline(
    env=env,
    config=config,
    n_episodes=5000,
    save_dir="experiments/dqn_dueling_double"
)


EXAMPLE 3: Train with prioritized replay
-----------------------------------------
config = DQNConfig(
    state_dim=env.observation_space.shape[0],
    use_prioritized_replay=True,
    priority_alpha=0.6,
    use_dueling=True,
    use_double_dqn=True
)

agent, history = train_dqn_baseline(
    env=env,
    config=config,
    n_episodes=5000,
    save_dir="experiments/dqn_prioritized"
)


EXAMPLE 4: Add to baseline comparison
--------------------------------------
# Load trained DQN
dqn_agent = DQNAgent(DQNConfig(state_dim=env.observation_space.shape[0]))
dqn_agent.load("experiments/dqn_baseline/dqn_final.pt")

# Add to comparison
methods = {
    'PPO': ppo_agent,
    'DQN': dqn_agent,
    'Random': RandomPolicy(),
    'MPC': mpc_policy
}

comparison_df = run_baseline_comparison(env, methods, n_episodes=100)
"""



def test_dqn_basic():
    """Test basic DQN functionality."""
    
    print("\n" + "=" * 80)
    print("TESTING DQN IMPLEMENTATION")
    print("=" * 80)
    
    # Create environment
    params = SystemParameters()
    env = EnergyHarvestingCameraEnv("NREL.csv", params)
    
    # Create DQN
    config = DQNConfig(state_dim=env.observation_space.shape[0])
    agent = DQNAgent(config)
    
    print(f"\nCreated DQN: {agent.name}")
    print(f"Network parameters: {sum(p.numel() for p in agent.q_network.parameters())}")
    
    # Run a few steps
    state, _ = env.reset(seed=42)
    
    print("\nRunning 10 test steps...")
    for step in range(10):
        action_tuple, _, _ = agent.select_action(state, deterministic=False)
        mode, capture, process, tx = action_tuple
        flat_action = mode * 18 + capture * 6 + process * 2 + tx
        
        next_state, reward, terminated, truncated, _ = env.step(flat_action)
        
        # Store transition
        agent.store_transition(state, action_tuple, reward, next_state, 
                             terminated or truncated)
        
        if step % 3 == 0:
            print(f"Step {step}: action={flat_action}, reward={reward:.2f}, "
                  f"epsilon={agent.epsilon:.3f}")
        
        state = next_state
        agent.steps += 1
        agent._update_epsilon()
        
        if terminated or truncated:
            break
    
    # Try training step
    print(f"\nBuffer size: {len(agent.replay_buffer)}")
    
    if len(agent.replay_buffer) >= config.batch_size:
        loss = agent.train_step()
        print(f"Training loss: {loss:.4f}")
    
    print("\n✅ DQN test completed successfully!")
    
    env.close()


if __name__ == "__main__":
    test_dqn_basic()
