"""
Comprehensive Evaluation System
Complete pipeline for training, wrapping, and statistically comparing all methods

Features:
- DQN training from scratch
- Multiple MPC variations
- Wrapper integration for all policies
- Statistical significance testing
- Publication-ready comparisons

Usage:
    python comprehensive_evaluation_system.py --train-dqn --episodes 100
    python comprehensive_evaluation_system.py --all --ppo path/to/ppo.pt --episodes 200

Author: Comprehensive Evaluation System
Date: 2026-02-07
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import json
import argparse
import logging
from tqdm import tqdm
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Import wrapper
from policy_wrapper import (
    PolicyCompletionWrapper,
    CompletionConfig,
    create_wrapped_policy
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# EVALUATION RESULT DATACLASS
# ============================================================================

@dataclass
class EvaluationResult:
    """Complete evaluation result for a single policy."""
    policy_name: str
    is_wrapped: bool
    n_episodes: int
    
    # Performance metrics
    mean_reward: float
    std_reward: float
    median_reward: float
    min_reward: float
    max_reward: float
    
    mean_delivery_rate: float
    std_delivery_rate: float
    
    mean_energy_efficiency: float
    mean_battery_final: float
    completion_rate: float
    mean_images_transmitted: float
    
    # Episode-level data
    episode_rewards: List[float]
    episode_delivery_rates: List[float]
    episode_lengths: List[float]
    
    # Wrapper statistics
    wrapper_stats: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)


# ============================================================================
# DQN TRAINER
# ============================================================================

class DQNTrainer:
    """Train DQN from scratch."""
    
    def __init__(self, env, config, save_dir: Path):
        self.env = env
        self.config = config
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        from dqn_baseline import DQNAgent
        self.agent = DQNAgent(config)
        
        self.history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'delivery_rates': [],
            'epsilon': [],
            'loss': []
        }
    
    def train(self, n_episodes: int = 5000) -> Any:
        """Train DQN agent."""
        logger.info(f"Training DQN for {n_episodes} episodes...")
        
        best_reward = -np.inf
        
        for episode in tqdm(range(n_episodes), desc="Training DQN"):
            state, _ = self.env.reset()
            episode_reward = 0.0
            episode_length = 0
            done = False
            
            while not done:
                # Select action - DQN returns (action_tuple, log_prob, value)
                action_tuple, _, _ = self.agent.select_action(state, deterministic=False)
                
                # Convert to flat action
                sleep, capture, process, tx = action_tuple
                flat_action = sleep * 18 + capture * 6 + process * 2 + tx
                
                # Step
                next_state, reward, terminated, truncated, _ = self.env.step(flat_action)
                done = terminated or truncated
                
                # Store transition
                self.agent.store_transition(state, action_tuple, reward, next_state, done)
                
                # Train
                if len(self.agent.replay_buffer) >= self.config.min_buffer_size:
                    loss = self.agent.train_step()
                    if loss is not None:
                        self.history['loss'].append(loss)
                
                episode_reward += reward
                episode_length += 1
                state = next_state
            
            # Record
            self.history['episode_rewards'].append(episode_reward)
            self.history['episode_lengths'].append(episode_length)
            
            event_stats = self.env.buffer.get_event_delivery_stats()
            self.history['delivery_rates'].append(event_stats['delivery_rate'])
            self.history['epsilon'].append(self.agent.epsilon)
            
            # Save best
            if episode_reward > best_reward:
                best_reward = episode_reward
                self.agent.save(str(self.save_dir / "dqn_best.pt"))
            
            # Periodic save
            if (episode + 1) % 500 == 0:
                self.agent.save(str(self.save_dir / f"dqn_episode_{episode+1}.pt"))
                logger.info(f"Episode {episode+1}: Reward={episode_reward:.1f}, "
                          f"Delivery={event_stats['delivery_rate']:.1%}, "
                          f"Epsilon={self.agent.epsilon:.3f}")
        
        # Save final
        self.agent.save(str(self.save_dir / "dqn_final.pt"))
        
        # Save history
        with open(self.save_dir / "dqn_history.json", 'w') as f:
            json.dump(self.history, f, indent=2)
        
        logger.info(f"✅ DQN training complete. Best reward: {best_reward:.2f}")
        
        return self.agent


# ============================================================================
# MPC VARIATIONS
# ============================================================================
# FIXED MPC VARIATIONS FOR COMPREHENSIVE EVALUATION SYSTEM
# ============================================================================
"""
Fixed MPC wrapper classes that correctly handle tuple returns from MPC.

The issue was that the wrappers were calling mpc.select_action() which returns
a TUPLE, then trying to convert it back to tuple assuming it was a flat int.

Fix: Just return the tuple directly since MPC already returns the right format!

Author: Fixed MPC Wrappers
Date: 2026-02-08
"""

import numpy as np
from typing import Tuple, Optional, Dict


class MPCVariant:
    """Base class for MPC variations."""
    
    def __init__(self, env, name: str, horizon: int = 10):
        self.env = env
        self.name = name
        self.horizon = horizon
    
    def select_action(self, obs, info=None):
        """Must be implemented by subclasses."""
        raise NotImplementedError


class GreedyMPC(MPCVariant):
    """Greedy MPC: Maximize immediate reward."""
    
    def __init__(self, env):
        super().__init__(env, "MPC_Greedy", horizon=1)
    
    def select_action(self, obs, info=None):
        battery_soc = obs[0]
        
        # Simple greedy: maximize delivery while conserving battery
        if battery_soc > 0.5:
            return (0, 2, 2, 1)  # Aggressive
        elif battery_soc > 0.3:
            return (0, 1, 1, 1)  # Moderate
        else:
            return (1, 1, 0, 0)  # Conservative


class ShortHorizonMPC(MPCVariant):
    """MPC with short horizon (5 steps)."""
    
    def __init__(self, env):
        super().__init__(env, "MPC_Short_Horizon", horizon=5)
        try:
            from mpc_baseline import ModelPredictiveController, MPCConfig
            from src.env.core import SystemParameters
            
            params = SystemParameters()
            config = MPCConfig(horizon=5, sampling_strategy="smart", n_random_samples=20)
            self.mpc = ModelPredictiveController(params, config)
        except Exception as e:
            raise ImportError(f"Failed to initialize ShortHorizonMPC: {e}")
    
    def select_action(self, obs, info=None):
        # FIX: MPC now returns tuple directly, just return it!
        action_tuple = self.mpc.select_action(obs, info if info else {})
        
        # Validate it's a tuple
        if not isinstance(action_tuple, tuple) or len(action_tuple) != 4:
            raise ValueError(f"MPC returned invalid format: {action_tuple}")
        
        return action_tuple  # ✅ Return tuple directly


class LongHorizonMPC(MPCVariant):
    """MPC with long horizon (20 steps)."""
    
    def __init__(self, env):
        super().__init__(env, "MPC_Long_Horizon", horizon=20)
        try:
            from mpc_baseline import ModelPredictiveController, MPCConfig
            from src.env.core import SystemParameters
            
            params = SystemParameters()
            config = MPCConfig(horizon=20, sampling_strategy="random", n_random_samples=50)
            self.mpc = ModelPredictiveController(params, config)
        except Exception as e:
            raise ImportError(f"Failed to initialize LongHorizonMPC: {e}")
    
    def select_action(self, obs, info=None):
        # FIX: MPC now returns tuple directly, just return it!
        action_tuple = self.mpc.select_action(obs, info if info else {})
        
        # Validate it's a tuple
        if not isinstance(action_tuple, tuple) or len(action_tuple) != 4:
            raise ValueError(f"MPC returned invalid format: {action_tuple}")
        
        return action_tuple  # ✅ Return tuple directly


class AdaptiveMPC(MPCVariant):
    """MPC with adaptive horizon based on battery level."""
    
    def __init__(self, env):
        super().__init__(env, "MPC_Adaptive", horizon=10)
        try:
            from mpc_baseline import ModelPredictiveController, MPCConfig
            from src.env.core import SystemParameters
            
            params = SystemParameters()
            
            # Two MPC controllers with different horizons
            short_config = MPCConfig(horizon=5, sampling_strategy="smart", n_random_samples=20)
            long_config = MPCConfig(horizon=15, sampling_strategy="smart", n_random_samples=40)
            
            self.short_mpc = ModelPredictiveController(params, short_config)
            self.long_mpc = ModelPredictiveController(params, long_config)
        except Exception as e:
            raise ImportError(f"Failed to initialize AdaptiveMPC: {e}")
    
    def select_action(self, obs, info=None):
        battery_soc = obs[0]
        
        # Use long horizon when battery is good (can afford to plan ahead)
        # Use short horizon when battery is low (need immediate action)
        if battery_soc > 0.4:
            action_tuple = self.long_mpc.select_action(obs, info if info else {})
        else:
            action_tuple = self.short_mpc.select_action(obs, info if info else {})
        
        # Validate it's a tuple
        if not isinstance(action_tuple, tuple) or len(action_tuple) != 4:
            raise ValueError(f"MPC returned invalid format: {action_tuple}")
        
        return action_tuple  # ✅ Return tuple directly


def create_mpc_variations(env) -> Dict[str, MPCVariant]:
    """Create all MPC variations."""
    mpc_dict = {}
    
    # Always include greedy (doesn't need mpc_baseline)
    try:
        mpc_dict['MPC_Greedy'] = GreedyMPC(env)
        logger.info("  Created MPC_Greedy")
    except Exception as e:
        logger.warning(f"Failed to create MPC_Greedy: {e}")
    
    # Try to create other MPC variations
    try:
        mpc_dict['MPC_Short_Horizon'] = ShortHorizonMPC(env)
        logger.info("  Created MPC_Short_Horizon")
    except (ImportError, Exception) as e:
        logger.warning(f"Skipping MPC_Short_Horizon: {e}")
    
    try:
        mpc_dict['MPC_Long_Horizon'] = LongHorizonMPC(env)
        logger.info("  Created MPC_Long_Horizon")
    except (ImportError, Exception) as e:
        logger.warning(f"Skipping MPC_Long_Horizon: {e}")
    
    try:
        mpc_dict['MPC_Adaptive'] = AdaptiveMPC(env)
        logger.info("  Created MPC_Adaptive")
    except (ImportError, Exception) as e:
        logger.warning(f"Skipping MPC_Adaptive: {e}")
    
    if len(mpc_dict) == 0:
        logger.warning("No MPC variations could be created")
    else:
        logger.info(f"  Total MPC variations: {len(mpc_dict)}")
    
    return mpc_dict


# For importing
import logging
logger = logging.getLogger(__name__)

print("✅ Fixed MPC variations loaded")
print("")
print("USAGE:")
print("======")
print("In comprehensive_evaluation_system.py, replace the MPC variation classes")
print("(GreedyMPC, ShortHorizonMPC, LongHorizonMPC, AdaptiveMPC)")
print("and the create_mpc_variations() function with these fixed versions.")
print("")
print("The key fix:")
print("  OLD: flat_action = self.mpc.select_action(...)")
print("       sleep = flat_action // 18  # ❌ Assumes int, but got tuple!")
print("")
print("  NEW: action_tuple = self.mpc.select_action(...)")
print("       return action_tuple  # ✅ Just return the tuple directly!")


# ============================================================================
# POLICY EVALUATOR
# ============================================================================

def evaluate_policy(
    env,
    policy,
    n_episodes: int = 100,
    seed: int = 42,
    verbose: bool = True
) -> EvaluationResult:
    """Evaluate a single policy."""
    np.random.seed(seed)
    
    is_wrapped = isinstance(policy, PolicyCompletionWrapper)
    if is_wrapped:
        policy.reset_statistics()
    
    # Storage
    episode_rewards = []
    episode_delivery_rates = []
    episode_lengths = []
    episode_energy_efficiency = []
    episode_battery_finals = []
    episode_images_tx = []
    completed_episodes = 0
    
    policy_name = policy.name if hasattr(policy, 'name') else 'Unknown'
    iterator = tqdm(range(n_episodes), desc=f"Evaluating {policy_name}") if verbose else range(n_episodes)
    
    for ep in iterator:
        obs, info = env.reset(seed=seed + ep)
        episode_reward = 0.0
        episode_length = 0
        done = False
        
        while not done:
            # Get action - handle different policy interfaces
            try:
                # Try PPO/DQN interface with deterministic parameter
                if hasattr(policy, 'actor'):  # PPO agent
                    result = policy.select_action(obs, deterministic=True)
                    if isinstance(result, tuple) and len(result) == 3:
                        action_tuple, _, _ = result
                    else:
                        action_tuple = result
                else:
                    # Try without deterministic (baselines, wrapped policies)
                    result = policy.select_action(obs, info)
                    if isinstance(result, tuple) and len(result) == 3:
                        action_tuple, _, _ = result
                    else:
                        action_tuple = result
            except TypeError:
                # Fallback: call without any extra arguments
                result = policy.select_action(obs)
                if isinstance(result, tuple) and len(result) == 3:
                    action_tuple, _, _ = result
                else:
                    action_tuple = result
            
            # Validate action tuple
            if not isinstance(action_tuple, tuple) or len(action_tuple) != 4:
                raise ValueError(f"Invalid action from {policy_name}: {action_tuple}")
            
            # Convert to flat action
            sleep, capture, process, tx = action_tuple
            flat_action = sleep * 18 + capture * 6 + process * 2 + tx
            
            # Step
            obs, reward, terminated, truncated, info = env.step(flat_action)
            episode_reward += reward
            episode_length += 1
            done = terminated or truncated
        
        # Record
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        event_stats = env.buffer.get_event_delivery_stats()
        episode_delivery_rates.append(event_stats['delivery_rate'])
        
        energy_h = env.episode_stats['energy_harvested']
        energy_c = env.episode_stats['energy_consumed']
        episode_energy_efficiency.append(energy_h / max(energy_c, 1.0))
        
        episode_battery_finals.append(env.battery.get_normalized_level())
        episode_images_tx.append(env.episode_stats['images_transmitted'])
        
        if truncated:
            completed_episodes += 1
    
    # Wrapper stats
    wrapper_stats = None
    if is_wrapped:
        wrapper_stats = policy.get_statistics()
    
    result = EvaluationResult(
        policy_name=policy_name,
        is_wrapped=is_wrapped,
        n_episodes=n_episodes,
        mean_reward=float(np.mean(episode_rewards)),
        std_reward=float(np.std(episode_rewards)),
        median_reward=float(np.median(episode_rewards)),
        min_reward=float(np.min(episode_rewards)),
        max_reward=float(np.max(episode_rewards)),
        mean_delivery_rate=float(np.mean(episode_delivery_rates)),
        std_delivery_rate=float(np.std(episode_delivery_rates)),
        mean_energy_efficiency=float(np.mean(episode_energy_efficiency)),
        mean_battery_final=float(np.mean(episode_battery_finals)),
        completion_rate=float(completed_episodes / n_episodes),
        mean_images_transmitted=float(np.mean(episode_images_tx)),
        episode_rewards=episode_rewards,
        episode_delivery_rates=episode_delivery_rates,
        episode_lengths=episode_lengths,
        wrapper_stats=wrapper_stats
    )
    
    if verbose:
        logger.info(f"\n{policy_name}: Reward={result.mean_reward:.2f}±{result.std_reward:.2f}, "
                   f"Delivery={result.mean_delivery_rate:.1%}")
        if is_wrapped and wrapper_stats:
            logger.info(f"  Wrapper: {wrapper_stats['failures']} failures, "
                       f"{wrapper_stats['completions']} completions")
    
    return result


# ============================================================================
# STATISTICAL ANALYSIS
# ============================================================================

def compute_statistical_significance(results: Dict[str, EvaluationResult]) -> pd.DataFrame:
    """Compute pairwise statistical significance (t-tests)."""
    names = list(results.keys())
    n = len(names)
    
    # Create matrices for p-values and effect sizes
    p_values = np.ones((n, n))
    effect_sizes = np.zeros((n, n))
    
    for i, name1 in enumerate(names):
        for j, name2 in enumerate(names):
            if i != j:
                rewards1 = results[name1].episode_rewards
                rewards2 = results[name2].episode_rewards
                
                # T-test
                t_stat, p_val = stats.ttest_ind(rewards1, rewards2)
                p_values[i, j] = p_val
                
                # Cohen's d (effect size)
                mean1, mean2 = np.mean(rewards1), np.mean(rewards2)
                std1, std2 = np.std(rewards1, ddof=1), np.std(rewards2, ddof=1)
                pooled_std = np.sqrt(((len(rewards1)-1)*std1**2 + (len(rewards2)-1)*std2**2) / 
                                    (len(rewards1) + len(rewards2) - 2))
                effect_sizes[i, j] = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0
    
    # Create DataFrame
    df_pvalues = pd.DataFrame(p_values, index=names, columns=names)
    df_effects = pd.DataFrame(effect_sizes, index=names, columns=names)
    
    return df_pvalues, df_effects


# ============================================================================
# COMPREHENSIVE EVALUATION RUNNER
# ============================================================================

def run_comprehensive_evaluation(
    env,
    ppo_agent=None,
    dqn_agent=None,
    n_episodes: int = 100,
    save_dir: str = "comprehensive_results",
    seed: int = 42,
    include_wrapped: bool = True,
    include_mpc_variations: bool = True
) -> Dict[str, EvaluationResult]:
    """
    Run complete evaluation of all methods.
    
    Args:
        env: Environment
        ppo_agent: Trained PPO agent (optional)
        dqn_agent: Trained DQN agent (optional)
        n_episodes: Episodes per policy
        save_dir: Output directory
        seed: Random seed
        include_wrapped: Also evaluate wrapped versions
        include_mpc_variations: Include MPC variations
    
    Returns:
        Dictionary of results
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*80)
    logger.info("COMPREHENSIVE EVALUATION")
    logger.info("="*80)
    
    results = {}
    policies_to_evaluate = {}
    
    # ========================================================================
    # 1. LOAD BASELINE POLICIES
    # ========================================================================
    logger.info("\n1. Loading baseline policies...")
    
    from baseline_policies import (
        RandomPolicy,
        AlwaysSleepPolicy,
        AlwaysMaxThroughputPolicy,
        BatteryThresholdPolicy,
        SmartHeuristicPolicy
    )
    
    baselines = {
        'Random': RandomPolicy(),
        'Always_Sleep': AlwaysSleepPolicy(),
        'Max_Throughput': AlwaysMaxThroughputPolicy(),
        'Battery_Threshold': BatteryThresholdPolicy(),
        'Smart_Heuristic': SmartHeuristicPolicy()
    }
    
    policies_to_evaluate.update(baselines)
    logger.info(f"  Loaded {len(baselines)} baseline policies")
    
    # ========================================================================
    # 2. ADD MPC VARIATIONS
    # ========================================================================
    if include_mpc_variations:
        logger.info("\n2. Creating MPC variations...")
        mpc_policies = create_mpc_variations(env)
        policies_to_evaluate.update(mpc_policies)
        logger.info(f"  Created {len(mpc_policies)} MPC variations")
    
    # ========================================================================
    # 3. ADD DQN
    # ========================================================================
    if dqn_agent is not None:
        logger.info("\n3. Adding DQN agent...")
        policies_to_evaluate['DQN'] = dqn_agent
    
    # ========================================================================
    # 4. ADD PPO
    # ========================================================================
    if ppo_agent is not None:
        logger.info("\n4. Adding PPO agent...")
        policies_to_evaluate['PPO'] = ppo_agent
    
    # ========================================================================
    # 5. EVALUATE UNWRAPPED POLICIES
    # ========================================================================
    logger.info(f"\n5. Evaluating {len(policies_to_evaluate)} unwrapped policies...")
    
    for name, policy in policies_to_evaluate.items():
        logger.info(f"\n  Evaluating: {name}")
        result = evaluate_policy(env, policy, n_episodes, seed, verbose=False)
        results[name] = result
    
    # ========================================================================
    # 6. EVALUATE WRAPPED POLICIES
    # ========================================================================
    if include_wrapped:
        logger.info(f"\n6. Evaluating wrapped versions...")
        
        wrapper_config = CompletionConfig(
            use_battery_adaptive=True,
            low_battery_threshold=0.35,
            critical_battery_threshold=0.25,
            log_failures=False
        )
        
        for name, policy in policies_to_evaluate.items():
            wrapped_name = f"{name}_Wrapped"
            logger.info(f"\n  Evaluating: {wrapped_name}")
            
            wrapped_policy = create_wrapped_policy(policy, config=wrapper_config)
            result = evaluate_policy(env, wrapped_policy, n_episodes, seed, verbose=False)
            results[wrapped_name] = result
    
    # ========================================================================
    # 7. STATISTICAL ANALYSIS
    # ========================================================================
    logger.info("\n7. Computing statistical significance...")
    
    df_pvalues, df_effects = compute_statistical_significance(results)
    
    # Save statistical results
    df_pvalues.to_csv(save_path / "pairwise_pvalues.csv")
    df_effects.to_csv(save_path / "effect_sizes.csv")
    
    # ========================================================================
    # 8. SAVE RESULTS
    # ========================================================================
    logger.info("\n8. Saving results...")
    
    # Main comparison table
    comparison_data = []
    for name, result in results.items():
        row = {
            'Policy': name,
            'Wrapped': result.is_wrapped,
            'Mean_Reward': result.mean_reward,
            'Std_Reward': result.std_reward,
            'Mean_Delivery': result.mean_delivery_rate,
            'Completion_Rate': result.completion_rate,
            'Energy_Efficiency': result.mean_energy_efficiency,
            'Images_TX': result.mean_images_transmitted
        }
        
        if result.wrapper_stats:
            row['Wrapper_Failures'] = result.wrapper_stats['failures']
            row['Wrapper_Completions'] = result.wrapper_stats['completions']
        
        comparison_data.append(row)
    
    df_comparison = pd.DataFrame(comparison_data)
    df_comparison = df_comparison.sort_values('Mean_Reward', ascending=False)
    df_comparison.to_csv(save_path / "comprehensive_comparison.csv", index=False)
    
    # Save full results
    with open(save_path / "full_results.json", 'w') as f:
        serializable_results = {
            name: result.to_dict() for name, result in results.items()
        }
        json.dump(serializable_results, f, indent=2)
    
    # ========================================================================
    # 9. VISUALIZATION
    # ========================================================================
    logger.info("\n9. Creating visualizations...")
    
    plot_comprehensive_results(results, df_pvalues, save_path / "comprehensive_plots.png")
    
    logger.info(f"\n✅ Evaluation complete! Results saved to {save_path}")
    
    return results


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_comprehensive_results(
    results: Dict[str, EvaluationResult],
    pvalues: pd.DataFrame,
    save_path: Path
):
    """Create comprehensive visualization."""
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Prepare data
    names = list(results.keys())
    names_sorted = sorted(names, key=lambda x: results[x].mean_reward, reverse=True)
    
    # Color coding
    colors = []
    for name in names_sorted:
        if 'Wrapped' in name:
            colors.append('orange')
        elif 'PPO' in name:
            colors.append('gold')
        elif 'DQN' in name:
            colors.append('purple')
        elif 'MPC' in name:
            colors.append('green')
        else:
            colors.append('steelblue')
    
    # 1. Reward comparison
    ax = fig.add_subplot(gs[0, 0])
    rewards = [results[n].mean_reward for n in names_sorted]
    errors = [results[n].std_reward for n in names_sorted]
    ax.barh(range(len(names_sorted)), rewards, xerr=errors, color=colors, 
           edgecolor='black', alpha=0.7)
    ax.set_yticks(range(len(names_sorted)))
    ax.set_yticklabels(names_sorted, fontsize=8)
    ax.set_xlabel('Mean Reward ± Std')
    ax.set_title('Reward Comparison', fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # 2. Delivery rate
    ax = fig.add_subplot(gs[0, 1])
    delivery = [results[n].mean_delivery_rate * 100 for n in names_sorted]
    ax.barh(range(len(names_sorted)), delivery, color=colors, 
           edgecolor='black', alpha=0.7)
    ax.set_yticks(range(len(names_sorted)))
    ax.set_yticklabels(names_sorted, fontsize=8)
    ax.set_xlabel('Delivery Rate (%)')
    ax.set_title('Event Delivery Performance', fontweight='bold')
    ax.set_xlim([0, 100])
    ax.grid(axis='x', alpha=0.3)
    
    # 3. Completion rate
    ax = fig.add_subplot(gs[0, 2])
    completion = [results[n].completion_rate * 100 for n in names_sorted]
    ax.barh(range(len(names_sorted)), completion, color=colors,
           edgecolor='black', alpha=0.7)
    ax.set_yticks(range(len(names_sorted)))
    ax.set_yticklabels(names_sorted, fontsize=8)
    ax.set_xlabel('Completion Rate (%)')
    ax.set_title('Episode Completion', fontweight='bold')
    ax.set_xlim([0, 105])
    ax.grid(axis='x', alpha=0.3)
    
    # 4. Statistical significance heatmap
    ax = fig.add_subplot(gs[1, :])
    # Only show top policies for readability
    top_n = min(15, len(names_sorted))
    top_names = names_sorted[:top_n]
    
    pval_matrix = pvalues.loc[top_names, top_names].values
    
    # Mask diagonal
    mask = np.eye(len(top_names), dtype=bool)
    
    sns.heatmap(pval_matrix, mask=mask, annot=True, fmt='.3f', 
               cmap='RdYlGn_r', center=0.05, vmin=0, vmax=0.1,
               xticklabels=top_names, yticklabels=top_names,
               ax=ax, cbar_kws={'label': 'p-value'})
    ax.set_title('Statistical Significance (p-values, green=significant)', fontweight='bold')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=8)
    
    # 5. Wrapper effectiveness
    ax = fig.add_subplot(gs[2, 0])
    wrapped_results = {k: v for k, v in results.items() if v.is_wrapped and v.wrapper_stats}
    if wrapped_results:
        wrapper_names = list(wrapped_results.keys())
        failures = [wrapped_results[n].wrapper_stats['failures'] for n in wrapper_names]
        completions = [wrapped_results[n].wrapper_stats['completions'] for n in wrapper_names]
        
        x = np.arange(len(wrapper_names))
        width = 0.35
        ax.bar(x - width/2, failures, width, label='Failures', color='red', alpha=0.7)
        ax.bar(x + width/2, completions, width, label='Completions', color='green', alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(wrapper_names, rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Count')
        ax.set_title('Wrapper Activity', fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
    
    # 6. Energy efficiency
    ax = fig.add_subplot(gs[2, 1])
    efficiency = [results[n].mean_energy_efficiency for n in names_sorted]
    ax.barh(range(len(names_sorted)), efficiency, color=colors,
           edgecolor='black', alpha=0.7)
    ax.set_yticks(range(len(names_sorted)))
    ax.set_yticklabels(names_sorted, fontsize=8)
    ax.set_xlabel('Energy Efficiency')
    ax.set_title('Harvested/Consumed Ratio', fontweight='bold')
    ax.axvline(x=1.0, color='red', linestyle='--', label='Neutral')
    ax.legend()
    ax.grid(axis='x', alpha=0.3)
    
    # 7. Reward distribution
    ax = fig.add_subplot(gs[2, 2])
    # Show top 10 for readability
    top_10_names = names_sorted[:10]
    reward_distributions = [results[n].episode_rewards for n in top_10_names]
    
    bp = ax.boxplot(reward_distributions, labels=top_10_names, vert=True,
                   patch_artist=True, showfliers=False)
    
    for patch, name in zip(bp['boxes'], top_10_names):
        idx = names_sorted.index(name)
        patch.set_facecolor(colors[idx])
        patch.set_alpha(0.7)
    
    ax.set_xticklabels(top_10_names, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Episode Reward')
    ax.set_title('Reward Distribution (Top 10)', fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Comprehensive Policy Comparison', fontsize=16, fontweight='bold', y=0.995)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"  Plot saved to {save_path}")
    plt.close()


logger.info("✅ Comprehensive evaluation system loaded")
