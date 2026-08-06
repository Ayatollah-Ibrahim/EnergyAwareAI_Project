"""
COMPREHENSIVE STATISTICAL EVALUATION SYSTEM
===========================================

Complete framework for evaluating PPO against all baselines with:
- Automatic compatibility checking
- Statistical significance testing
- Professional tables and visualizations
- Support for DQN, MPC, and simple baselines

Author: Statistical Evaluation Framework
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
import logging
from tqdm import tqdm
from scipy import stats
import warnings
import sys

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Set plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10


# ============================================================================
# COMPATIBILITY CHECKER
# ============================================================================

class CompatibilityChecker:
    """Comprehensive compatibility checker for all baseline files."""
    
    def __init__(self):
        self.compatibility_report = {}
        self.compatible_baselines = []
        self.available_methods = []
    
    def check_baseline_policies(self) -> bool:
        """Check baseline_policies.py compatibility."""
        logger.info("\n🔍 Checking baseline_policies.py...")
        
        try:
            from baseline_policies import (
                RandomPolicy,
                AlwaysSleepPolicy,
                AlwaysMaxThroughputPolicy,
                BatteryThresholdPolicy,
                SmartHeuristicPolicy,
                OraclePolicy
            )
            
            # Test instantiation
            test_policies = [
                RandomPolicy(),
                AlwaysSleepPolicy(),
                BatteryThresholdPolicy(),
                SmartHeuristicPolicy()
            ]
            
            # Test select_action interface
            test_obs = np.zeros(44)
            test_info = {}
            
            for policy in test_policies:
                action = policy.select_action(test_obs, test_info)
                assert isinstance(action, tuple), f"{policy.name} doesn't return tuple"
                assert len(action) == 4, f"{policy.name} returns wrong action length"
            
            logger.info("✅ baseline_policies.py is COMPATIBLE")
            self.compatibility_report['baseline_policies'] = 'OK'
            self.compatible_baselines.extend(['Random', 'Battery Threshold', 'Smart Heuristic'])
            return True
            
        except Exception as e:
            logger.error(f"❌ baseline_policies.py INCOMPATIBLE: {e}")
            self.compatibility_report['baseline_policies'] = f'ERROR: {e}'
            return False
    
    def check_dqn_baseline(self) -> bool:
        """Check dqn_baseline.py compatibility."""
        logger.info("\n🔍 Checking dqn_baseline.py...")
        
        try:
            from dqn_baseline import DQNAgent, DQNConfig
            
            # Test instantiation
            config = DQNConfig(state_dim=44)
            config.validate()
            
            # Test agent creation
            agent = DQNAgent(config, n_actions=54)
            
            # Test action selection
            test_state = np.zeros(44)
            action_tuple, _, _ = agent.select_action(test_state, deterministic=True)
            assert isinstance(action_tuple, tuple)
            assert len(action_tuple) == 4
            
            logger.info("✅ dqn_baseline.py is COMPATIBLE")
            self.compatibility_report['dqn_baseline'] = 'OK'
            self.compatible_baselines.append('DQN')
            return True
            
        except Exception as e:
            logger.error(f"⚠️  dqn_baseline.py INCOMPATIBLE: {e}")
            logger.info("    (DQN will be skipped in evaluation)")
            self.compatibility_report['dqn_baseline'] = f'SKIP: {e}'
            return False
    
    def check_mpc_baseline(self) -> bool:
        """Check mpc_baseline.py compatibility."""
        logger.info("\n🔍 Checking mpc_baseline.py...")
        
        try:
            from mpc_baseline import MPCAgent, MPCConfig
            
            # MPC needs environment reference
            config = MPCConfig()
            config.validate()
            
            logger.info("✅ mpc_baseline.py is COMPATIBLE (needs env for initialization)")
            self.compatibility_report['mpc_baseline'] = 'OK'
            self.compatible_baselines.append('MPC')
            return True
            
        except Exception as e:
            logger.error(f"⚠️  mpc_baseline.py INCOMPATIBLE: {e}")
            logger.info("    (MPC will be skipped in evaluation)")
            self.compatibility_report['mpc_baseline'] = f'SKIP: {e}'
            return False
    
    def check_evaluation_script(self) -> bool:
        """Check evaluate_baselines_with_wrapper.py compatibility."""
        logger.info("\n🔍 Checking evaluate_baselines_with_wrapper.py...")
        
        try:
            from evaluate_baselines_with_wrapper import (
                CompletionConfig,
                PolicyCompletionWrapper
            )
            
            # Test config
            config = CompletionConfig()
            
            logger.info("✅ evaluate_baselines_with_wrapper.py is COMPATIBLE")
            self.compatibility_report['evaluation_script'] = 'OK'
            return True
            
        except Exception as e:
            logger.error(f"⚠️  evaluate_baselines_with_wrapper.py INCOMPATIBLE: {e}")
            logger.info("    (Wrapper evaluation will be skipped)")
            self.compatibility_report['evaluation_script'] = f'SKIP: {e}'
            return False
    
    def check_environment(self) -> bool:
        """Check if environment is accessible."""
        logger.info("\n🔍 Checking environment files...")
        
         
            # Test instantiation
            params = SystemParameters()
            params.validate()
            
            logger.info("✅ Environment files are COMPATIBLE")
            self.compatibility_report['environment'] = 'OK'
            return True
            
        except Exception as e:
            logger.error(f"❌ Environment files INCOMPATIBLE: {e}")
            self.compatibility_report['environment'] = f'ERROR: {e}'
            return False
    
    def run_all_checks(self) -> Dict[str, str]:
        """Run all compatibility checks."""
        logger.info("="*80)
        logger.info("COMPATIBILITY CHECK")
        logger.info("="*80)
        
        # Check environment first (critical)
        env_ok = self.check_environment()
        if not env_ok:
            logger.error("\n❌ CRITICAL: Environment files are not accessible!")
            logger.error("   Please ensure Part1_Core and Part3_Environment are available.")
            return self.compatibility_report
        
        # Check baseline policies
        self.check_baseline_policies()
        
        # Check advanced baselines (optional)
        self.check_dqn_baseline()
        self.check_mpc_baseline()
        
        # Check wrapper script
        self.check_evaluation_script()
        
        # Summary
        logger.info("\n" + "="*80)
        logger.info("COMPATIBILITY SUMMARY")
        logger.info("="*80)
        
        for component, status in self.compatibility_report.items():
            if status == 'OK':
                logger.info(f"✅ {component}: {status}")
            elif 'SKIP' in status:
                logger.info(f"⚠️  {component}: OPTIONAL - will be skipped")
            else:
                logger.info(f"❌ {component}: {status}")
        
        logger.info(f"\n📊 Available for evaluation: {', '.join(self.compatible_baselines)}")
        logger.info("="*80)
        
        return self.compatibility_report


# ============================================================================
# EVALUATION RESULT DATACLASS
# ============================================================================

@dataclass
class EvaluationResult:
    """Results from evaluating a single policy."""
    
    policy_name: str
    policy_type: str  # 'baseline', 'dqn', 'mpc', 'ppo'
    n_episodes: int
    
    # Episode-level data (for statistical testing)
    episode_rewards: List[float]
    episode_delivery_rates: List[float]
    episode_completion_rates: List[float]
    episode_energy_efficiencies: List[float]
    
    # Summary statistics
    mean_reward: float
    std_reward: float
    median_reward: float
    
    mean_delivery: float
    std_delivery: float
    
    mean_completion: float
    mean_energy_efficiency: float
    
    # Additional metrics
    total_battery_failures: int
    total_images_transmitted: int
    
    def to_dict(self) -> Dict:
        """Convert to dictionary (exclude episode-level data)."""
        return {
            'policy_name': self.policy_name,
            'policy_type': self.policy_type,
            'n_episodes': self.n_episodes,
            'mean_reward': self.mean_reward,
            'std_reward': self.std_reward,
            'median_reward': self.median_reward,
            'mean_delivery': self.mean_delivery,
            'std_delivery': self.std_delivery,
            'mean_completion': self.mean_completion,
            'mean_energy_efficiency': self.mean_energy_efficiency,
            'total_battery_failures': self.total_battery_failures,
            'total_images_transmitted': self.total_images_transmitted
        }


# ============================================================================
# STATISTICAL SIGNIFICANCE TESTING
# ============================================================================

class StatisticalComparison:
    """Statistical comparison between policies with comprehensive tests."""
    
    @staticmethod
    def compare_two_policies(
        result1: EvaluationResult,
        result2: EvaluationResult,
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """
        Perform comprehensive statistical comparison between two policies.
        
        Tests:
        - T-test (parametric)
        - Mann-Whitney U (non-parametric)
        - Effect size (Cohen's d)
        
        Returns:
            Dictionary with test results for each metric
        """
        
        comparison = {}
        
        metrics = {
            'reward': (result1.episode_rewards, result2.episode_rewards),
            'delivery': (result1.episode_delivery_rates, result2.episode_delivery_rates),
            'completion': (result1.episode_completion_rates, result2.episode_completion_rates)
        }
        
        for metric_name, (data1, data2) in metrics.items():
            # Convert to numpy arrays
            arr1 = np.array(data1)
            arr2 = np.array(data2)
            
            # Compute means
            mean1 = np.mean(arr1)
            mean2 = np.mean(arr2)
            diff = mean1 - mean2
            
            # T-test (parametric)
            t_stat, t_pval = stats.ttest_ind(arr1, arr2)
            
            # Mann-Whitney U test (non-parametric)
            u_stat, u_pval = stats.mannwhitneyu(arr1, arr2, alternative='two-sided')
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt(
                ((len(arr1) - 1) * np.var(arr1, ddof=1) +
                 (len(arr2) - 1) * np.var(arr2, ddof=1)) /
                (len(arr1) + len(arr2) - 2)
            )
            cohens_d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0
            
            # Interpret effect size
            abs_d = abs(cohens_d)
            if abs_d < 0.2:
                effect_interpretation = "negligible"
            elif abs_d < 0.5:
                effect_interpretation = "small"
            elif abs_d < 0.8:
                effect_interpretation = "medium"
            else:
                effect_interpretation = "large"
            
            # Determine significance (both tests must agree)
            is_significant = (t_pval < alpha) and (u_pval < alpha)
            
            # Confidence interval for difference
            se_diff = np.sqrt(np.var(arr1, ddof=1)/len(arr1) + np.var(arr2, ddof=1)/len(arr2))
            ci_low = diff - 1.96 * se_diff
            ci_high = diff + 1.96 * se_diff
            
            comparison[metric_name] = {
                'mean1': mean1,
                'mean2': mean2,
                'difference': diff,
                'percent_diff': (diff / abs(mean2) * 100) if mean2 != 0 else 0,
                't_statistic': t_stat,
                't_pvalue': t_pval,
                'u_statistic': u_stat,
                'u_pvalue': u_pval,
                'cohens_d': cohens_d,
                'effect_size': effect_interpretation,
                'is_significant': is_significant,
                'better_policy': result1.policy_name if mean1 > mean2 else result2.policy_name,
                'ci_95': (ci_low, ci_high)
            }
        
        return comparison
    
    @staticmethod
    def create_comparison_table(
        results: List[EvaluationResult],
        reference_policy: str = "PPO Agent"
    ) -> pd.DataFrame:
        """
        Create publication-ready comparison table.
        
        Args:
            results: List of evaluation results
            reference_policy: Policy to compare against
        
        Returns:
            DataFrame with statistical comparison
        """
        
        # Find reference result
        ref_result = None
        for r in results:
            if r.policy_name == reference_policy:
                ref_result = r
                break
        
        if ref_result is None:
            logger.warning(f"Reference policy '{reference_policy}' not found")
            # Use first result as reference
            ref_result = results[0]
            reference_policy = ref_result.policy_name
            logger.info(f"Using '{reference_policy}' as reference instead")
        
        comparison_data = []
        
        for result in results:
            if result.policy_name == reference_policy:
                # Add reference row
                row = {
                    'Policy': f"{result.policy_name} (ref)",
                    'Type': result.policy_type.upper(),
                    'Reward': f"{result.mean_reward:.1f} ± {result.std_reward:.1f}",
                    'Reward Δ': '—',
                    'p-value': '—',
                    'Delivery': f"{result.mean_delivery:.1%}",
                    'Delivery Δ': '—',
                    'Completion': f"{result.mean_completion:.1%}",
                    'Effect': '—'
                }
                comparison_data.append(row)
                continue
            
            # Perform comparison
            comp = StatisticalComparison.compare_two_policies(ref_result, result)
            
            # Significance markers
            if comp['reward']['is_significant']:
                if comp['reward']['t_pvalue'] < 0.001:
                    sig_reward = '***'
                elif comp['reward']['t_pvalue'] < 0.01:
                    sig_reward = '**'
                else:
                    sig_reward = '*'
            else:
                sig_reward = 'ns'
            
            if comp['delivery']['is_significant']:
                if comp['delivery']['t_pvalue'] < 0.001:
                    sig_delivery = '***'
                elif comp['delivery']['t_pvalue'] < 0.01:
                    sig_delivery = '**'
                else:
                    sig_delivery = '*'
            else:
                sig_delivery = 'ns'
            
            row = {
                'Policy': result.policy_name,
                'Type': result.policy_type.upper(),
                'Reward': f"{result.mean_reward:.1f} ± {result.std_reward:.1f}",
                'Reward Δ': f"{comp['reward']['difference']:+.1f} ({sig_reward})",
                'p-value': f"{comp['reward']['t_pvalue']:.4f}",
                'Delivery': f"{result.mean_delivery:.1%}",
                'Delivery Δ': f"{comp['delivery']['difference']*100:+.1f}% ({sig_delivery})",
                'Completion': f"{result.mean_completion:.1%}",
                'Effect': comp['reward']['effect_size']
            }
            
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        
        return df
    
    @staticmethod
    def create_pairwise_matrix(
        results: List[EvaluationResult],
        metric: str = 'reward'
    ) -> pd.DataFrame:
        """
        Create pairwise significance matrix.
        
        Shows p-values for all pairs of policies.
        """
        
        n = len(results)
        policy_names = [r.policy_name for r in results]
        
        # Initialize matrix
        matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    matrix[i, j] = np.nan
                else:
                    comp = StatisticalComparison.compare_two_policies(
                        results[i], results[j]
                    )
                    matrix[i, j] = comp[metric]['t_pvalue']
        
        df = pd.DataFrame(
            matrix,
            index=policy_names,
            columns=policy_names
        )
        
        return df


# ============================================================================
# POLICY EVALUATOR
# ============================================================================

def evaluate_policy(
    env,
    policy,
    policy_type: str,
    n_episodes: int = 100,
    seed: int = 42,
    verbose: bool = True
) -> EvaluationResult:
    """
    Evaluate a single policy comprehensively.
    
    Args:
        env: Environment
        policy: Policy to evaluate
        policy_type: 'baseline', 'dqn', 'mpc', 'ppo'
        n_episodes: Number of episodes
        seed: Random seed
        verbose: Show progress bar
    
    Returns:
        EvaluationResult object
    """
    
    np.random.seed(seed)
    
    # Get policy name
    if hasattr(policy, 'name'):
        policy_name = policy.name
    elif hasattr(policy, '__class__'):
        policy_name = policy.__class__.__name__
    else:
        policy_name = str(policy)
    
    # Storage
    episode_rewards = []
    episode_delivery_rates = []
    episode_completion_rates = []
    episode_energy_efficiencies = []
    total_battery_failures = 0
    total_images_transmitted = 0
    
    iterator = tqdm(range(n_episodes), desc=f"Evaluating {policy_name}") if verbose else range(n_episodes)
    
    for episode in iterator:
        state, info = env.reset(seed=seed + episode)
        episode_reward = 0.0
        done = False
        
        while not done:
            # Get action based on policy type
            try:
                if policy_type == 'baseline':
                    action_tuple = policy.select_action(state, info)
                    mode, capture, process, tx = action_tuple
                    flat_action = mode * 18 + capture * 6 + process * 2 + tx
                
                elif policy_type == 'dqn':
                    action_tuple, _, _ = policy.select_action(state, deterministic=True)
                    mode, capture, process, tx = action_tuple
                    flat_action = mode * 18 + capture * 6 + process * 2 + tx
                
                elif policy_type == 'mpc':
                    flat_action = policy.select_action(state)
                
                elif policy_type == 'ppo':
                    action_tuple, _, _ = policy.select_action(state, deterministic=True)
                    mode, capture, process, tx = action_tuple
                    flat_action = mode * 18 + capture * 6 + process * 2 + tx
                
                else:
                    raise ValueError(f"Unknown policy type: {policy_type}")
            
            except Exception as e:
                logger.error(f"Error in {policy_name}.select_action(): {e}")
                raise
            
            # Step
            state, reward, terminated, truncated, info = env.step(flat_action)
            done = terminated or truncated
            episode_reward += reward
        
        # Record episode metrics
        episode_rewards.append(episode_reward)
        
        # Get delivery rate
        event_stats = env.buffer.get_event_delivery_stats()
        episode_delivery_rates.append(event_stats['delivery_rate'])
        
        # Completion
        completed = 1.0 if truncated else 0.0
        episode_completion_rates.append(completed)
        
        if terminated:
            total_battery_failures += 1
        
        # Energy efficiency
        energy_eff = (env.episode_stats['energy_harvested'] / 
                     max(env.episode_stats['energy_consumed'], 1.0))
        episode_energy_efficiencies.append(energy_eff)
        
        # Images
        total_images_transmitted += env.episode_stats['images_transmitted']
    
    # Create result
    result = EvaluationResult(
        policy_name=policy_name,
        policy_type=policy_type,
        n_episodes=n_episodes,
        
        episode_rewards=episode_rewards,
        episode_delivery_rates=episode_delivery_rates,
        episode_completion_rates=episode_completion_rates,
        episode_energy_efficiencies=episode_energy_efficiencies,
        
        mean_reward=np.mean(episode_rewards),
        std_reward=np.std(episode_rewards),
        median_reward=np.median(episode_rewards),
        
        mean_delivery=np.mean(episode_delivery_rates),
        std_delivery=np.std(episode_delivery_rates),
        
        mean_completion=np.mean(episode_completion_rates),
        mean_energy_efficiency=np.mean(episode_energy_efficiencies),
        
        total_battery_failures=total_battery_failures,
        total_images_transmitted=total_images_transmitted
    )
    
    if verbose:
        logger.info(f"\n{policy_name} Results:")
        logger.info(f"  Reward: {result.mean_reward:.2f} ± {result.std_reward:.2f}")
        logger.info(f"  Delivery: {result.mean_delivery:.1%} ± {result.std_delivery:.1%}")
        logger.info(f"  Completion: {result.mean_completion:.1%}")
        logger.info(f"  Battery Failures: {total_battery_failures}/{n_episodes}")
    
    return result


# ============================================================================
# COMPREHENSIVE EVALUATION RUNNER
# ============================================================================

def run_comprehensive_evaluation(
    env,
    ppo_agent=None,
    dqn_agent=None,
    mpc_agent=None,
    n_episodes: int = 100,
    save_dir: str = "statistical_evaluation",
    seed: int = 42
) -> Dict[str, EvaluationResult]:
    """
    Run comprehensive evaluation with all available policies.
    
    Args:
        env: Environment
        ppo_agent: Trained PPO agent (optional)
        dqn_agent: Trained DQN agent (optional)
        mpc_agent: MPC agent (optional)
        n_episodes: Number of episodes per policy
        save_dir: Directory to save results
        seed: Random seed
    
    Returns:
        Dictionary of evaluation results
    """
    
    logger.info("="*80)
    logger.info("COMPREHENSIVE POLICY EVALUATION")
    logger.info("="*80)
    logger.info(f"Episodes per policy: {n_episodes}")
    logger.info(f"Random seed: {seed}")
    
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # ========================================================================
    # 1. SIMPLE BASELINE POLICIES
    # ========================================================================
    logger.info("\n📊 SECTION 1: SIMPLE BASELINE POLICIES")
    logger.info("-"*80)
    
    try:
        from baseline_policies import (
            RandomPolicy,
            BatteryThresholdPolicy,
            SmartHeuristicPolicy
        )
        
        baselines = [
            RandomPolicy(),
            BatteryThresholdPolicy(),
            SmartHeuristicPolicy()
        ]
        
        for policy in baselines:
            result = evaluate_policy(env, policy, 'baseline', n_episodes, seed)
            results[policy.name] = result
    
    except ImportError as e:
        logger.warning(f"⚠️  Baseline policies not available: {e}")
    
    # ========================================================================
    # 2. DQN AGENT
    # ========================================================================
    if dqn_agent is not None:
        logger.info("\n🤖 SECTION 2: DQN AGENT")
        logger.info("-"*80)
        
        result = evaluate_policy(env, dqn_agent, 'dqn', n_episodes, seed)
        results['DQN Agent'] = result
    
    # ========================================================================
    # 3. MPC AGENT
    # ========================================================================
    if mpc_agent is not None:
        logger.info("\n🎯 SECTION 3: MPC AGENT")
        logger.info("-"*80)
        
        result = evaluate_policy(env, mpc_agent, 'mpc', n_episodes, seed)
        results['MPC Agent'] = result
    
    # ========================================================================
    # 4. PPO AGENT
    # ========================================================================
    if ppo_agent is not None:
        logger.info("\n🚀 SECTION 4: PPO AGENT")
        logger.info("-"*80)
        
        result = evaluate_policy(env, ppo_agent, 'ppo', n_episodes, seed)
        results['PPO Agent'] = result
    
    # ========================================================================
    # 5. STATISTICAL COMPARISON TABLES
    # ========================================================================
    logger.info("\n"+"="*80)
    logger.info("STATISTICAL SIGNIFICANCE TESTING")
    logger.info("="*80)
    
    if len(results) < 2:
        logger.warning("⚠️  Need at least 2 policies for comparison")
        return results
    
    # Determine reference policy
    if 'PPO Agent' in results:
        ref_policy = 'PPO Agent'
    else:
        # Use policy with highest mean reward
        ref_policy = max(results.values(), key=lambda r: r.mean_reward).policy_name
    
    logger.info(f"\nReference policy: {ref_policy}")
    
    # Create comparison table
    comparison_table = StatisticalComparison.create_comparison_table(
        list(results.values()),
        reference_policy=ref_policy
    )
    
    logger.info("\n" + "="*80)
    logger.info("STATISTICAL COMPARISON TABLE")
    logger.info("="*80)
    logger.info("\nSignificance levels: *** p<0.001, ** p<0.01, * p<0.05, ns: not significant")
    logger.info("\n" + comparison_table.to_string(index=False))
    
    # Save comparison table
    comparison_table.to_csv(
        Path(save_dir) / "statistical_comparison.csv",
        index=False
    )
    logger.info(f"\n✅ Comparison table saved to {save_dir}/statistical_comparison.csv")
    
    # Create pairwise matrix
    if len(results) >= 3:
        pairwise_matrix = StatisticalComparison.create_pairwise_matrix(
            list(results.values()),
            metric='reward'
        )
        
        logger.info("\n" + "="*80)
        logger.info("PAIRWISE SIGNIFICANCE MATRIX (p-values)")
        logger.info("="*80)
        logger.info("\n" + pairwise_matrix.to_string())
        
        pairwise_matrix.to_csv(
            Path(save_dir) / "pairwise_significance.csv"
        )
    
    # ========================================================================
    # 6. VISUALIZATIONS
    # ========================================================================
    logger.info("\n"+"="*80)
    logger.info("CREATING VISUALIZATIONS")
    logger.info("="*80)
    
    create_evaluation_plots(results, save_dir)
    
    # ========================================================================
    # 7. SAVE DETAILED RESULTS
    # ========================================================================
    logger.info("\n"+"="*80)
    logger.info("SAVING DETAILED RESULTS")
    logger.info("="*80)
    
    # Save summary
    summary = pd.DataFrame([r.to_dict() for r in results.values()])
    summary = summary.sort_values('mean_reward', ascending=False)
    summary.to_csv(Path(save_dir) / "evaluation_summary.csv", index=False)
    logger.info(f"✅ Summary saved to {save_dir}/evaluation_summary.csv")
    
    # Save episode-level data for each policy
    for policy_name, result in results.items():
        policy_data = pd.DataFrame({
            'episode': range(result.n_episodes),
            'reward': result.episode_rewards,
            'delivery_rate': result.episode_delivery_rates,
            'completion': result.episode_completion_rates,
            'energy_efficiency': result.episode_energy_efficiencies
        })
        
        safe_name = policy_name.replace(' ', '_').lower()
        policy_data.to_csv(
            Path(save_dir) / f"episodes_{safe_name}.csv",
            index=False
        )
    
    logger.info(f"✅ Episode-level data saved for all {len(results)} policies")
    
    logger.info("\n"+"="*80)
    logger.info("EVALUATION COMPLETE")
    logger.info("="*80)
    logger.info(f"Results saved to: {save_dir}/")
    
    return results


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_evaluation_plots(
    results: Dict[str, EvaluationResult],
    save_dir: str
):
    """Create comprehensive evaluation visualizations."""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    policy_names = list(results.keys())
    colors = sns.color_palette("husl", len(policy_names))
    
    # 1. Mean Rewards with Error Bars
    ax = axes[0, 0]
    means = [results[name].mean_reward for name in policy_names]
    stds = [results[name].std_reward for name in policy_names]
    x_pos = np.arange(len(policy_names))
    
    ax.bar(x_pos, means, yerr=stds, capsize=5, color=colors, alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(policy_names, rotation=45, ha='right')
    ax.set_ylabel('Mean Reward')
    ax.set_title('Mean Episode Reward (±1 SD)')
    ax.grid(axis='y', alpha=0.3)
    
    # 2. Delivery Rates
    ax = axes[0, 1]
    deliveries = [results[name].mean_delivery * 100 for name in policy_names]
    delivery_stds = [results[name].std_delivery * 100 for name in policy_names]
    
    ax.bar(x_pos, deliveries, yerr=delivery_stds, capsize=5, color=colors, alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(policy_names, rotation=45, ha='right')
    ax.set_ylabel('Delivery Rate (%)')
    ax.set_title('Event Delivery Rate (±1 SD)')
    ax.axhline(y=75, color='red', linestyle='--', alpha=0.5, label='Target: 75%')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # 3. Completion Rates
    ax = axes[0, 2]
    completions = [results[name].mean_completion * 100 for name in policy_names]
    
    ax.bar(x_pos, completions, color=colors, alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(policy_names, rotation=45, ha='right')
    ax.set_ylabel('Completion Rate (%)')
    ax.set_title('Episode Completion Rate')
    ax.axhline(y=95, color='red', linestyle='--', alpha=0.5, label='Target: 95%')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # 4. Reward Distributions (Violin Plot)
    ax = axes[1, 0]
    data_to_plot = [results[name].episode_rewards for name in policy_names]
    parts = ax.violinplot(data_to_plot, positions=x_pos, showmeans=True, showmedians=True)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(policy_names, rotation=45, ha='right')
    ax.set_ylabel('Reward')
    ax.set_title('Reward Distributions')
    ax.grid(axis='y', alpha=0.3)
    
    # 5. Delivery vs Completion Scatter
    ax = axes[1, 1]
    for i, name in enumerate(policy_names):
        ax.scatter(
            results[name].mean_delivery * 100,
            results[name].mean_completion * 100,
            s=200, c=[colors[i]], alpha=0.7, label=name
        )
    
    ax.set_xlabel('Delivery Rate (%)')
    ax.set_ylabel('Completion Rate (%)')
    ax.set_title('Delivery vs Completion Trade-off')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(alpha=0.3)
    
    # 6. Energy Efficiency
    ax = axes[1, 2]
    efficiencies = [results[name].mean_energy_efficiency for name in policy_names]
    
    ax.bar(x_pos, efficiencies, color=colors, alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(policy_names, rotation=45, ha='right')
    ax.set_ylabel('Energy Efficiency Ratio')
    ax.set_title('Energy Efficiency (Harvested/Consumed)')
    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='Break-even')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(Path(save_dir) / "evaluation_plots.png", dpi=300, bbox_inches='tight')
    logger.info(f"✅ Plots saved to {save_dir}/evaluation_plots.png")
    
    plt.close()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    logger.info("\n"+"="*80)
    logger.info("STATISTICAL EVALUATION SYSTEM")
    logger.info("="*80)
    
    # Run compatibility check
    checker = CompatibilityChecker()
    compatibility = checker.run_all_checks()
    
    # Check if we can proceed
    if compatibility.get('environment') != 'OK':
        logger.error("\n❌ Cannot proceed without environment files!")
        sys.exit(1)
    
    if compatibility.get('baseline_policies') != 'OK':
        logger.error("\n❌ Cannot proceed without baseline_policies.py!")
        sys.exit(1)
    
    logger.info("\n✅ System is ready for evaluation!")
    logger.info("\nNext steps:")
    logger.info("1. Load your trained PPO agent")
    logger.info("2. (Optional) Train/load DQN and MPC agents")
    logger.info("3. Run: run_comprehensive_evaluation(env, ppo_agent=agent)")
