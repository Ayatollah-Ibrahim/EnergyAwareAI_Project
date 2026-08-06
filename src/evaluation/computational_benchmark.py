"""
Computational Efficiency Benchmark
===================================

Benchmark PPO, DQN, and MPC based on:
- Inference time (ms per decision)
- FLOPs (floating point operations)
- Memory usage
- Throughput (decisions per second)

Author: Computational Benchmarking
Date: 2026-02-07
"""

import numpy as np
import time
import torch
import psutil
import os
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Try to import FLOPs counter
try:
    from thop import profile, clever_format
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False
    logger.warning("⚠️  'thop' not available. Install with: pip install thop")
    logger.warning("   FLOPs counting will use estimation instead.")


@dataclass
class ComputationalMetrics:
    """Metrics for computational efficiency."""
    
    policy_name: str
    policy_type: str  # 'ppo', 'dqn', 'mpc', 'baseline'
    
    # Timing metrics (milliseconds)
    mean_inference_time: float
    std_inference_time: float
    min_inference_time: float
    max_inference_time: float
    median_inference_time: float
    p95_inference_time: float  # 95th percentile
    
    # Throughput
    decisions_per_second: float
    
    # FLOPs
    total_flops: float  # Total floating point operations
    flops_per_decision: float
    
    # Memory
    model_parameters: int  # Number of parameters
    model_size_mb: float  # Model size in MB
    peak_memory_mb: float  # Peak memory during inference
    
    # Additional info
    n_measurements: int
    device: str  # 'cpu' or 'cuda'
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


class ComputationalBenchmark:
    """Benchmark computational efficiency of policies."""
    
    def __init__(
        self,
        env,
        n_warmup: int = 10,
        n_measurements: int = 1000,
        device: str = 'cpu'
    ):
        """
        Initialize benchmark.
        
        Args:
            env: Environment (for observation shape)
            n_warmup: Number of warmup iterations
            n_measurements: Number of measurements for timing
            device: 'cpu' or 'cuda'
        """
        self.env = env
        self.n_warmup = n_warmup
        self.n_measurements = n_measurements
        self.device = device
        
        # Get observation shape
        self.obs_shape = env.observation_space.shape
        self.state_dim = self.obs_shape[0]
        
        logger.info(f"Computational Benchmark initialized")
        logger.info(f"  State dimension: {self.state_dim}")
        logger.info(f"  Warmup iterations: {n_warmup}")
        logger.info(f"  Measurements: {n_measurements}")
        logger.info(f"  Device: {device}")
    
    def _warmup(self, policy, policy_type: str):
        """Warmup policy to get consistent timing."""
        logger.info(f"  Warming up {policy.name if hasattr(policy, 'name') else 'policy'}...")
        
        for _ in range(self.n_warmup):
            obs = np.random.randn(self.state_dim).astype(np.float32)
            
            if policy_type == 'ppo':
                _ = policy.select_action(obs, deterministic=True)
            elif policy_type == 'dqn':
                _ = policy.select_action(obs, deterministic=True)
            elif policy_type == 'mpc':
                _ = policy.select_action(obs, {})
            elif policy_type == 'baseline':
                _ = policy.select_action(obs, {})
    
    def _measure_inference_time(
        self,
        policy,
        policy_type: str
    ) -> Tuple[List[float], int]:
        """
        Measure inference time.
        
        Returns:
            Tuple of (inference_times, n_successful)
        """
        inference_times = []
        n_successful = 0
        
        for _ in range(self.n_measurements):
            obs = np.random.randn(self.state_dim).astype(np.float32)
            
            # Time the inference
            start = time.perf_counter()
            
            try:
                if policy_type == 'ppo':
                    _ = policy.select_action(obs, deterministic=True)
                elif policy_type == 'dqn':
                    _ = policy.select_action(obs, deterministic=True)
                elif policy_type == 'mpc':
                    _ = policy.select_action(obs, {})
                elif policy_type == 'baseline':
                    _ = policy.select_action(obs, {})
                
                elapsed = (time.perf_counter() - start) * 1000  # Convert to ms
                inference_times.append(elapsed)
                n_successful += 1
                
            except Exception as e:
                logger.warning(f"    Inference failed: {e}")
                continue
        
        return inference_times, n_successful
    
    def _count_flops_torch(self, model, input_shape: Tuple) -> Tuple[float, float]:
        """
        Count FLOPs for PyTorch model using thop.
        
        Returns:
            Tuple of (total_flops, flops_per_forward)
        """
        if not THOP_AVAILABLE:
            logger.warning("  thop not available, using estimation")
            return self._estimate_flops(model)
        
        try:
            # Create dummy input
            dummy_input = torch.randn(1, *input_shape).to(self.device)
            
            # Profile model
            flops, params = profile(
                model,
                inputs=(dummy_input,),
                verbose=False
            )
            
            return flops, flops
            
        except Exception as e:
            logger.warning(f"  FLOPs counting failed: {e}, using estimation")
            return self._estimate_flops(model)
    
    def _estimate_flops(self, model) -> Tuple[float, float]:
        """
        Estimate FLOPs based on model parameters.
        
        Rough estimate: 2 FLOPs per parameter (multiply-add)
        """
        if hasattr(model, 'parameters'):
            n_params = sum(p.numel() for p in model.parameters())
            estimated_flops = 2 * n_params  # Rough estimate
            return estimated_flops, estimated_flops
        else:
            return 0.0, 0.0
    
    def _count_parameters(self, policy, policy_type: str) -> int:
        """Count trainable parameters."""
        if policy_type in ['ppo', 'dqn']:
            # Neural network policies
            total_params = 0
            
            if hasattr(policy, 'actor'):
                total_params += sum(p.numel() for p in policy.actor.parameters())
            if hasattr(policy, 'critic'):
                total_params += sum(p.numel() for p in policy.critic.parameters())
            if hasattr(policy, 'q_network'):
                total_params += sum(p.numel() for p in policy.q_network.parameters())
            
            return total_params
        else:
            # Non-neural policies (MPC, baselines)
            return 0
    
    def _measure_model_size(self, policy, policy_type: str) -> float:
        """
        Measure model size in MB.
        
        Returns:
            Size in megabytes
        """
        if policy_type not in ['ppo', 'dqn']:
            return 0.0
        
        # Save to temporary file and measure size
        import tempfile
        
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=True) as tmp:
            try:
                if policy_type == 'ppo':
                    torch.save({
                        'actor': policy.actor.state_dict(),
                        'critic': policy.critic.state_dict()
                    }, tmp.name)
                elif policy_type == 'dqn':
                    torch.save({
                        'q_network': policy.q_network.state_dict()
                    }, tmp.name)
                
                size_bytes = os.path.getsize(tmp.name)
                size_mb = size_bytes / (1024 * 1024)
                return size_mb
                
            except Exception as e:
                logger.warning(f"  Could not measure model size: {e}")
                return 0.0
    
    def _measure_memory_usage(
        self,
        policy,
        policy_type: str
    ) -> float:
        """
        Measure peak memory usage during inference.
        
        Returns:
            Peak memory in MB
        """
        process = psutil.Process(os.getpid())
        
        # Get baseline memory
        baseline_memory = process.memory_info().rss / (1024 * 1024)
        
        # Run inference
        obs = np.random.randn(self.state_dim).astype(np.float32)
        
        if policy_type == 'ppo':
            _ = policy.select_action(obs, deterministic=True)
        elif policy_type == 'dqn':
            _ = policy.select_action(obs, deterministic=True)
        elif policy_type == 'mpc':
            _ = policy.select_action(obs, {})
        elif policy_type == 'baseline':
            _ = policy.select_action(obs, {})
        
        # Get peak memory
        peak_memory = process.memory_info().rss / (1024 * 1024)
        
        return peak_memory - baseline_memory
    
    def _count_flops_policy(
        self,
        policy,
        policy_type: str
    ) -> Tuple[float, float]:
        """Count FLOPs for a policy."""
        
        if policy_type == 'ppo':
            # Count FLOPs for actor network
            actor_flops, _ = self._count_flops_torch(
                policy.actor,
                (self.state_dim,)
            )
            return actor_flops, actor_flops
        
        elif policy_type == 'dqn':
            # Count FLOPs for Q-network
            q_flops, _ = self._count_flops_torch(
                policy.q_network,
                (self.state_dim,)
            )
            return q_flops, q_flops
        
        elif policy_type == 'mpc':
            # Estimate FLOPs for MPC
            # MPC does N simulations of H steps
            horizon = policy.config.horizon
            n_samples = policy.config.n_random_samples
            
            # Each simulation step: ~100 FLOPs (energy calculations, etc.)
            flops_per_simulation = horizon * 100
            total_flops = n_samples * flops_per_simulation
            
            return total_flops, total_flops
        
        else:  # baseline
            # Simple policies: negligible FLOPs (~10-100)
            return 100.0, 100.0
    
    def benchmark_policy(
        self,
        policy,
        policy_type: str,
        policy_name: Optional[str] = None
    ) -> ComputationalMetrics:
        """
        Benchmark a single policy.
        
        Args:
            policy: Policy to benchmark
            policy_type: 'ppo', 'dqn', 'mpc', 'baseline'
            policy_name: Optional name (auto-detected if not provided)
        
        Returns:
            ComputationalMetrics object
        """
        
        if policy_name is None:
            policy_name = getattr(policy, 'name', f'{policy_type.upper()} Agent')
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Benchmarking: {policy_name}")
        logger.info(f"{'='*60}")
        
        # Warmup
        self._warmup(policy, policy_type)
        
        # Measure inference time
        logger.info(f"  Measuring inference time ({self.n_measurements} iterations)...")
        inference_times, n_successful = self._measure_inference_time(policy, policy_type)
        
        if n_successful == 0:
            logger.error(f"  All inference attempts failed!")
            return None
        
        inference_times = np.array(inference_times)
        
        # Compute timing statistics
        mean_time = np.mean(inference_times)
        std_time = np.std(inference_times)
        min_time = np.min(inference_times)
        max_time = np.max(inference_times)
        median_time = np.median(inference_times)
        p95_time = np.percentile(inference_times, 95)
        
        # Throughput
        throughput = 1000.0 / mean_time  # decisions per second
        
        logger.info(f"  ✓ Mean: {mean_time:.3f} ms")
        logger.info(f"  ✓ Std: {std_time:.3f} ms")
        logger.info(f"  ✓ Median: {median_time:.3f} ms")
        logger.info(f"  ✓ P95: {p95_time:.3f} ms")
        logger.info(f"  ✓ Throughput: {throughput:.1f} decisions/sec")
        
        # Count FLOPs
        logger.info(f"  Counting FLOPs...")
        total_flops, flops_per_decision = self._count_flops_policy(policy, policy_type)
        
        logger.info(f"  ✓ FLOPs: {total_flops:,.0f}")
        
        # Count parameters
        logger.info(f"  Counting parameters...")
        n_params = self._count_parameters(policy, policy_type)
        logger.info(f"  ✓ Parameters: {n_params:,}")
        
        # Measure model size
        logger.info(f"  Measuring model size...")
        model_size = self._measure_model_size(policy, policy_type)
        logger.info(f"  ✓ Model size: {model_size:.2f} MB")
        
        # Measure memory usage
        logger.info(f"  Measuring memory usage...")
        peak_memory = self._measure_memory_usage(policy, policy_type)
        logger.info(f"  ✓ Peak memory: {peak_memory:.2f} MB")
        
        # Create metrics object
        metrics = ComputationalMetrics(
            policy_name=policy_name,
            policy_type=policy_type,
            mean_inference_time=mean_time,
            std_inference_time=std_time,
            min_inference_time=min_time,
            max_inference_time=max_time,
            median_inference_time=median_time,
            p95_inference_time=p95_time,
            decisions_per_second=throughput,
            total_flops=total_flops,
            flops_per_decision=flops_per_decision,
            model_parameters=n_params,
            model_size_mb=model_size,
            peak_memory_mb=peak_memory,
            n_measurements=n_successful,
            device=self.device
        )
        
        return metrics
    
    def create_comparison_table(
        self,
        metrics_list: List[ComputationalMetrics]
    ) -> pd.DataFrame:
        """Create comparison table."""
        
        data = []
        for m in metrics_list:
            data.append({
                'Policy': m.policy_name,
                'Type': m.policy_type.upper(),
                'Inference (ms)': f"{m.mean_inference_time:.3f} ± {m.std_inference_time:.3f}",
                'P95 (ms)': f"{m.p95_inference_time:.3f}",
                'Throughput (Hz)': f"{m.decisions_per_second:.1f}",
                'FLOPs': f"{m.total_flops:,.0f}",
                'Parameters': f"{m.model_parameters:,}",
                'Model Size (MB)': f"{m.model_size_mb:.2f}",
                'Memory (MB)': f"{m.peak_memory_mb:.2f}"
            })
        
        df = pd.DataFrame(data)
        
        # Sort by inference time
        df = df.sort_values('Inference (ms)')
        
        return df
    
    def plot_comparison(
        self,
        metrics_list: List[ComputationalMetrics],
        save_path: Optional[str] = None
    ):
        """Create comparison plots."""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Computational Efficiency Comparison', 
                     fontsize=16, fontweight='bold')
        
        # Extract data
        names = [m.policy_name for m in metrics_list]
        colors = sns.color_palette("husl", len(names))
        
        # 1. Inference Time (bar plot with error bars)
        ax = axes[0, 0]
        means = [m.mean_inference_time for m in metrics_list]
        stds = [m.std_inference_time for m in metrics_list]
        x_pos = np.arange(len(names))
        
        ax.bar(x_pos, means, yerr=stds, capsize=5, color=colors, alpha=0.7)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.set_ylabel('Inference Time (ms)')
        ax.set_title('Mean Inference Time')
        ax.grid(axis='y', alpha=0.3)
        
        # Add real-time threshold
        ax.axhline(y=100, color='red', linestyle='--', alpha=0.5, 
                  label='Real-time (100ms)')
        ax.legend()
        
        # 2. Throughput
        ax = axes[0, 1]
        throughputs = [m.decisions_per_second for m in metrics_list]
        
        ax.bar(x_pos, throughputs, color=colors, alpha=0.7)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.set_ylabel('Decisions per Second')
        ax.set_title('Throughput')
        ax.grid(axis='y', alpha=0.3)
        
        # 3. FLOPs
        ax = axes[0, 2]
        flops = [m.total_flops for m in metrics_list]
        
        ax.bar(x_pos, flops, color=colors, alpha=0.7)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.set_ylabel('FLOPs')
        ax.set_title('Computational Complexity')
        ax.set_yscale('log')
        ax.grid(axis='y', alpha=0.3)
        
        # 4. Parameters
        ax = axes[1, 0]
        params = [m.model_parameters for m in metrics_list]
        
        ax.bar(x_pos, params, color=colors, alpha=0.7)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.set_ylabel('Parameters')
        ax.set_title('Model Parameters')
        ax.grid(axis='y', alpha=0.3)
        
        # 5. Model Size
        ax = axes[1, 1]
        sizes = [m.model_size_mb for m in metrics_list]
        
        ax.bar(x_pos, sizes, color=colors, alpha=0.7)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.set_ylabel('Size (MB)')
        ax.set_title('Model Size')
        ax.grid(axis='y', alpha=0.3)
        
        # 6. Memory Usage
        ax = axes[1, 2]
        memory = [m.peak_memory_mb for m in metrics_list]
        
        ax.bar(x_pos, memory, color=colors, alpha=0.7)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(names, rotation=45, ha='right')
        ax.set_ylabel('Memory (MB)')
        ax.set_title('Peak Memory Usage')
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        plt.show()


def run_computational_benchmark(
    env,
    ppo_agent=None,
    dqn_agent=None,
    mpc_agent=None,
    baseline_policies: Optional[List] = None,
    n_measurements: int = 1000,
    save_dir: str = "computational_benchmark",
    device: str = 'cpu'
) -> Dict[str, ComputationalMetrics]:
    """
    Run complete computational benchmark.
    
    Args:
        env: Environment
        ppo_agent: PPO agent (optional)
        dqn_agent: DQN agent (optional)
        mpc_agent: MPC agent (optional)
        baseline_policies: List of baseline policies (optional)
        n_measurements: Number of timing measurements
        save_dir: Directory to save results
        device: 'cpu' or 'cuda'
    
    Returns:
        Dictionary of metrics
    """
    
    logger.info("\n" + "="*80)
    logger.info("COMPUTATIONAL EFFICIENCY BENCHMARK")
    logger.info("="*80)
    
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # Create benchmark
    benchmark = ComputationalBenchmark(
        env,
        n_warmup=10,
        n_measurements=n_measurements,
        device=device
    )
    
    metrics_list = []
    
    # Benchmark PPO
    if ppo_agent is not None:
        logger.info("\n🚀 Benchmarking PPO Agent...")
        metrics = benchmark.benchmark_policy(ppo_agent, 'ppo', 'PPO Agent')
        if metrics:
            metrics_list.append(metrics)
    
    # Benchmark DQN
    if dqn_agent is not None:
        logger.info("\n🤖 Benchmarking DQN Agent...")
        metrics = benchmark.benchmark_policy(dqn_agent, 'dqn', 'DQN Agent')
        if metrics:
            metrics_list.append(metrics)
    
    # Benchmark MPC
    if mpc_agent is not None:
        logger.info("\n🎯 Benchmarking MPC Agent...")
        metrics = benchmark.benchmark_policy(mpc_agent, 'mpc', 'MPC Agent')
        if metrics:
            metrics_list.append(metrics)
    
    # Benchmark baselines
    if baseline_policies:
        logger.info("\n📊 Benchmarking Baseline Policies...")
        for policy in baseline_policies:
            metrics = benchmark.benchmark_policy(
                policy,
                'baseline',
                policy.name if hasattr(policy, 'name') else 'Baseline'
            )
            if metrics:
                metrics_list.append(metrics)
    
    # Create comparison table
    logger.info("\n" + "="*80)
    logger.info("COMPARISON TABLE")
    logger.info("="*80)
    
    comparison_table = benchmark.create_comparison_table(metrics_list)
    print("\n" + comparison_table.to_string(index=False))
    
    # Save table
    comparison_table.to_csv(
        Path(save_dir) / "computational_comparison.csv",
        index=False
    )
    logger.info(f"\n✅ Table saved to {save_dir}/computational_comparison.csv")
    
    # Save detailed metrics
    detailed_data = [m.to_dict() for m in metrics_list]
    with open(Path(save_dir) / "detailed_metrics.json", 'w') as f:
        json.dump(detailed_data, f, indent=2)
    logger.info(f"✅ Detailed metrics saved to {save_dir}/detailed_metrics.json")
    
    # Create plots
    logger.info("\n📊 Creating plots...")
    benchmark.plot_comparison(
        metrics_list,
        save_path=Path(save_dir) / "computational_comparison.png"
    )
    
    logger.info("\n" + "="*80)
    logger.info("BENCHMARK COMPLETE")
    logger.info("="*80)
    logger.info(f"Results saved to: {save_dir}/")
    
    # Convert to dictionary
    results = {m.policy_name: m for m in metrics_list}
    
    return results


if __name__ == "__main__":
    logger.info(__doc__)
