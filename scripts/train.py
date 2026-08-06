"""
PPO Training for Energy Harvesting Camera System - Jetson Nano
Part 7: Main Execution Script

This is the main entry point that ties everything together.

Usage:
    python PPO_Jetson_Main.py --mode train
    python PPO_Jetson_Main.py --mode test
    python PPO_Jetson_Main.py --mode diagnose

Author: Revised Implementation
Date: 2026-02-06
"""

import argparse
import sys
from typing import List
from pathlib import Path

# ============================================================================
# IMPORTS (assumes all parts are available)
# ============================================================================

# Part 1: Core infrastructure
from src.env.core import (
    SystemParameters,
    create_curriculum_stages,
    GHILoader,
    create_synthetic_ghi_data
)

# Part 2: Energy model
from src.env.energy import (
    EnergyModel,
    EnhancedDualBufferManager,
    BatteryManager
)

# Part 3: Environment
from src.env.environment import EnergyHarvestingCameraEnv

# Part 4: PPO
from src.agents.ppo import (
    PPOConfig,
    PPOAgent
)

# Part 5: Training
from src.training.trainer import (
    train_curriculum,
    evaluate_policy,
    plot_curriculum_results
)

# Part 6: Diagnostics
from src.evaluation.diagnostics import (
    run_preflight_checks,
    test_curriculum_params,
    quick_diagnostic_test,
    diagnose_pipeline_bottleneck,
    analyze_agent_behavior,
    compare_curriculum_stages
)

import logging

logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_DATA_PATH = "NREL.csv"
DEFAULT_SAVE_DIR = "experiments_Jetson/main_training"
DEFAULT_N_EVAL_EPISODES = 10

# ============================================================================
# MODE: DIAGNOSE
# ============================================================================

def run_diagnostics(args):
    """Run diagnostic tests."""
    logger.info("=" * 80)
    logger.info("RUNNING DIAGNOSTICS")
    logger.info("=" * 80)
    
    # Ensure GHI data exists
    if not Path(DEFAULT_DATA_PATH).exists():
        logger.info(f"Creating synthetic GHI data at {DEFAULT_DATA_PATH}")
        create_synthetic_ghi_data(DEFAULT_DATA_PATH)
    
    # Run all pre-flight checks
    success = run_preflight_checks()
    
    if success:
        logger.info("\n✅ All diagnostics passed! System is ready for training.")
        return 0
    else:
        logger.warning("\n⚠️  Some diagnostics failed. Please review before training.")
        return 1

# ============================================================================
# MODE: TRAIN
# ============================================================================

def run_training(args):
    """Run full curriculum training."""
    logger.info("=" * 80)
    logger.info("STARTING TRAINING")
    logger.info("=" * 80)
    
    # Ensure GHI data exists
    if not Path(args.data_path).exists():
        logger.info(f"Creating synthetic GHI data at {args.data_path}")
        create_synthetic_ghi_data(args.data_path)
    
    # Optional: Run pre-flight checks
    if args.skip_checks:
        logger.info("Skipping pre-flight checks (--skip-checks enabled)")
    else:
        logger.info("Running pre-flight checks...")
        success = run_preflight_checks()
        if not success and not args.force:
            logger.error("Pre-flight checks failed! Use --force to train anyway.")
            return 1
        elif not success:
            logger.warning("Pre-flight checks failed but continuing (--force enabled)")
    
    # Load curriculum stages
    if args.stages:
        logger.info(f"Loading custom stages from {args.stages}")
        import json
        with open(args.stages, 'r') as f:
            stage_configs = json.load(f)
        # TODO: Convert to CurriculumStage objects
        stages = None  # Use default for now
    else:
        stages = None  # Use default curriculum
    
    # Train
    logger.info(f"\nStarting curriculum training...")
    logger.info(f"  Data: {args.data_path}")
    logger.info(f"  Save: {args.save_dir}")
    logger.info(f"  Eval interval: {args.eval_interval}")
    
    agent, history = train_curriculum(
        data_path=args.data_path,
        stages=stages,
        save_dir=args.save_dir,
        eval_interval=args.eval_interval
    )
    
    # Final evaluation
    logger.info("\n" + "=" * 80)
    logger.info("FINAL EVALUATION")
    logger.info("=" * 80)
    
    params = SystemParameters()
    env = EnergyHarvestingCameraEnv(args.data_path, params)
    
    final_reward = evaluate_policy(env, agent, n_episodes=args.n_eval)
    logger.info(f"Final evaluation reward: {final_reward:.2f}")
    
    env.close()
    
    # Stage comparison
    compare_curriculum_stages(history)
    
    logger.info("\n✅ Training completed successfully!")
    logger.info(f"   Results saved to: {args.save_dir}")
    
    return 0

# ============================================================================
# MODE: TEST
# ============================================================================

def run_testing(args):
    """Test a trained agent."""
    logger.info("=" * 80)
    logger.info("TESTING TRAINED AGENT")
    logger.info("=" * 80)
    
    if not Path(args.agent_path).exists():
        logger.error(f"Agent file not found: {args.agent_path}")
        return 1
    
    # Load agent
    logger.info(f"Loading agent from {args.agent_path}")
    
    params = SystemParameters()
    env = EnergyHarvestingCameraEnv(args.data_path, params)
    
    config = PPOConfig(state_dim=env.observation_space.shape[0])
    agent = PPOAgent(config)
    agent.load(args.agent_path)
    
    logger.info("✅ Agent loaded successfully")
    
    # Evaluate
    logger.info(f"\nEvaluating for {args.n_eval} episodes...")
    eval_reward = evaluate_policy(env, agent, n_episodes=args.n_eval)
    
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 80)
    logger.info(f"Mean reward: {eval_reward:.2f}")
    
    # Analyze behavior
    if args.analyze:
        logger.info("\nAnalyzing agent behavior...")
        analyze_agent_behavior(env, agent, n_steps=1440)
    
    env.close()
    
    logger.info("\n✅ Testing completed successfully!")
    
    return 0

# ============================================================================
# MODE: ANALYZE
# ============================================================================

def run_analysis(args):
    """Analyze training results."""
    logger.info("=" * 80)
    logger.info("ANALYZING RESULTS")
    logger.info("=" * 80)
    
    history_path = Path(args.save_dir) / "full_history.json"
    
    if not history_path.exists():
        logger.error(f"History file not found: {history_path}")
        return 1
    
    # Load history
    import json
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    logger.info(f"Loaded history from {history_path}")
    
    # Compare stages
    compare_curriculum_stages(history)
    
    # Plot results
    plot_path = Path(args.save_dir) / "analysis_plot.png"
    plot_curriculum_results(history, save_path=str(plot_path))
    
    logger.info(f"\n✅ Analysis completed! Plot saved to {plot_path}")
    
    return 0

# ============================================================================
# MAIN
# ============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="PPO Training for Energy Harvesting Camera System"
    )
    
    # Mode
    parser.add_argument(
        '--mode',
        type=str,
        choices=['train', 'test', 'diagnose', 'analyze'],
        default='train',
        help='Execution mode'
    )
    
    # Data
    parser.add_argument(
        '--data-path',
        type=str,
        default=DEFAULT_DATA_PATH,
        help='Path to GHI CSV data'
    )
    
    # Training
    parser.add_argument(
        '--save-dir',
        type=str,
        default=DEFAULT_SAVE_DIR,
        help='Directory for saving results'
    )
    
    parser.add_argument(
        '--stages',
        type=str,
        default=None,
        help='Path to custom curriculum stages JSON'
    )
    
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=100,
        help='Episodes between evaluations'
    )
    
    parser.add_argument(
        '--skip-checks',
        action='store_true',
        help='Skip pre-flight checks'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='Train even if checks fail'
    )
    
    # Testing
    parser.add_argument(
        '--agent-path',
        type=str,
        default=None,
        help='Path to trained agent checkpoint'
    )
    
    parser.add_argument(
        '--n-eval',
        type=int,
        default=DEFAULT_N_EVAL_EPISODES,
        help='Number of evaluation episodes'
    )
    
    parser.add_argument(
        '--analyze',
        action='store_true',
        help='Analyze agent behavior during testing'
    )
    
    # Logging
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    logger.info("=" * 80)
    logger.info("PPO TRAINING FOR ENERGY HARVESTING CAMERA SYSTEM")
    logger.info("Jetson Nano - Revised Implementation")
    logger.info("=" * 80)
    
    # Execute mode
    if args.mode == 'diagnose':
        return run_diagnostics(args)
    elif args.mode == 'train':
        return run_training(args)
    elif args.mode == 'test':
        return run_testing(args)
    elif args.mode == 'analyze':
        return run_analysis(args)
    else:
        logger.error(f"Unknown mode: {args.mode}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
