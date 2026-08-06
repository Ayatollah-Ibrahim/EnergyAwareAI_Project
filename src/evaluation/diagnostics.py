"""
PPO Training for Energy Harvesting Camera System - Jetson Nano
Part 6: Testing, Diagnostics, and Analysis

This module provides:
- Pre-flight checks
- Pipeline diagnostics
- Agent behavior analysis
- Parameter verification
- Quick diagnostic tests

Author: Revised Implementation
Date: 2026-02-06
"""

import numpy as np
import matplotlib.pyplot as plt
import logging
from typing import Dict, Optional, Tuple, List
from pathlib import Path

logger = logging.getLogger(__name__)

# ============================================================================
# PARAMETER VERIFICATION
# ============================================================================

def test_curriculum_params():
    """Verify curriculum parameters are correctly configured."""
    from src.env.core import create_curriculum_stages, SystemParameters
    
    stages = create_curriculum_stages()
    
    print("\n" + "=" * 80)
    print("CURRICULUM PARAMETER VERIFICATION")
    print("=" * 80)
    
    for stage in stages:
        params = stage.apply_to_params(SystemParameters())
        print(f"\n{stage.name} Stage:")
        print(f"  Event Probability:   {params.prob_event_per_step:.2%}")
        print(f"  Images per Event:    {params.images_per_event_mean:.0f}")
        print(f"  TX Capacity:         {params.n_tx_img_max}")
        print(f"  Event Timeout:       {stage.event_timeout}")
        print(f"  Battery Start:       {params.initial_battery_soc:.0%}")
        
        # Theoretical delivery rate
        events_per_100 = 100 * params.prob_event_per_step
        images_per_100 = events_per_100 * params.images_per_event_mean
        max_tx_per_100 = 100 * params.n_tx_img_max
        theoretical_max = min(1.0, max_tx_per_100 / images_per_100)
        
        print(f"  Theoretical Max Delivery: {theoretical_max:.1%}")
        
        if stage.name == "Tutorial" and theoretical_max < 0.8:
            print(f"  ⚠️  WARNING: Theoretical max is only {theoretical_max:.1%}")
        elif stage.name == "Tutorial":
            print(f"  ✅ Tutorial should achieve {theoretical_max:.1%} delivery!")
    
    print("=" * 80)

# ============================================================================
# PIPELINE DIAGNOSTICS
# ============================================================================

def diagnose_pipeline_bottleneck(env, n_steps: int = 100) -> Dict:
    """
    Diagnose pipeline bottlenecks.
    
    Args:
        env: Environment
        n_steps: Number of steps to run
    
    Returns:
        Statistics dictionary
    """
    obs, _ = env.reset(seed=42)
    
    stats = {
        'images_captured': 0,
        'images_processed': 0,
        'images_transmitted': 0,
        'buffer_raw_overflows': 0,
        'buffer_proc_overflows': 0,
        'events_generated': 0,
        'step_transmissions': []
    }
    
    for step in range(n_steps):
        # Aggressive policy: max effort
        action = env.multidiscrete_to_action[(0, 2, 2, 1)]
        obs, reward, terminated, truncated, info = env.step(action)
        
        stats['images_transmitted'] += info['n_transmitted']
        stats['buffer_raw_overflows'] += info['n_dropped_raw']
        stats['buffer_proc_overflows'] += info['n_dropped_proc']
        stats['step_transmissions'].append(info['n_transmitted'])
        
        if info['current_event_id'] is not None:
            stats['events_generated'] += 1
        
        if terminated or truncated:
            break
    
    print("\n" + "=" * 80)
    print("PIPELINE BOTTLENECK ANALYSIS")
    print("=" * 80)
    print(f"Environment: {env.params.get_curriculum_stage_name()}")
    print(f"Images per Event: {env.params.images_per_event_mean:.0f}")
    print(f"TX Capacity: {env.params.n_tx_img_max}")
    print(f"Event Timeout: {env.event_timeout}")
    print("-" * 80)
    print(f"Events Generated:       {stats['events_generated']}")
    print(f"Images Transmitted:     {stats['images_transmitted']}")
    print(f"Raw Buffer Overflows:   {stats['buffer_raw_overflows']}")
    print(f"Proc Buffer Overflows:  {stats['buffer_proc_overflows']}")
    print(f"\nTransmission Rate:      {stats['images_transmitted'] / n_steps:.1f} images/step")
    print(f"Theoretical Max:        {env.params.n_tx_img_max:.0f} images/step")
    print(f"TX Utilization:         {stats['images_transmitted'] / (n_steps * env.params.n_tx_img_max):.1%}")
    print("=" * 80)
    
    # Delivery rate
    event_stats = env.buffer.get_event_delivery_stats()
    print(f"\nFinal Delivery Rate:    {event_stats['delivery_rate']:.1%}")
    print(f"Important Events:       {event_stats['total_important_events']}")
    print(f"Delivered:              {event_stats['important_delivered']}")
    print(f"Missed:                 {event_stats['important_missed']}")
    
    if event_stats['delivery_rate'] < 0.5:
        print("\n⚠️  WARNING: Delivery rate is very low!")
        print("   Consider:")
        print("   - Increasing n_tx_img_max")
        print("   - Decreasing images_per_event_mean")
        print("   - Decreasing prob_event_per_step")
        print("   - Increasing event_timeout")
    elif event_stats['delivery_rate'] > 0.7:
        print("\n✅ Delivery rate looks good!")
    
    print("=" * 80)
    
    return stats

# ============================================================================
# AGENT BEHAVIOR ANALYSIS
# ============================================================================

def analyze_agent_behavior(
    env, 
    agent, 
    n_steps: int = 1440
) -> Dict[str, Dict[int, int]]:
    """
    Analyze agent action distribution.
    
    Args:
        env: Environment
        agent: Trained agent
        n_steps: Number of steps
    
    Returns:
        Action counts dictionary
    """
    state, _ = env.reset(seed=42)
    
    action_counts = {
        'sleep': {0: 0, 1: 0, 2: 0},
        'capture': {0: 0, 1: 0, 2: 0},
        'process': {0: 0, 1: 0, 2: 0},
        'tx': {0: 0, 1: 0}
    }
    
    for step in range(n_steps):
        action, _, _ = agent.select_action(state, deterministic=True)
        mode, capture, process, tx = action
        
        action_counts['sleep'][mode] += 1
        action_counts['capture'][capture] += 1
        action_counts['process'][process] += 1
        action_counts['tx'][tx] += 1
        
        flat_action = mode * 18 + capture * 6 + process * 2 + tx
        state, _, terminated, truncated, _ = env.step(flat_action)
        
        if terminated or truncated:
            break
    
    print("=" * 80)
    print("AGENT ACTION DISTRIBUTION")
    print("=" * 80)
    
    total_steps = sum(action_counts['sleep'].values())
    
    print("\nSleep Mode:")
    for mode, label in [(0, "Active"), (1, "Idle"), (2, "Deep Sleep")]:
        count = action_counts['sleep'][mode]
        pct = count / total_steps * 100
        print(f"  {label:12s} ({mode}): {count:4d} ({pct:5.1f}%)")
    
    print("\nCapture Mode:")
    for mode, label in [(0, "Off"), (1, "Low"), (2, "High")]:
        count = action_counts['capture'][mode]
        pct = count / total_steps * 100
        print(f"  {label:12s} ({mode}): {count:4d} ({pct:5.1f}%)")
    
    print("\nProcess Mode:")
    for mode, label in [(0, "Off"), (1, "Simple"), (2, "Complex")]:
        count = action_counts['process'][mode]
        pct = count / total_steps * 100
        print(f"  {label:12s} ({mode}): {count:4d} ({pct:5.1f}%)")
    
    print("\nTransmission:")
    for mode, label in [(0, "Off"), (1, "On")]:
        count = action_counts['tx'][mode]
        pct = count / total_steps * 100
        print(f"  {label:12s} ({mode}): {count:4d} ({pct:5.1f}%)")
    
    print("=" * 80)
    
    return action_counts

# ============================================================================
# QUICK DIAGNOSTIC TEST
# ============================================================================

def quick_diagnostic_test() -> Tuple[float, list, float]:
    """
    Quick test to verify rewards and pipeline.
    
    Returns:
        Tuple of (total_reward, rewards_per_step, delivery_rate)
    """
    from src.env.core import create_curriculum_stages, SystemParameters
    from src.env.environment import EnergyHarvestingCameraEnv
    
    logger.info("=" * 80)
    logger.info("QUICK DIAGNOSTIC TEST")
    logger.info("=" * 80)
    
    # Use tutorial stage parameters
    stages = create_curriculum_stages()
    tutorial_stage = stages[0]
    
    params = tutorial_stage.apply_to_params(SystemParameters())
    
    env = EnergyHarvestingCameraEnv(
        "NREL.csv",
        params,
        event_timeout=tutorial_stage.event_timeout
    )
    
    logger.info(f"\nTutorial Parameters:")
    logger.info(f"  Battery SOC: {params.initial_battery_soc:.0%}")
    logger.info(f"  Event Prob: {params.prob_event_per_step:.1%}")
    logger.info(f"  Images/Event: {params.images_per_event_mean:.0f}")
    logger.info(f"  TX Capacity: {params.n_tx_img_max}")
    logger.info(f"  Event Timeout: {tutorial_stage.event_timeout}")
    
    # Test aggressive policy
    obs, _ = env.reset(seed=42)
    
    total_reward = 0.0
    rewards_per_step = []
    
    for step in range(200):
        # Aggressive: always active, high capture, complex process, always transmit
        action_tuple = (0, 2, 2, 1)
        action = env.multidiscrete_to_action[action_tuple]
        
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        rewards_per_step.append(reward)
        
        if step % 50 == 0:
            logger.info(f"\nStep {step}:")
            logger.info(f"  Reward: {reward:.2f}")
            logger.info(f"  Cumulative: {total_reward:.2f}")
            logger.info(f"  Delivered Events: {info['event_stats']['important_delivered']}")
            logger.info(f"  Missed Events: {info['event_stats']['important_missed']}")
            logger.info(f"  Battery: {info['battery_soc']:.1%}")
            logger.info(f"  Transmitted: {info['n_transmitted']}")
        
        if terminated or truncated:
            break
    
    logger.info("\n" + "=" * 80)
    logger.info("DIAGNOSTIC RESULTS")
    logger.info("=" * 80)
    logger.info(f"Total Reward: {total_reward:.2f}")
    logger.info(f"Mean Reward per Step: {np.mean(rewards_per_step):.2f}")
    logger.info(f"Positive Steps: {sum(1 for r in rewards_per_step if r > 0)}/{len(rewards_per_step)}")
    logger.info(f"Final Delivery Rate: {info['event_stats']['delivery_rate']:.1%}")
    logger.info(f"Total Events: {info['event_stats']['total_important_events']}")
    
    if info['event_stats']['delivery_rate'] >= 0.7:
        logger.info("\n✅ DELIVERY RATE EXCELLENT! Tutorial should work.")
    elif info['event_stats']['delivery_rate'] >= 0.5:
        logger.info("\n⚠️  DELIVERY RATE MODERATE. Tutorial might struggle.")
    else:
        logger.warning("\n❌ DELIVERY RATE TOO LOW! Agent won't learn effectively.")
        logger.warning("   Adjust pipeline parameters further.")
    
    if total_reward > 0 and np.mean(rewards_per_step) > 0:
        logger.info("✅ REWARDS WORKING! Agent should learn from this signal.")
    else:
        logger.warning("⚠️  REWARDS TOO LOW! Agent won't learn.")
    
    env.close()
    
    return total_reward, rewards_per_step, info['event_stats']['delivery_rate']

# ============================================================================
# PRE-FLIGHT CHECKS
# ============================================================================

def run_preflight_checks() -> bool:
    """
    Run all pre-flight checks before training.
    
    Returns:
        True if all checks pass
    """
    from src.env.core import create_curriculum_stages, SystemParameters
    from src.env.environment import EnergyHarvestingCameraEnv
    
    print("\n" + "=" * 80)
    print("PRE-FLIGHT CHECKS")
    print("=" * 80)
    
    all_passed = True
    
    # Test 1: Parameter verification
    print("\n🔍 TEST 1: Verifying Curriculum Parameters...")
    try:
        test_curriculum_params()
        print("✅ PASS")
    except Exception as e:
        print(f"❌ FAIL: {e}")
        all_passed = False
    
    # Test 2: Quick diagnostic
    print("\n🔍 TEST 2: Running Quick Diagnostic...")
    try:
        total_reward, rewards, delivery_rate = quick_diagnostic_test()
        if delivery_rate >= 0.7 and total_reward > 0:
            print("✅ PASS")
        else:
            print(f"⚠️  MARGINAL: Delivery={delivery_rate:.1%}, Reward={total_reward:.2f}")
            if delivery_rate < 0.5:
                all_passed = False
    except Exception as e:
        print(f"❌ FAIL: {e}")
        all_passed = False
    
    # Test 3: Pipeline check
    print("\n🔍 TEST 3: Checking Pipeline Bottleneck...")
    try:
        stages = create_curriculum_stages()
        tutorial_stage = stages[0]
        tutorial_params = tutorial_stage.apply_to_params(SystemParameters())
        tutorial_env = EnergyHarvestingCameraEnv(
            "NREL.csv",
            tutorial_params,
            event_timeout=tutorial_stage.event_timeout
        )
        pipeline_stats = diagnose_pipeline_bottleneck(tutorial_env, n_steps=200)
        tutorial_env.close()
        
        if pipeline_stats['images_transmitted'] > 100:
            print("✅ PASS")
        else:
            print(f"⚠️  MARGINAL: Only {pipeline_stats['images_transmitted']} images transmitted")
    except Exception as e:
        print(f"❌ FAIL: {e}")
        all_passed = False
    
    # Final summary
    print("\n" + "=" * 80)
    print("PRE-FLIGHT CHECK SUMMARY")
    print("=" * 80)
    
    if all_passed:
        print("\n🚀 ALL CHECKS PASSED! Ready to start training.")
    else:
        print("\n⚠️  SOME CHECKS FAILED! Review parameters before training.")
    
    print("=" * 80)
    
    return all_passed

# ============================================================================
# STAGE COMPARISON
# ============================================================================

def compare_curriculum_stages(history: Dict) -> List[Dict]:
    """
    Compare performance across curriculum stages.
    
    Args:
        history: Training history
    
    Returns:
        List of stage statistics
    """
    stages = []
    current_stage = None
    stage_data = {'rewards': [], 'deliveries': [], 'efficiency': []}
    
    for i, stage_name in enumerate(history['stage_names']):
        if stage_name != current_stage:
            if current_stage is not None:
                stages.append({
                    'name': current_stage,
                    'mean_reward': np.mean(stage_data['rewards']),
                    'mean_delivery': np.mean(stage_data['deliveries']),
                    'mean_efficiency': np.mean(stage_data['efficiency']),
                    'final_delivery': (
                        np.mean(stage_data['deliveries'][-100:]) 
                        if len(stage_data['deliveries']) >= 100 
                        else np.mean(stage_data['deliveries'])
                    )
                })
            current_stage = stage_name
            stage_data = {'rewards': [], 'deliveries': [], 'efficiency': []}
        
        stage_data['rewards'].append(history['episode_rewards'][i])
        stage_data['deliveries'].append(history['delivery_rates'][i])
        stage_data['efficiency'].append(history['energy_efficiency'][i])
    
    # Add last stage
    if current_stage is not None:
        stages.append({
            'name': current_stage,
            'mean_reward': np.mean(stage_data['rewards']),
            'mean_delivery': np.mean(stage_data['deliveries']),
            'mean_efficiency': np.mean(stage_data['efficiency']),
            'final_delivery': (
                np.mean(stage_data['deliveries'][-100:]) 
                if len(stage_data['deliveries']) >= 100 
                else np.mean(stage_data['deliveries'])
            )
        })
    
    # Print comparison
    print("\n" + "=" * 80)
    print("CURRICULUM STAGE COMPARISON")
    print("=" * 80)
    print(f"{'Stage':<15} {'Mean Reward':>15} {'Delivery Rate':>15} {'Final Delivery':>15} {'Efficiency':>15}")
    print("-" * 80)
    
    for stage in stages:
        print(
            f"{stage['name']:<15} "
            f"{stage['mean_reward']:>15.2f} "
            f"{stage['mean_delivery']:>14.1%} "
            f"{stage['final_delivery']:>14.1%} "
            f"{stage['mean_efficiency']:>15.2f}"
        )
    
    print("=" * 80)
    
    return stages

logger.info("✅ Part 6: Testing and Diagnostics loaded successfully")
