"""
Baseline Policies for Energy Harvesting Camera System
Professional implementations for rigorous PPO evaluation

These baselines should establish the performance ordering:
    Optimal Heuristic > PPO > Smart Heuristic > Random > Always Sleep

Author: Professional RL Evaluation
Date: 2026-02-07
"""

import numpy as np
from typing import Tuple, Dict, Any
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# BASE POLICY CLASS
# ============================================================================

class BaselinePolicy(ABC):
    """Abstract base class for all baseline policies."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.total_actions = 0
        self.action_counts = {}
    
    @abstractmethod
    def select_action(
        self, 
        observation: np.ndarray, 
        info: Dict[str, Any]
    ) -> Tuple[int, int, int, int]:
        """
        Select action given observation.
        
        Args:
            observation: Environment observation
            info: Additional information from environment
        
        Returns:
            Tuple of (sleep_mode, capture_mode, process_mode, tx_mode)
        """
        pass
    
    def reset(self) -> None:
        """Reset policy statistics."""
        self.total_actions = 0
        self.action_counts = {}
    
    def record_action(self, action: Tuple[int, int, int, int]) -> None:
        """Record action for statistics."""
        if action not in self.action_counts:
            self.action_counts[action] = 0
        self.action_counts[action] += 1
        self.total_actions += 1

# ============================================================================
# BASELINE 1: RANDOM POLICY (Lower Bound)
# ============================================================================

class RandomPolicy(BaselinePolicy):
    """
    Uniformly random action selection.
    
    Expected Performance: WORST
    Purpose: Establish lower bound - PPO MUST beat this
    """
    
    def __init__(self):
        super().__init__(
            name="Random",
            description="Uniformly random actions (lower bound)"
        )
    
    def select_action(
        self, 
        observation: np.ndarray, 
        info: Dict[str, Any]
    ) -> Tuple[int, int, int, int]:
        """Random action from valid space."""
        sleep = np.random.randint(0, 3)
        capture = np.random.randint(0, 3)
        process = np.random.randint(0, 3)
        tx = np.random.randint(0, 2)
        
        action = (sleep, capture, process, tx)
        self.record_action(action)
        return action

# ============================================================================
# BASELINE 2: ALWAYS SLEEP (Sanity Check)
# ============================================================================

class AlwaysSleepPolicy(BaselinePolicy):
    """
    Always deep sleep, never do anything.
    
    Expected Performance: WORST (tied with random or worse)
    Purpose: Sanity check - should get 0% delivery
    """
    
    def __init__(self):
        super().__init__(
            name="Always Sleep",
            description="Deep sleep only (sanity check)"
        )
        self.action = (2, 0, 0, 0)  # Halt, no capture, no process, no tx
    
    def select_action(
        self, 
        observation: np.ndarray, 
        info: Dict[str, Any]
    ) -> Tuple[int, int, int, int]:
        """Always return deep sleep."""
        self.record_action(self.action)
        return self.action

# ============================================================================
# BASELINE 3: ALWAYS MAX THROUGHPUT (Energy-Blind)
# ============================================================================

class AlwaysMaxThroughputPolicy(BaselinePolicy):
    """
    Always capture high, process complex, transmit.
    Ignores battery completely.
    
    Expected Performance: HIGH delivery BUT many failures
    Purpose: Test if energy constraint is real
    """
    
    def __init__(self):
        super().__init__(
            name="Max Throughput",
            description="Always maximum effort (energy-blind)"
        )
        self.action = (0, 2, 2, 1)  # Active, high capture, complex, tx
    
    def select_action(
        self, 
        observation: np.ndarray, 
        info: Dict[str, Any]
    ) -> Tuple[int, int, int, int]:
        """Always return max throughput."""
        self.record_action(self.action)
        return self.action

# ============================================================================
# BASELINE 4: BATTERY-AWARE THRESHOLD POLICY
# ============================================================================

class BatteryThresholdPolicy(BaselinePolicy):
    """
    Simple threshold-based policy using battery SOC.
    
    Expected Performance: DECENT (should beat random)
    Purpose: Establish that battery-awareness helps
    """
    
    def __init__(self):
        super().__init__(
            name="Battery Threshold",
            description="Simple battery-aware thresholds"
        )
    
    def select_action(
        self, 
        observation: np.ndarray, 
        info: Dict[str, Any]
    ) -> Tuple[int, int, int, int]:
        """Select action based on battery thresholds."""
        # Extract battery SOC (first element of observation)
        battery_soc = observation[0]
        
        # Determine sleep mode
        if battery_soc < 0.15:
            sleep = 2  # Deep sleep
        elif battery_soc < 0.30:
            sleep = 1  # Idle
        else:
            sleep = 0  # Active
        
        # Determine capture mode
        if battery_soc < 0.20:
            capture = 0  # Off
        elif battery_soc > 0.60:
            capture = 2  # High rate
        else:
            capture = 1  # Low rate
        
        # Determine processing mode
        if battery_soc < 0.25:
            process = 0  # Off
        elif battery_soc > 0.65:
            process = 2  # Complex
        else:
            process = 1  # Simple
        
        # Determine transmission
        if battery_soc < 0.20:
            tx = 0  # Off
        else:
            tx = 1  # On
        
        action = (sleep, capture, process, tx)
        self.record_action(action)
        return action

# ============================================================================
# BASELINE 5: SMART HEURISTIC (Buffer + Battery)
# ============================================================================

class SmartHeuristicPolicy(BaselinePolicy):
    """
    Hand-crafted policy using battery + buffer state.
    
    Expected Performance: GOOD (competitive with early PPO)
    Purpose: Establish that task is learnable with domain knowledge
    """
    
    def __init__(self):
        super().__init__(
            name="Smart Heuristic",
            description="Buffer + battery heuristic"
        )
    
    def select_action(
        self, 
        observation: np.ndarray, 
        info: Dict[str, Any]
    ) -> Tuple[int, int, int, int]:
        """Select action using buffer + battery state."""
        # Extract features
        battery_soc = observation[0]
        buffer_raw_pct = observation[1]
        buffer_proc_pct = observation[2]
        
        # Extract GHI forecast (last 24 elements)
        ghi_forecast = observation[-24:]
        avg_ghi = np.mean(ghi_forecast[:12])  # Next 12 steps
        
        # Adaptive thresholds based on solar forecast
        if avg_ghi > 500:  # Good solar
            battery_low = 0.25
            battery_high = 0.55
        else:  # Poor solar
            battery_low = 0.40
            battery_high = 0.70
        
        # Sleep mode
        if battery_soc < 0.10:
            sleep = 2  # Emergency deep sleep
        elif battery_soc < battery_low:
            sleep = 1  # Idle
        else:
            sleep = 0  # Active
        
        # Capture mode (avoid overflow)
        if battery_soc < battery_low or buffer_raw_pct > 0.85:
            capture = 0
        elif buffer_raw_pct > 0.70:
            capture = 1  # Slow down
        elif battery_soc > battery_high and avg_ghi > 400:
            capture = 2  # Opportunistic
        else:
            capture = 1
        
        # Processing mode (clear backlog when needed)
        if battery_soc < battery_low or buffer_raw_pct < 0.10:
            process = 0
        elif buffer_raw_pct > 0.60:  # Backlog clearing
            process = 2 if battery_soc > 0.50 else 1
        elif battery_soc > battery_high:
            process = 2
        else:
            process = 1
        
        # Transmission (prioritize when buffer filling)
        if battery_soc < 0.20:
            tx = 0
        elif buffer_proc_pct > 0.40:  # Buffer getting full
            tx = 1
        elif buffer_proc_pct > 0.15 and battery_soc > 0.35:
            tx = 1
        else:
            tx = 0
        
        action = (sleep, capture, process, tx)
        self.record_action(action)
        return action

# ============================================================================
# BASELINE 6: ORACLE POLICY (Theoretical Upper Bound)
# ============================================================================

class OraclePolicy(BaselinePolicy):
    """
    Cheating policy with perfect future knowledge.
    Uses true GHI forecast and event schedule.
    
    Expected Performance: BEST POSSIBLE
    Purpose: Establish theoretical upper bound
    
    NOTE: This is NOT fair comparison - it cheats!
    Used only to estimate problem difficulty.
    """
    
    def __init__(self, env):
        super().__init__(
            name="Oracle (Cheating)",
            description="Perfect future knowledge (upper bound)"
        )
        self.env = env
    
    def select_action(
        self, 
        observation: np.ndarray, 
        info: Dict[str, Any]
    ) -> Tuple[int, int, int, int]:
        """Select action with perfect foresight."""
        battery_soc = observation[0]
        buffer_raw_pct = observation[1]
        buffer_proc_pct = observation[2]
        
        # CHEAT: Look ahead at next 100 steps of GHI
        future_energy = 0.0
        for i in range(100):
            future_timestep = self.env.timestep + i
            future_ghi = self.env.GHI_loader.get_GHI(future_timestep)
            future_energy += self.env.energy_model.compute_harvested_energy(
                future_ghi
            )
        
        # CHEAT: Check if event is active
        event_active = self.env.current_event_id is not None
        
        # Smart decisions with future knowledge
        if future_energy > 5000:  # Lots of energy coming
            # Aggressive
            sleep = 0
            capture = 2
            process = 2
            tx = 1
        elif future_energy > 2000:  # Some energy
            # Moderate
            sleep = 0
            capture = 1
            process = 1
            tx = 1 if buffer_proc_pct > 0.2 else 0
        else:  # Low energy ahead
            # Conservative
            sleep = 1
            capture = 1 if event_active else 0
            process = 1 if buffer_raw_pct > 0.3 else 0
            tx = 1 if buffer_proc_pct > 0.5 else 0
        
        # Emergency overrides
        if battery_soc < 0.15:
            sleep = 2
            capture = 0
            process = 0
            tx = 0
        
        action = (sleep, capture, process, tx)
        self.record_action(action)
        return action

# ============================================================================
# BASELINE FACTORY
# ============================================================================

def create_all_baselines(env=None) -> Dict[str, BaselinePolicy]:
    """
    Create all baseline policies for comparison.
    
    Args:
        env: Environment (needed for Oracle policy)
    
    Returns:
        Dictionary mapping policy names to policy objects
    """
    baselines = {
        'random': RandomPolicy(),
        'always_sleep': AlwaysSleepPolicy(),
        'max_throughput': AlwaysMaxThroughputPolicy(),
        'battery_threshold': BatteryThresholdPolicy(),
        'smart_heuristic': SmartHeuristicPolicy(),
    }
    
    # Add oracle only if environment provided
    if env is not None:
        baselines['oracle'] = OraclePolicy(env)
    
    return baselines


def get_fair_baselines() -> Dict[str, BaselinePolicy]:
    """
    Get only fair baselines (no cheating).
    
    Returns:
        Dictionary of fair baseline policies
    """
    return {
        'random': RandomPolicy(),
        'battery_threshold': BatteryThresholdPolicy(),
        'smart_heuristic': SmartHeuristicPolicy(),
    }

logger.info("✅ Baseline policies loaded")
logger.info("   Available: Random, Always Sleep, Max Throughput, Battery Threshold, Smart Heuristic, Oracle")
