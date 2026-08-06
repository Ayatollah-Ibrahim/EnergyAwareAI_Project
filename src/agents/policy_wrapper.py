"""
Policy Completion Wrapper
A wrapper that ensures policies produce complete, valid actions even when they fail

This wrapper can be applied to any policy (PPO, baseline, or custom) to provide:
- Graceful handling of policy failures
- Default/fallback action generation
- Action validation and completion
- Logging and debugging capabilities

Author: Completion Wrapper Implementation
Date: 2026-02-07
"""

import numpy as np
import logging
from typing import Tuple, Optional, Any, Dict
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CompletionConfig:
    """Configuration for completion wrapper."""
    
    # Default actions when policy fails
    default_sleep_mode: int = 1  # Light sleep
    default_capture_mode: int = 1  # Medium quality
    default_process_mode: int = 1  # Medium processing
    default_tx_mode: int = 0  # No transmission
    
    # Battery-based adaptive defaults
    use_battery_adaptive: bool = True
    low_battery_threshold: float = 0.35
    critical_battery_threshold: float = 0.25
    
    # Validation
    validate_actions: bool = True
    n_modes: int = 3
    n_capture: int = 3
    n_process: int = 3
    n_tx: int = 2
    
    # Logging
    log_failures: bool = True
    log_completions: bool = False
    
    def get_conservative_action(self, battery_level: float) -> Tuple[int, int, int, int]:
        """Get conservative action based on battery level."""
        if battery_level < self.critical_battery_threshold:
            # Critical battery: deep sleep, minimal activity
            return (2, 0, 0, 0)  # Deep sleep, no capture, no processing, no TX
        elif battery_level < self.low_battery_threshold:
            # Low battery: light sleep, reduced activity
            return (2, 1, 1, 1)  # Deep sleep, low capture, low processing, low tx
        else:
            # Normal battery: default actions
            return (
                self.default_sleep_mode,
                self.default_capture_mode,
                self.default_process_mode,
                self.default_tx_mode
            )


class PolicyCompletionWrapper:
    """
    Wrapper that ensures any policy produces valid, complete actions.
    
    This wrapper can be applied to:
    - PPO agents
    - Baseline policies
    - Heuristic policies
    - Custom policies
    
    Benefits:
    - Graceful degradation when policy fails
    - Action validation
    - Battery-aware fallback actions
    - Consistent interface across all policies
    """
    
    def __init__(
        self,
        policy: Any,
        config: Optional[CompletionConfig] = None,
        name_suffix: str = " (Wrapped)"
    ):
        """
        Initialize completion wrapper.
        
        Args:
            policy: Any policy object with a select_action method
            config: Completion configuration
            name_suffix: Suffix to add to wrapped policy name
        """
        self.policy = policy
        self.config = config or CompletionConfig()
        self.name_suffix = name_suffix
        
        # Statistics
        self.total_calls = 0
        self.failures = 0
        self.completions = 0
        self.validation_fixes = 0
        self.battery_overrides = 0
        
        # Get policy name
        if hasattr(policy, 'name'):
            self.name = policy.name + name_suffix
        elif hasattr(policy, '__class__'):
            self.name = policy.__class__.__name__ + name_suffix
        else:
            self.name = "Unknown Policy" + name_suffix
        
        logger.info(f"Created PolicyCompletionWrapper for: {self.name}")
    
    def select_action(
        self,
        obs: np.ndarray,
        info: Optional[Dict] = None,
        deterministic: bool = True
    ) -> Tuple[Tuple[int, int, int, int], Optional[float], Optional[float]]:
        """
        Select action with completion guarantee.
        
        Args:
            obs: Observation array
            info: Optional info dict (for baseline policies)
            deterministic: Whether to use deterministic action (for PPO)
        
        Returns:
            action_tuple: (sleep, capture, process, tx)
            log_prob: Log probability (if available, else None)
            value: State value (if available, else None)
        """
        self.total_calls += 1
        action_tuple = None
        log_prob = None
        value = None
        
        # Extract battery level for adaptive defaults
        battery_level = self._extract_battery_level(obs)
        
        # CHECK: If battery is low and battery-adaptive mode is enabled,
        # override policy and use conservative action immediately
        if self.config.use_battery_adaptive and battery_level < self.config.low_battery_threshold:
            self.completions += 1
            self.battery_overrides += 1
            action_tuple = self.config.get_conservative_action(battery_level)
            
            if self.config.log_completions:
                logger.info(
                    f"Battery override for {self.name}: {battery_level:.2f} < {self.config.low_battery_threshold:.2f}. "
                    f"Using conservative action: {action_tuple}"
                )
            
            return action_tuple, log_prob, value
        
        # Normal operation: try to use policy
        try:
            # Try to get action from wrapped policy
            if hasattr(self.policy, 'select_action'):
                # Detect policy type and call with appropriate interface
                if hasattr(self.policy, 'actor'):
                    # PPO agent - has actor network, use deterministic parameter
                    result = self.policy.select_action(obs, deterministic=deterministic)
                else:
                    # Baseline or DQN policy - try different interfaces
                    try:
                        # First try with info parameter (for baseline policies)
                        result = self.policy.select_action(obs, info)
                    except TypeError:
                        # Fallback: some policies don't need info parameter
                        try:
                            result = self.policy.select_action(obs, deterministic=deterministic)
                        except TypeError:
                            # Last fallback: just obs
                            result = self.policy.select_action(obs)
                
                # Handle different return formats
                if isinstance(result, tuple):
                    if len(result) == 3:
                        # PPO/DQN format: (action, log_prob, value)
                        action_tuple, log_prob, value = result
                    elif len(result) == 4:
                        # Direct action tuple
                        action_tuple = result
                    else:
                        raise ValueError(f"Unexpected return format: {len(result)} elements")
                else:
                    # Assume it's just the action tuple
                    action_tuple = result
            
            # Validate action
            if action_tuple is not None and self.config.validate_actions:
                action_tuple = self._validate_action(action_tuple)
        
        except Exception as e:
            # Policy failed - use fallback
            self.failures += 1
            if self.config.log_failures:
                logger.warning(
                    f"Policy {self.name} failed: {str(e)}. "
                    f"Using fallback action (battery: {battery_level:.2f})"
                )
            action_tuple = None
        
        # Complete action if needed
        if action_tuple is None:
            self.completions += 1
            if self.config.use_battery_adaptive:
                action_tuple = self.config.get_conservative_action(battery_level)
            else:
                action_tuple = (
                    self.config.default_sleep_mode,
                    self.config.default_capture_mode,
                    self.config.default_process_mode,
                    self.config.default_tx_mode
                )
            
            if self.config.log_completions:
                logger.debug(f"Completed action for {self.name}: {action_tuple}")
        
        return action_tuple, log_prob, value
    
    def _extract_battery_level(self, obs: np.ndarray) -> float:
        """Extract battery level from observation."""
        try:
            # Assuming battery level is at index 0 (normalized)
            # Adjust this based on your actual observation structure
            if len(obs) > 0:
                return float(obs[0])
            else:
                return 0.5  # Default to medium battery
        except:
            return 0.5
    
    def _validate_action(self, action: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        """
        Validate and potentially fix action components.
        
        Args:
            action: Action tuple (sleep, capture, process, tx)
        
        Returns:
            Valid action tuple
        """
        try:
            sleep, capture, process, tx = action
            
            # Validate ranges
            original_action = action
            sleep = np.clip(sleep, 0, self.config.n_modes - 1)
            capture = np.clip(capture, 0, self.config.n_capture - 1)
            process = np.clip(process, 0, self.config.n_process - 1)
            tx = np.clip(tx, 0, self.config.n_tx - 1)
            
            fixed_action = (int(sleep), int(capture), int(process), int(tx))
            
            # Log if we had to fix anything
            if fixed_action != original_action:
                self.validation_fixes += 1
                if self.config.log_failures:
                    logger.debug(
                        f"Validated action: {original_action} -> {fixed_action}"
                    )
            
            return fixed_action
        
        except Exception as e:
            logger.error(f"Action validation failed: {e}")
            # Return safe default
            return (
                self.config.default_sleep_mode,
                self.config.default_capture_mode,
                self.config.default_process_mode,
                self.config.default_tx_mode
            )
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get wrapper statistics."""
        return {
            'total_calls': self.total_calls,
            'failures': self.failures,
            'completions': self.completions,
            'battery_overrides': self.battery_overrides,
            'validation_fixes': self.validation_fixes,
            'failure_rate': self.failures / max(1, self.total_calls),
            'completion_rate': self.completions / max(1, self.total_calls),
            'battery_override_rate': self.battery_overrides / max(1, self.total_calls),
            'validation_fix_rate': self.validation_fixes / max(1, self.total_calls)
        }
    
    def print_statistics(self):
        """Print wrapper statistics."""
        stats = self.get_statistics()
        print(f"\n{'='*60}")
        print(f"PolicyCompletionWrapper Statistics: {self.name}")
        print(f"{'='*60}")
        print(f"Total Calls:        {stats['total_calls']}")
        print(f"Failures:           {stats['failures']} ({stats['failure_rate']:.2%})")
        print(f"Completions:        {stats['completions']} ({stats['completion_rate']:.2%})")
        print(f"  Battery Overrides: {stats['battery_overrides']} ({stats['battery_override_rate']:.2%})")
        print(f"Validation Fixes:   {stats['validation_fixes']} ({stats['validation_fix_rate']:.2%})")
        print(f"{'='*60}\n")
    
    def reset_statistics(self):
        """Reset wrapper statistics."""
        self.total_calls = 0
        self.failures = 0
        self.completions = 0
        self.validation_fixes = 0
        self.battery_overrides = 0
    
    # Pass through other methods to wrapped policy
    def __getattr__(self, name):
        """Forward attribute access to wrapped policy."""
        return getattr(self.policy, name)


def create_wrapped_policy(
    policy: Any,
    config: Optional[CompletionConfig] = None,
    **wrapper_kwargs
) -> PolicyCompletionWrapper:
    """
    Convenience function to create a wrapped policy.
    
    Args:
        policy: Policy to wrap
        config: Completion configuration
        **wrapper_kwargs: Additional arguments for wrapper
    
    Returns:
        Wrapped policy
    """
    return PolicyCompletionWrapper(policy, config=config, **wrapper_kwargs)


# Example usage
if __name__ == "__main__":
    # Example: Wrap a dummy policy
    class DummyPolicy:
        def __init__(self):
            self.name = "DummyPolicy"
        
        def select_action(self, obs, deterministic=True):
            # Sometimes fail
            if np.random.random() < 0.1:
                raise ValueError("Random failure!")
            return (1, 1, 1, 0), None, None
    
    # Create and wrap policy
    policy = DummyPolicy()
    wrapped_policy = create_wrapped_policy(policy)
    
    # Test it
    obs = np.random.random(44)
    for _ in range(100):
        action, _, _ = wrapped_policy.select_action(obs)
        assert action is not None
    
    # Print statistics
    wrapped_policy.print_statistics()
    
    print("✅ Policy wrapper test passed!")
