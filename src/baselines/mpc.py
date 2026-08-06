"""
Model Predictive Control (MPC) Baseline - COMPATIBLE WITH ACTUAL SYSTEMPARAMETERS
Fixed to work with the exact parameter names from Part1_Core.py and Part2_Energy.py

Key fixes:
- Uses correct parameter names (e_proc_simple, e_tx_img_total, etc.)
- Computes capture energy from measured components
- Compatible with Jetson-specific parameters
- Returns tuple format for evaluation compatibility

Author: MPC Compatible with Actual SystemParameters
Date: 2026-02-08
"""

import numpy as np
from typing import Tuple, List, Dict, Any, Optional
from dataclasses import dataclass
import logging
import time

logger = logging.getLogger(__name__)


@dataclass
class MPCConfig:
    """Configuration for Model Predictive Control."""
    
    # Prediction horizon
    horizon: int = 12
    
    # Forecasting
    forecast_method: str = "persistence"
    
    # Simplified action space
    use_simplified_actions: bool = True
    
    # Sampling strategy
    sampling_strategy: str = "smart"  # "grid", "random", "smart"
    n_random_samples: int = 100
    
    # Cost function weights
    w_delivery: float = 100.0
    w_energy: float = 10.0
    w_safety: float = 1000.0
    w_smoothness: float = 1.0
    
    # Safety constraints
    min_battery_threshold: float = 0.15
    safety_margin: float = 0.05
    
    # Computational limits
    max_computation_time: float = 0.1
    
    def validate(self):
        """Validate configuration."""
        assert self.horizon > 0
        assert 0 < self.min_battery_threshold < 1
        assert self.forecast_method in ["perfect", "persistence", "moving_average"]
        logger.info(f"MPC config validated: horizon={self.horizon}, forecast={self.forecast_method}")


class SimplifiedActionSpace:
    """Simplified action space for computational efficiency."""
    
    def __init__(self):
        self.action_presets = {
            0: (2, 0, 0, 0),  # Deep sleep
            1: (1, 0, 0, 0),  # Idle
            2: (0, 1, 1, 0),  # Low capture + Simple
            3: (0, 1, 1, 1),  # Low capture + Simple + TX
            4: (0, 2, 1, 0),  # High capture + Simple
            5: (0, 2, 1, 1),  # High capture + Simple + TX
            6: (0, 1, 2, 1),  # Low capture + Complex + TX
            7: (0, 2, 2, 1),  # Max throughput
        }
        self.n_actions = len(self.action_presets)
    
    def get_action_tuple(self, idx: int) -> Tuple[int, int, int, int]:
        return self.action_presets[idx]


class MPCEnergyModel:
    """
    Energy model compatible with actual SystemParameters.
    
    Uses the exact parameter names and calculations from Part1_Core.py
    """
    
    def __init__(self, params):
        self.params = params
    
    def predict_energy_harvested(self, ghi: float) -> float:
        """Predict energy harvested from solar."""
        return self.params.eta_solar * self.params.A_panel * ghi * self.params.dt
    
    def predict_capture_energy(self, capture_mode: int) -> float:
        """
        Predict capture energy using actual Jetson measurements.
        
        Based on Part1_Core.py parameters:
        - E_init_cap: Initialization energy
        - For low (mode 1): n_low images with specific per-image cost
        - For high (mode 2): n_high images with different costs
        """
        if capture_mode == 0:
            return 0.0
        elif capture_mode == 1:
            # Low quality: 6 images
            return (self.params.E_init_cap + 
                   self.params.n_low * self.params.low_active_per_image +
                   self.params.low_logging_fixed_round)
        else:  # capture_mode == 2
            # High quality: 120 images
            return (self.params.E_init_cap +
                   self.params.n_high * self.params.high_capture_per_image +
                   self.params.high_startup_fixed_round +
                   self.params.high_standby_power * self.params.high_inter_capture_delay +
                   self.params.high_logging_fixed_round)
    
    def predict_process_energy(self, process_mode: int, buffer_raw: int) -> float:
        """
        Predict processing energy using actual Jetson measurements.
        
        Based on Part1_Core.py:
        - E_proc_init_simple/complex: Initialization
        - e_proc_simple/complex: Per-image processing
        - n_proc_simple_max/complex_max: Maximum batch size
        """
        if process_mode == 0 or buffer_raw == 0:
            return 0.0
        elif process_mode == 1:
            # Simple processing
            n = min(buffer_raw, self.params.n_proc_simple_max)
            return (self.params.E_proc_init_simple +
                   n * self.params.e_proc_simple +
                   self.params.E_simple_log)
        else:  # process_mode == 2
            # Complex processing
            n = min(buffer_raw, self.params.n_proc_complex_max)
            return (self.params.E_proc_init_complex +
                   n * self.params.e_proc_complex +
                   self.params.E_complex_log)
    
    def predict_tx_energy(self, tx_mode: int, buffer_proc: int) -> float:
        """
        Predict transmission energy using actual Jetson measurements.
        
        Based on Part1_Core.py:
        - E_init_tx: WiFi initialization
        - e_tx_img_total: Per-image transmission
        - n_tx_img_max: Maximum images per batch
        - P_wifi: WiFi power during transmission
        - E_wifi_off: WiFi shutdown
        """
        if tx_mode == 0:
            return 0.0
        
        # Number of images to transmit
        n_imgs = min(self.params.n_tx_img_max, buffer_proc)
        
        if n_imgs == 0:
            return 0.0
        
        return (self.params.E_init_tx +
               n_imgs * self.params.e_tx_img_total +
               self.params.P_wifi * self.params.dt +
               self.params.e_tx_log_total +
               self.params.E_wifi_off)
    
    def predict_standby_energy(self, sleep_mode: int) -> float:
        """
        Predict standby energy based on sleep mode.
        
        Based on Part1_Core.py:
        - P_active, P_idle, P_sleep for different modes
        """
        if sleep_mode == 2:
            # Deep sleep
            return self.params.P_sleep * self.params.dt
        elif sleep_mode == 1:
            # Light sleep / idle
            return self.params.P_idle * self.params.dt
        else:  # sleep_mode == 0
            # Active
            return self.params.P_active * self.params.dt
    
    def predict_total_energy(
        self, 
        action: Tuple[int, int, int, int],
        buffer_raw: int,
        buffer_proc: int
    ) -> float:
        """Predict total energy consumption for an action."""
        sleep, capture, process, tx = action
        
        E_cap = self.predict_capture_energy(capture)
        E_proc = self.predict_process_energy(process, buffer_raw)
        E_tx = self.predict_tx_energy(tx, buffer_proc)
        E_standby = self.predict_standby_energy(sleep)
        
        return E_cap + E_proc + E_tx + E_standby


class MPCStatePredictor:
    """Predicts state evolution for MPC planning."""
    
    def __init__(self, params):
        self.params = params
        self.energy_model = MPCEnergyModel(params)
    
    def predict_next_state(
        self,
        current_state: Dict[str, float],
        action: Tuple[int, int, int, int],
        ghi_forecast: float
    ) -> Dict[str, float]:
        """Predict next state given current state, action, and GHI."""
        
        sleep, capture, process, tx = action
        
        # Energy dynamics
        E_harvested = self.energy_model.predict_energy_harvested(ghi_forecast)
        E_consumed = self.energy_model.predict_total_energy(
            action, 
            int(current_state['buffer_raw']),
            int(current_state['buffer_proc'])
        )
        
        next_battery = current_state['battery_energy'] + E_harvested - E_consumed
        next_battery = max(0.0, min(next_battery, self.params.B_max))
        
        # Buffer dynamics
        # Capture
        if capture == 1:
            n_captured = self.params.n_low
        elif capture == 2:
            n_captured = self.params.n_high
        else:
            n_captured = 0
        
        next_buffer_raw = current_state['buffer_raw'] + n_captured
        
        # Process
        if process > 0 and next_buffer_raw > 0:
            if process == 1:
                n_processed = min(next_buffer_raw, self.params.n_proc_simple_max)
            else:
                n_processed = min(next_buffer_raw, self.params.n_proc_complex_max)
            
            next_buffer_raw -= n_processed
            next_buffer_proc = current_state['buffer_proc'] + n_processed
        else:
            next_buffer_proc = current_state['buffer_proc']
        
        # Transmit
        if tx > 0 and next_buffer_proc > 0:
            n_transmitted = min(self.params.n_tx_img_max, next_buffer_proc)
            next_buffer_proc -= n_transmitted
        
        # Clip to buffer limits
        next_buffer_raw = min(next_buffer_raw, self.params.Q_max_raw)
        next_buffer_proc = min(next_buffer_proc, self.params.Q_max_proc)
        
        # Event dynamics (simplified)
        next_event_active = np.random.random() < self.params.prob_event_per_step
        
        return {
            'battery_energy': next_battery,
            'buffer_raw': next_buffer_raw,
            'buffer_proc': next_buffer_proc,
            'event_active': next_event_active
        }


class MPCCostFunction:
    """Cost function for MPC optimization."""
    
    def __init__(self, config: MPCConfig, params):
        self.config = config
        self.params = params
    
    def evaluate_trajectory(
        self,
        states: List[Dict[str, float]],
        actions: List[Tuple[int, int, int, int]]
    ) -> float:
        """Evaluate total cost of a state-action trajectory."""
        
        total_cost = 0.0
        
        for t in range(len(states) - 1):
            state = states[t]
            action = actions[t]
            next_state = states[t + 1]
            
            # 1. Delivery reward (negative cost)
            if action[3] == 1 and state['event_active']:
                total_cost -= self.config.w_delivery * \
                             min(state['buffer_proc'], self.params.n_tx_img_max)
            
            # 2. Energy preservation
            battery_soc = state['battery_energy'] / self.params.B_max
            if battery_soc < 0.3:
                total_cost += self.config.w_energy * (0.3 - battery_soc)
            
            # 3. Safety constraint
            if next_state['battery_energy'] < \
               (self.config.min_battery_threshold * self.params.B_max):
                total_cost += self.config.w_safety
            
            # 4. Buffer overflow penalty
            if state['buffer_raw'] > self.params.Q_max_raw * 0.8:
                total_cost += 10.0 * (state['buffer_raw'] - self.params.Q_max_raw * 0.8)
            
            if state['buffer_proc'] > self.params.Q_max_proc * 0.8:
                total_cost += 10.0 * (state['buffer_proc'] - self.params.Q_max_proc * 0.8)
            
            # 5. Smoothness
            if t > 0:
                prev_action = actions[t - 1]
                action_change = sum(abs(a - b) for a, b in zip(action, prev_action))
                total_cost += self.config.w_smoothness * action_change
        
        return total_cost


class ModelPredictiveController:
    """
    Model Predictive Control policy for energy harvesting camera.
    
    FIXED: Compatible with actual SystemParameters from Part1_Core.py
    """
    
    def __init__(self, params, config: Optional[MPCConfig] = None):
        """Initialize MPC controller."""
        
        self.params = params
        self.config = config if config is not None else MPCConfig()
        self.config.validate()
        
        # Components
        self.action_space = SimplifiedActionSpace()
        self.state_predictor = MPCStatePredictor(params)
        self.cost_function = MPCCostFunction(self.config, params)
        
        # For baseline compatibility
        self.name = f"MPC (h={self.config.horizon})"
        self.description = f"Model Predictive Control with {self.config.horizon}-step horizon"
        
        # Statistics
        self.action_counts = np.zeros(54, dtype=int)
        self.total_actions = 0
        self.computation_times = []
        
        logger.info(f"MPC initialized: {self.name}")
    
    def reset(self):
        """Reset policy statistics."""
        self.action_counts = np.zeros(54, dtype=int)
        self.total_actions = 0
        self.computation_times = []
    
    def _forecast_ghi(
        self,
        current_ghi: float,
        ghi_history: np.ndarray,
        horizon: int
    ) -> np.ndarray:
        """Forecast GHI for planning horizon."""
        
        if self.config.forecast_method == "perfect":
            return np.ones(horizon) * current_ghi
        
        elif self.config.forecast_method == "persistence":
            return np.ones(horizon) * current_ghi
        
        elif self.config.forecast_method == "moving_average":
            if len(ghi_history) < 10:
                return np.ones(horizon) * current_ghi
            
            trend = np.mean(np.diff(ghi_history[-10:]))
            forecast = np.zeros(horizon)
            for t in range(horizon):
                forecast[t] = max(0, current_ghi + trend * (t + 1))
            return forecast
        
        else:
            return np.ones(horizon) * current_ghi
    
    def _extract_state_from_observation(self, observation: np.ndarray) -> Dict[str, float]:
        """Extract MPC state from environment observation."""
        
        battery_soc = observation[0]
        buffer_raw_pct = observation[1] if len(observation) > 1 else 0.5
        buffer_proc_pct = observation[2] if len(observation) > 2 else 0.5
        
        return {
            'battery_energy': battery_soc * self.params.B_max,
            'buffer_raw': buffer_raw_pct * self.params.Q_max_raw,
            'buffer_proc': buffer_proc_pct * self.params.Q_max_proc,
            'event_active': np.random.random() < self.params.prob_event_per_step
        }
    
    def _simulate_trajectory(
        self,
        initial_state: Dict[str, float],
        action_sequence: List[Tuple[int, int, int, int]],
        ghi_forecast: np.ndarray
    ) -> Tuple[List[Dict[str, float]], float]:
        """Simulate a trajectory under the simplified model."""
        
        states = [initial_state]
        current_state = initial_state.copy()
        
        for t, action in enumerate(action_sequence):
            next_state = self.state_predictor.predict_next_state(
                current_state,
                action,
                ghi_forecast[t]
            )
            states.append(next_state)
            current_state = next_state
        
        # Evaluate cost
        cost = self.cost_function.evaluate_trajectory(states, action_sequence)
        
        return states, cost
    
    def _optimize_smart_sampling(
        self,
        initial_state: Dict[str, float],
        ghi_forecast: np.ndarray
    ) -> List[Tuple[int, int, int, int]]:
        """Smart sampling: combine heuristics with local search."""
        
        # Start with battery-aware heuristic
        battery_soc = initial_state['battery_energy'] / self.params.B_max
        
        if battery_soc > 0.6:
            base_action_idx = 7  # Aggressive
        elif battery_soc > 0.3:
            base_action_idx = 5  # Moderate
        else:
            base_action_idx = 1  # Conservative
        
        current_sequence = [
            self.action_space.get_action_tuple(base_action_idx)
            for _ in range(self.config.horizon)
        ]
        
        _, current_cost = self._simulate_trajectory(
            initial_state, current_sequence, ghi_forecast
        )
        
        # Local search
        n_iterations = min(100, self.config.n_random_samples)
        
        for _ in range(n_iterations):
            # Perturb random position
            perturbed_sequence = current_sequence.copy()
            perturb_pos = np.random.randint(self.config.horizon)
            perturb_action_idx = np.random.randint(self.action_space.n_actions)
            perturbed_sequence[perturb_pos] = \
                self.action_space.get_action_tuple(perturb_action_idx)
            
            # Evaluate
            _, perturbed_cost = self._simulate_trajectory(
                initial_state, perturbed_sequence, ghi_forecast
            )
            
            # Keep if better
            if perturbed_cost < current_cost:
                current_sequence = perturbed_sequence
                current_cost = perturbed_cost
        
        return current_sequence
    
    def select_action(
        self,
        observation: np.ndarray,
        info: Optional[Dict] = None
    ) -> Tuple[int, int, int, int]:
        """
        Select action using MPC.
        
        Returns action TUPLE (sleep, capture, process, tx) for compatibility.
        """
        
        start_time = time.time()
        
        # 1. Extract current state
        current_state = self._extract_state_from_observation(observation)
        
        # 2. Forecast GHI
        if info and 'GHI' in info:
            current_ghi = info['GHI']
        else:
            # Fallback: assume last observation element is GHI (normalized)
            current_ghi = observation[-1] * 1000 if len(observation) > 0 else 500
        
        # GHI history for forecasting
        ghi_history = observation[-24:] * 1000 if len(observation) >= 24 else np.array([current_ghi])
        
        ghi_forecast = self._forecast_ghi(
            current_ghi, ghi_history, self.config.horizon
        )
        
        # 3. Optimize action sequence
        action_sequence = self._optimize_smart_sampling(
            current_state, ghi_forecast
        )
        
        # 4. Take first action (receding horizon principle)
        first_action_tuple = action_sequence[0]
        
        # Record statistics
        sleep, capture, process, tx = first_action_tuple
        flat_action = sleep * 18 + capture * 6 + process * 2 + tx
        self.action_counts[flat_action] += 1
        self.total_actions += 1
        
        elapsed_time = time.time() - start_time
        self.computation_times.append(elapsed_time)
        
        if elapsed_time > self.config.max_computation_time:
            logger.warning(
                f"MPC computation time {elapsed_time:.3f}s exceeded limit "
                f"{self.config.max_computation_time}s"
            )
        
        # Return TUPLE for compatibility
        return first_action_tuple
    
    def get_action_distribution(self) -> np.ndarray:
        """Get normalized action distribution."""
        if self.total_actions == 0:
            return np.zeros(54)
        return self.action_counts / self.total_actions
    
    def get_stats(self) -> Dict[str, Any]:
        """Get MPC statistics."""
        return {
            'mean_computation_time': np.mean(self.computation_times) 
                                    if self.computation_times else 0,
            'max_computation_time': np.max(self.computation_times) 
                                   if self.computation_times else 0,
            'total_decisions': self.total_actions
        }


# Test if run directly
if __name__ == "__main__":
    print("Testing MPC with actual SystemParameters...")
    
    try:
        from src.env.core import SystemParameters
        from src.env.environment import EnergyHarvestingCameraEnv
        
        params = SystemParameters()
        env = EnergyHarvestingCameraEnv("NREL.csv", params)
        
        # Create MPC
        config = MPCConfig(horizon=6, sampling_strategy="smart")
        mpc = ModelPredictiveController(params, config)
        
        print(f"✓ Created: {mpc.name}")
        
        # Test a few steps
        obs, info = env.reset()
        
        for i in range(5):
            action_tuple = mpc.select_action(obs, info)
            print(f"Step {i+1}: action={action_tuple}")
            
            # Verify format
            assert isinstance(action_tuple, tuple), f"Expected tuple, got {type(action_tuple)}"
            assert len(action_tuple) == 4, f"Expected 4 elements, got {len(action_tuple)}"
            
            # Convert to flat for env
            sleep, capture, process, tx = action_tuple
            flat_action = sleep * 18 + capture * 6 + process * 2 + tx
            
            obs, reward, terminated, truncated, info = env.step(flat_action)
            
            if terminated or truncated:
                break
        
        print("\n✅ MPC test passed!")
        print(f"Stats: {mpc.get_stats()}")
        
        env.close()
    
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


logger.info("✅ MPC Baseline (Fixed for Actual SystemParameters) loaded successfully")
