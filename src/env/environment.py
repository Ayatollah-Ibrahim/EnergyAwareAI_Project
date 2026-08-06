"""
PPO Training for Energy Harvesting Camera System - Jetson Nano
Part 3: Environment - PROFESSIONAL REWARD REDESIGN

🔥 CRITICAL FIXES:
- Reward clipping to [-10, +10]
- Energy penalty added (economic clarity)
- Activity reward REMOVED
- Smooth reward progression (no spikes)
- Expected value rewards (reduced variance)

Author: Professional RL Redesign
Date: 2026-02-07
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Dict, Optional, List, Any
from collections import deque
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# ENERGY HARVESTING CAMERA ENVIRONMENT - FIXED
# ============================================================================

class EnergyHarvestingCameraEnv(gym.Env):
    """
    Energy Harvesting Camera Control Environment.
    
    🔥 PROFESSIONAL RL REDESIGN APPLIED
    """
    
    metadata = {'render_modes': ['human', 'ansi']}
    
    def __init__(
        self,
        nrel_data_path: str,
        params: Optional = None,
        render_mode: Optional[str] = None,
        event_timeout: int = 100
    ):
        """Initialize environment with fixed reward structure."""
        super().__init__()
        
        if params is None:
            from src.env.core import SystemParameters
            params = SystemParameters()
        self.params = params
        self.params.validate()
        
        self.event_timeout = event_timeout
        self.render_mode = render_mode
        
        # Load GHI data
        from src.env.core import GHILoader
        self.GHI_loader = GHILoader(
            nrel_data_path, 
            self.params.use_linear_interpolation
        )
        
        # Initialize subsystems (assuming Part 2 unchanged)
        from src.env.energy import (
            EnergyModel, 
            EnhancedDualBufferManager, 
            BatteryManager
        )
        
        self.energy_model = EnergyModel(self.params)
        self.buffer = EnhancedDualBufferManager(
            self.params.Q_max_raw,
            self.params.Q_max_proc,
            self.params
        )
        self.battery = BatteryManager(
            self.params.B_max,
            self.params.B_min,
            self.params.lambda_leak,
            self.params.initial_battery_soc
        )
        
        # Action space
        self.action_space = spaces.Discrete(54)
        self._create_action_mapping()
        
        # Previous action tracking
        self.prev_action_components = (0, 0, 0, 0)
        
        # Temporal features
        if self.params.include_temporal_features:
            self.battery_soc_history: deque = deque(
                maxlen=self.params.ma_window_battery
            )
            self.buffer_raw_history: deque = deque(
                maxlen=self.params.ma_window_buffer
            )
            self.buffer_proc_history: deque = deque(
                maxlen=self.params.ma_window_buffer
            )
        
        # Observation space
        obs_dim = self._compute_observation_dim()
        self.observation_space = spaces.Box(
            low=0.0, 
            high=1.0, 
            shape=(obs_dim,), 
            dtype=np.float32
        )
        
        # Episode state
        self.timestep = 0
        self.episode_steps = 0
        self.GHI_window = np.zeros(self.params.window_size, dtype=np.float32)
        
        # Event tracking
        self.current_event_id = None
        self.event_duration = 0
        self.max_event_duration = 5
        
        # Safety tracking
        self.safety_overrides_soft = 0
        self.safety_overrides_hard = 0
        self.last_was_overridden = False
        
        # Statistics
        self.total_steps = 0
        self.episode_stats = {
            'total_reward': 0.0,
            'images_transmitted': 0,
            'energy_harvested': 0.0,
            'energy_consumed': 0.0,
            'soft_safety_triggers': 0,
            'hard_safety_triggers': 0,
            'important_events_delivered': 0,
            'important_events_missed': 0,
            'total_important_events': 0
        }
        
        # Reward tracking
        self._prev_delivered = 0
        self._prev_missed = 0
        self._episode_step = 0
        
        logger.info(
            f"Environment initialized (PROFESSIONAL RL REDESIGN)"
        )
        logger.info(f"  Event timeout: {self.event_timeout} steps")
        logger.info(f"  Event frequency: {self.params.prob_event_per_step:.3f}")
    
    def _compute_observation_dim(self) -> int:
        """Calculate total observation dimension."""
        dim = 5
        if self.params.include_previous_action:
            dim += 11
        if self.params.include_temporal_features:
            dim += 3
        if self.params.include_safety_flag:
            dim += 1
        dim += self.params.window_size
        return dim
    
    def _create_action_mapping(self) -> None:
        """Create bidirectional action mapping."""
        self.action_to_multidiscrete = []
        self.multidiscrete_to_action = {}
        
        idx = 0
        for sleep in range(3):
            for capture in range(3):
                for process in range(3):
                    for tx in range(2):
                        tpl = (sleep, capture, process, tx)
                        self.action_to_multidiscrete.append(tpl)
                        self.multidiscrete_to_action[tpl] = idx
                        idx += 1
    
    def _get_observation(self) -> np.ndarray:
        """Construct observation vector."""
        GHI_norm = np.clip(self.GHI_window / 1000.0, 0, 1)
        buffer_stats = self.buffer.get_stats()
        
        base_obs = np.array([
            self.battery.get_normalized_level(),
            np.clip(buffer_stats['occupancy_raw'] / self.params.Q_max_raw, 0, 1),
            np.clip(buffer_stats['occupancy_proc'] / self.params.Q_max_proc, 0, 1),
            np.clip(buffer_stats['simple'] / self.params.Q_max_proc, 0, 1),
            np.clip(buffer_stats['complex'] / self.params.Q_max_proc, 0, 1)
        ])
        
        obs_components = [base_obs]
        
        if self.params.include_previous_action:
            prev_action_one_hot = self._encode_action_one_hot(
                *self.prev_action_components
            )
            obs_components.append(prev_action_one_hot)
        
        if self.params.include_temporal_features:
            temporal_features = self._compute_temporal_features()
            obs_components.append(temporal_features)
        
        if self.params.include_safety_flag:
            safety_flag = np.array(
                [1.0 if self.last_was_overridden else 0.0], 
                dtype=np.float32
            )
            obs_components.append(safety_flag)
        
        obs_components.append(GHI_norm)
        obs = np.concatenate(obs_components).astype(np.float32)
        return obs
    
    def _compute_temporal_features(self) -> np.ndarray:
        """Compute moving average features."""
        if len(self.battery_soc_history) > 0:
            ma_battery = np.mean(self.battery_soc_history)
        else:
            ma_battery = self.battery.get_normalized_level()
        
        if len(self.buffer_raw_history) > 0:
            ma_buffer_raw = np.mean(self.buffer_raw_history)
        else:
            ma_buffer_raw = self.buffer.occupancy_raw / self.params.Q_max_raw
        
        if len(self.buffer_proc_history) > 0:
            ma_buffer_proc = np.mean(self.buffer_proc_history)
        else:
            ma_buffer_proc = self.buffer.occupancy_proc / self.params.Q_max_proc
        
        return np.array(
            [ma_battery, ma_buffer_raw, ma_buffer_proc], 
            dtype=np.float32
        )
    
    def _update_temporal_features(self) -> None:
        """Update temporal feature histories."""
        if self.params.include_temporal_features:
            self.battery_soc_history.append(
                self.battery.get_normalized_level()
            )
            self.buffer_raw_history.append(
                self.buffer.occupancy_raw / self.params.Q_max_raw
            )
            self.buffer_proc_history.append(
                self.buffer.occupancy_proc / self.params.Q_max_proc
            )
    
    def _encode_action_one_hot(
        self, 
        sleep: int, 
        capture: int, 
        process: int, 
        tx: int
    ) -> np.ndarray:
        """Encode action as one-hot vector."""
        one_hot = np.zeros(11, dtype=np.float32)
        one_hot[sleep] = 1.0
        one_hot[3 + capture] = 1.0
        one_hot[6 + process] = 1.0
        one_hot[9 + tx] = 1.0
        return one_hot
    
    def _apply_soft_safety_shield(
        self,
        sleep_mode: int,
        capture_mode: int,
        process_mode: int,
        tx_mode: int
    ) -> Tuple:
        """Apply soft safety intervention based on battery level."""
        battery_soc = self.battery.get_normalized_level()
        
        if battery_soc <= self.params.hard_safety_threshold:
            self.safety_overrides_hard += 1
            return 2, 0, 0, 0, True, 'hard'
        
        if battery_soc <= self.params.soft_safety_threshold:
            E_required = self._simulate_energy_consumption(
                sleep_mode, capture_mode, process_mode, tx_mode
            )
            
            if E_required > self.battery.charge:
                self.safety_overrides_soft += 1
                
                if capture_mode == 2:
                    capture_mode = 1
                elif capture_mode == 1:
                    capture_mode = 0
                
                if process_mode == 2:
                    process_mode = 1
                elif process_mode == 1:
                    process_mode = 0
                
                tx_mode = 0
                
                if sleep_mode == 0:
                    sleep_mode = 1
                
                return sleep_mode, capture_mode, process_mode, tx_mode, True, 'soft'
        
        return sleep_mode, capture_mode, process_mode, tx_mode, False, 'none'
    
    def _simulate_energy_consumption(
        self,
        sleep_mode: int,
        capture_mode: int,
        process_mode: int,
        tx_mode: int
    ) -> float:
        """Simulate total energy consumption for safety check."""
        prev_cap = self.energy_model.prev_capture_mode
        prev_proc = self.energy_model.prev_process_mode
        prev_tx = self.energy_model.prev_tx_mode
        prev_wifi = self.energy_model.wifi_active
        
        E_capture, _ = self.energy_model.compute_capture_energy(capture_mode)
        
        n_available_proc = self.buffer.occupancy_raw
        E_process, _ = self.energy_model.compute_processing_energy(
            process_mode, n_available_proc
        )
        
        n_to_transmit = min(
            self.params.n_tx_img_max, 
            self.buffer.occupancy_proc
        )
        E_transmit = self.energy_model.compute_transmission_energy(
            n_to_transmit, tx_mode
        )
        
        E_standby = self.energy_model.compute_standby_energy(sleep_mode)
        
        self.energy_model.prev_capture_mode = prev_cap
        self.energy_model.prev_process_mode = prev_proc
        self.energy_model.prev_tx_mode = prev_tx
        self.energy_model.wifi_active = prev_wifi
        
        return E_capture + E_process + E_transmit + E_standby
    
    def _compute_reward(
        self,
        n_simple_tx: int,
        n_complex_tx: int,
        tx_mode: int,
        energy_consumed: float,
        energy_harvested: float,
        n_dropped_raw: int,
        n_dropped_proc: int,
        transmitted_images: List,
        dropped_raw_images: List,
        dropped_proc_images: List
    ) -> Tuple[float, Dict[str, float]]:
        """
        🔥 PROFESSIONALLY REDESIGNED REWARD FUNCTION
        
        KEY FIXES:
        1. Reward clipping to [-10, +10]
        2. Energy penalty added (economic clarity)
        3. Expected value rewards (reduced variance)
        4. Activity bonus REMOVED
        5. Smooth progression (no spikes)
        """
        
        n_total_tx = n_simple_tx + n_complex_tx
        SOC = self.battery.get_normalized_level()
        
        # Get event statistics
        event_stats = self.buffer.get_event_delivery_stats()
        
        # Calculate DELTA since last step
        current_delivered = event_stats['important_delivered']
        current_missed = event_stats['important_missed']
        
        new_delivered = current_delivered - self._prev_delivered
        new_missed = current_missed - self._prev_missed
        
        self._prev_delivered = current_delivered
        self._prev_missed = current_missed
        
        # ====================================================================
        # 1. PRIMARY OBJECTIVE: Event Delivery (COMPRESSED MAGNITUDE)
        # ====================================================================
        R_delivery = new_delivered * self.params.r_event_delivered  # 6.0 per event
        
        # ====================================================================
        # 2. QUALITY BONUSES - Using EXPECTED VALUE (reduced variance)
        # ====================================================================
        quality_bonus = 0.0
        for image in transmitted_images:
            if image.is_important:
                # 🔥 FIX: Use expected value instead of sampling
                if image.processing_model == 1:
                    # Expected reward = accuracy * quality
                    quality_bonus += (
                        self.params.acc_simple *
                        self.params.phi_model_simple *
                        self.params.r_quality_bonus
                    )
                elif image.processing_model == 2:
                    quality_bonus += (
                        self.params.acc_complex *
                        self.params.phi_model_complex *
                        self.params.r_quality_bonus
                    )
        
        R_quality = quality_bonus
        
        # Redundancy bonus
        R_redundancy = 0.0
        if new_delivered > 0:
            R_redundancy = min(new_delivered, 1.0) * self.params.r_redundancy_bonus
        
        # ====================================================================
        # 3. PENALTIES
        # ====================================================================
        P_missed = new_missed * self.params.r_event_missed
        
        P_overflow_raw = n_dropped_raw * self.params.r_overflow_raw
        P_overflow_proc = n_dropped_proc * self.params.r_overflow_proc
        
        important_dropped_raw = sum(
            1 for img in dropped_raw_images if img.is_important
        )
        important_dropped_proc = sum(
            1 for img in dropped_proc_images if img.is_important
        )
        
        P_overflow_important = (
            important_dropped_raw * self.params.r_overflow_important +
            important_dropped_proc * self.params.r_overflow_important
        )
        
        P_overflow = P_overflow_raw + P_overflow_proc + P_overflow_important
        
        # Battery penalty (stronger now)
        if SOC < 0.2:
            P_battery = self.params.r_battery_low * (0.2 - SOC) / 0.2
        else:
            P_battery = 0.0
        
        # 🔥 NEW: Energy economics penalty
        P_energy = -self.params.r_energy_penalty_scale * energy_consumed / 1000.0
        
        # ====================================================================
        # 4. SHAPING BONUSES (Rebalanced)
        # ====================================================================
        R_shaping = 0.0
        
        # 🔥 FIX: Only reward meaningful throughput
        if tx_mode > 0 and n_total_tx > 0:
            R_shaping += self.params.r_transmission_bonus * min(n_total_tx, 3) / 3.0
        
        # Processing bonus
        if n_simple_tx > 0 or n_complex_tx > 0:
            total_proc = n_simple_tx + n_complex_tx
            R_shaping += self.params.r_processing_bonus * min(total_proc, 3) / 3.0
        
        # 🔥 ACTIVITY BONUS REMOVED - Was rewarding laziness!
        
        # ====================================================================
        # 5. TOTAL REWARD with HARD CLIPPING
        # ====================================================================
        reward_raw = (
            R_delivery +
            R_quality +
            R_redundancy +
            R_shaping +
            P_missed +
            P_overflow +
            P_battery +
            P_energy
        )
        
        # 🔥 CRITICAL: Clip to PPO-safe range
        reward = np.clip(
            reward_raw,
            self.params.reward_clip_min,
            self.params.reward_clip_max
        )
        
        # ====================================================================
        # 6. BREAKDOWN FOR LOGGING
        # ====================================================================
        breakdown = {
            'important_events_delivered': current_delivered,
            'important_events_missed': current_missed,
            'new_delivered': new_delivered,
            'new_missed': new_missed,
            'R_delivery_raw': R_delivery,
            'R_quality_raw': R_quality,
            'R_redundancy_raw': R_redundancy,
            'R_shaping_raw': R_shaping,
            'P_missed_raw': P_missed,
            'P_overflow_raw': P_overflow,
            'P_battery_raw': P_battery,
            'P_energy_raw': P_energy,
            'reward_before_clip': reward_raw,
            'reward_after_clip': reward,
            'was_clipped': abs(reward_raw) > self.params.reward_clip_max,
            'pct_delivery': (
                100 * R_delivery / max(abs(reward), 1e-6) 
                if reward != 0 else 0
            ),
            'pct_quality': (
                100 * (R_quality + R_redundancy) / max(abs(reward), 1e-6) 
                if reward != 0 else 0
            ),
            'pct_shaping': (
                100 * R_shaping / max(abs(reward), 1e-6) 
                if reward != 0 else 0
            ),
            'pct_penalties': (
                100 * (P_missed + P_overflow + P_battery + P_energy) / max(abs(reward), 1e-6) 
                if reward != 0 else 0
            ),
            'total_reward': reward,
            'n_simple_tx': n_simple_tx,
            'n_complex_tx': n_complex_tx,
            'n_total_tx': n_total_tx,
            'energy_consumed': energy_consumed,
            'energy_harvested': energy_harvested,
            'battery_soc': SOC,
        }
        
        return reward, breakdown
    
    # Rest of the methods (reset, step) remain the same as revised Part 3
    # ... (copying from revised implementation)
    
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Reset environment to initial state."""
        if seed is not None:
            np.random.seed(seed)
            self.action_space.seed(seed)
        
        if self.params.random_start_cycling:
            self.timestep = np.random.randint(0, self.GHI_loader.max_steps)
        else:
            self.timestep = 0
        
        self.episode_steps = 0
        
        self.battery.reset()
        self.buffer.reset()
        self.energy_model.reset()
        self._prev_delivered = 0
        self._prev_missed = 0
        
        self.current_event_id = None
        self.event_duration = 0
        
        self.GHI_window = np.zeros(self.params.window_size, dtype=np.float32)
        for i in range(self.params.window_size):
            lookback_idx = max(0, self.timestep - self.params.window_size + i)
            self.GHI_window[i] = self.GHI_loader.get_GHI(lookback_idx)
        
        self.prev_action_components = (2, 0, 0, 0)
        
        if self.params.include_temporal_features:
            self.battery_soc_history.clear()
            self.buffer_raw_history.clear()
            self.buffer_proc_history.clear()
            
            for _ in range(self.params.ma_window_battery):
                self.battery_soc_history.append(
                    self.battery.get_normalized_level()
                )
            for _ in range(self.params.ma_window_buffer):
                self.buffer_raw_history.append(0.0)
                self.buffer_proc_history.append(0.0)
        
        self.safety_overrides_soft = 0
        self.safety_overrides_hard = 0
        self.last_was_overridden = False
        
        self.total_steps = 0
        self.episode_stats = {
            'total_reward': 0.0,
            'images_transmitted': 0,
            'energy_harvested': 0.0,
            'energy_consumed': 0.0,
            'soft_safety_triggers': 0,
            'hard_safety_triggers': 0,
            'important_events_delivered': 0,
            'important_events_missed': 0,
            'total_important_events': 0
        }
        
        obs = self._get_observation()
        info = {
            'start_timestep': self.timestep, 
            'battery_level': self.battery.charge
        }
        
        return obs, info
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one environment step with fixed reward."""
        self.total_steps += 1
        self.episode_steps += 1
        
        sleep_mode, capture_mode, process_mode, tx_mode = (
            self.action_to_multidiscrete[action]
        )
        
        (
            sleep_mode, capture_mode, process_mode, tx_mode, 
            was_overridden, intervention_type
        ) = self._apply_soft_safety_shield(
            sleep_mode, capture_mode, process_mode, tx_mode
        )
        
        self.last_was_overridden = was_overridden
        
        if intervention_type == 'soft':
            self.episode_stats['soft_safety_triggers'] += 1
        elif intervention_type == 'hard':
            self.episode_stats['hard_safety_triggers'] += 1
        
        self.prev_action_components = (
            sleep_mode, capture_mode, process_mode, tx_mode
        )
        
        GHI = self.GHI_loader.get_GHI(self.timestep)
        E_harvested = self.energy_model.compute_harvested_energy(GHI)
        
        if self.current_event_id is None:
            self.current_event_id = self.buffer.generate_event(self.timestep)
            if self.current_event_id is not None:
                self.event_duration = 0
        
        if self.current_event_id is not None:
            self.event_duration += 1
            if self.event_duration >= self.max_event_duration:
                self.current_event_id = None
                self.event_duration = 0
        
        E_capture, n_captured = self.energy_model.compute_capture_energy(
            capture_mode
        )
        
        n_avail_proc = self.buffer.occupancy_raw + n_captured
        E_process, n_processed = self.energy_model.compute_processing_energy(
            process_mode, n_avail_proc
        )
        
        (
            occupancy_raw, occupancy_proc, n_dropped_raw, n_dropped_proc,
            n_simple_proc, n_complex_proc, dropped_raw_images, dropped_proc_images
        ) = self.buffer.update(
            n_captured, n_processed, process_mode, 
            self.current_event_id, self.timestep
        )
        
        n_simple_tx, n_complex_tx, transmitted_images = 0, 0, []
        E_tx = 0.0
        
        if tx_mode > 0:
            n_tx_limit = min(
                self.params.n_tx_img_max, 
                self.buffer.occupancy_proc
            )
            E_tx = self.energy_model.compute_transmission_energy(
                n_tx_limit, tx_mode
            )
            n_simple_tx, n_complex_tx, transmitted_images = (
                self.buffer.transmit(n_tx_limit)
            )
        else:
            E_tx = self.energy_model.compute_transmission_energy(0, tx_mode)
        
        n_total_tx = n_simple_tx + n_complex_tx
        
        if self.timestep % 50 == 0:
            self.buffer.finalize_old_events(
                self.timestep, 
                event_timeout=self.event_timeout
            )
        
        E_standby = self.energy_model.compute_standby_energy(sleep_mode)
        E_total = E_capture + E_process + E_tx + E_standby
        
        self.battery.update(E_harvested, E_total)
        self._update_temporal_features()
        
        self.GHI_window = np.roll(self.GHI_window, 1)
        self.GHI_window[0] = GHI
        
        # 🔥 FIXED REWARD COMPUTATION
        reward, reward_breakdown = self._compute_reward(
            n_simple_tx, n_complex_tx, tx_mode, E_total, E_harvested,
            n_dropped_raw, n_dropped_proc, transmitted_images,
            dropped_raw_images, dropped_proc_images
        )
        
        self.episode_stats['total_reward'] += reward
        self.episode_stats['images_transmitted'] += n_total_tx
        self.episode_stats['energy_harvested'] += E_harvested
        self.episode_stats['energy_consumed'] += E_total
        
        event_stats = self.buffer.get_event_delivery_stats()
        self.episode_stats['important_events_delivered'] = (
            event_stats['important_delivered']
        )
        self.episode_stats['important_events_missed'] = (
            event_stats['important_missed']
        )
        self.episode_stats['total_important_events'] = (
            event_stats['total_important_events']
        )
        
        terminated = self.battery.is_depleted()
        truncated = self.episode_steps >= self.params.max_episode_steps
        
        buffer_stats = self.buffer.get_stats()
        info = {
            'was_overridden': was_overridden,
            'intervention_type': intervention_type,
            'action_components': self.prev_action_components,
            'battery_level': self.battery.charge,
            'battery_percentage': self.battery.get_percentage(),
            'battery_soc': self.battery.get_normalized_level(),
            'buffer_occupancy_raw': self.buffer.occupancy_raw,
            'buffer_occupancy_proc': self.buffer.occupancy_proc,
            'n_transmitted': n_total_tx,
            'n_dropped_raw': n_dropped_raw,
            'n_dropped_proc': n_dropped_proc,
            'energy_harvested': E_harvested,
            'energy_consumed': E_total,
            'GHI': GHI,
            'reward_breakdown': reward_breakdown,
            'event_stats': event_stats,
            'current_event_id': self.current_event_id,
            'timestep': self.timestep,
            'episode_step': self.episode_steps
        }
        
        self.timestep += 1
        
        return self._get_observation(), reward, terminated, truncated, info
    
    def close(self) -> None:
        """Clean up resources."""
        pass

logger.info("✅ Part 3: Environment loaded (PROFESSIONAL RL REDESIGN)")
logger.info("   🔥 Reward clipping: [-10, +10]")
logger.info("   🔥 Expected value rewards (reduced variance)")
logger.info("   🔥 Energy penalty added")
