

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import json
from pathlib import Path
from typing import Tuple, Dict, Optional, List, Any, Deque
from dataclasses import dataclass, field, asdict
from collections import deque
from abc import ABC, abstractmethod

# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Setup logger with consistent formatting."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

logger = setup_logger(__name__)

# ============================================================================
# ACTION MAPPING UTILITIES
# ============================================================================

def tuple_to_flat(action_tuple: Tuple[int, int, int, int]) -> int:
    """Convert (sleep, capture, process, tx) tuple to flat action index."""
    sleep, capture, process, tx = action_tuple
    return sleep * 18 + capture * 6 + process * 2 + tx


def flat_to_tuple(flat_action: int) -> Tuple[int, int, int, int]:
    """Convert flat action index to tuple."""
    sleep = flat_action // 18
    capture = (flat_action % 18) // 6
    process = (flat_action % 6) // 2
    tx = flat_action % 2
    return (sleep, capture, process, tx)

# ============================================================================
# SYSTEM PARAMETERS - PROFESSIONALLY TUNED FOR PPO
# ============================================================================

@dataclass
class SystemParameters:
    
    
    # Battery & Energy
    B_max: float = 70000.0
    B_min: float = 7000.0
    lambda_leak: float = 0.001
    
    # Buffers (Jetson)
    Q_max_raw: int = 888
    Q_max_proc: int = 500
    
    # Solar panel
    eta_solar: float = 0.17
    A_panel: float = 0.23
    
    # Timing
    dt: float = 60.0
    dt_proc: float = 53.51
    
    # Capture energy (Jetson MEASURED)
    E_init_cap: float = 0.8725
    n_low: int = 6
    low_active_per_image = 1.17215
    low_logging_fixed_round = 0.01193
    n_high: int = 120
    high_startup_fixed_round = 1.36947
    high_capture_per_image = 0.01049
    high_standby_power = 0.68225
    high_logging_fixed_round = 0.41937
    high_inter_capture_delay = 0.42

    
    # Processing energy (Jetson MEASURED)
    E_proc_init_simple: float = 0.02324
    E_proc_init_complex: float = 0.02467
    e_proc_simple: float = 0.05942
    e_proc_complex: float = 0.15358
    n_proc_simple_max: int = 840
    n_proc_complex_max: int = 360
    E_simple_log: float = 0.03524
    E_complex_log: float = 0.1871
    
    # Transmission energy (Jetson MEASURED)
    E_init_tx: float = 3.77708
    e_tx_img_total: float = 0.24879388
    e_tx_res_total: float = 0.004046873
    n_tx_res_max: int = 2000
    n_tx_img_max: int = 5
    e_tx_log_total: float = 0.047750
    P_wifi: float = 0.00985
    E_wifi_off: float = 0.543805
    
    # Standby power (Jetson MEASURED)
    P_active: float = 1.745815
    P_idle: float = 1.745815
    P_sleep: float = 0.2625
    
    # ========================================================================
    # 🔥 REWARD REDESIGN - PROFESSIONALLY TUNED FOR PPO
    # ========================================================================
    # Target range: typical step ∈ [-3, +3], hard ceiling [-10, +10]
    
    # PRIMARY OBJECTIVE - Compressed from 100→6 (83x reduction!)
    r_event_delivered: float = 6.0        # Was 100.0 - MAJOR FIX
    r_event_missed: float = -3.0          # Was -30.0 - Maintained ratio
    
    # QUALITY BONUSES - Rebalanced
    r_quality_bonus: float = 0.05          # Was 5.0 - 10x reduction
    r_redundancy_bonus: float = 0.002       # Was 2.0 - 10x reduction
    
    # PENALTIES - Increased importance
    r_overflow_raw: float = -0.001        # Minor
    r_overflow_proc: float = -0.001       # Minor
    r_overflow_important: float = -0.1    # Was -1.0 - DOUBLED (critical loss)
    r_energy_waste: float = -0.00001       # Minor
    r_battery_low: float = -0.5           # Was -5.0 - Slight reduction but still strong
    
    # SHAPING REWARDS - Rebalanced to 20-30% of total
    r_transmission_bonus: float = 0.15    # Was 0.1 - Increased for guidance
    r_processing_bonus: float = 0.07      # Was 0.05 - Increased for guidance
    # r_activity_bonus REMOVED - Was rewarding laziness!
    
    # ENERGY ECONOMICS - NEW! Make agent feel energy cost
    r_energy_penalty_scale: float = 0.00002 # Penalty per joule consumed
    
    # ========================================================================
    # EVENT GENERATION - REDUCED FOR CREDIT ASSIGNMENT
    # ========================================================================
    prob_event_per_step: float = 0.4    
    prob_important_given_event: float = 0.8
    images_per_event_mean: float = 100.0
    
    # Model accuracy
    acc_simple: float = 0.75
    acc_complex: float = 0.92
    
    # Quality multipliers
    phi_model_simple: float = 0.1
    phi_model_complex: float = 0.13
    phi_tx_full: float = 0.15
    r: float = 0.1
    
    # Safety thresholds
    soft_safety_threshold: float = 0.12
    safety_scale_factor: float = 0.5
    hard_safety_threshold: float = 0.05
    
    # Observation features
    include_temporal_features: bool = True
    ma_window_battery: int = 10
    ma_window_buffer: int = 5
    include_safety_flag: bool = True
    include_previous_action: bool = True
    window_size: int = 24
    
    # GHI processing
    use_linear_interpolation: bool = True
    
    # Episode configuration
    max_episode_steps: int = 1440
    initial_battery_soc: float = 0.6
    random_start_cycling: bool = True
    
    # Reward normalization - CRITICAL FOR PPO
    normalize_by_episode_length: bool = True
    reward_scale_factor: float = 1.0
    reward_clip_min: float = -10.0        # NEW! Hard ceiling
    reward_clip_max: float = 10.0         # NEW! Hard ceiling
    
    def validate(self) -> None:
        """Validate parameters."""
        errors = []
        
        # Battery validation
        if self.B_max <= 0:
            errors.append(f"B_max must be positive, got {self.B_max}")
        if self.B_min >= self.B_max:
            errors.append(f"B_min must be < B_max")
        
        # 🔥 REWARD MAGNITUDE VALIDATION - CRITICAL!
        if abs(self.r_event_delivered) > 15:
            errors.append(f"⚠️ r_event_delivered too large for PPO: {self.r_event_delivered}")
        if abs(self.r_event_missed) > 10:
            errors.append(f"⚠️ r_event_missed too large for PPO: {self.r_event_missed}")
        
        # Reward balance check
        if abs(self.r_event_delivered) < abs(self.r_event_missed) * 0.5:
            errors.append("Delivery reward should dominate missed penalty")
        
        # Probability validation
        if not (0 <= self.prob_event_per_step <= 1):
            errors.append("Event probability must be in [0, 1]")
        
        if errors:
            raise ValueError("Parameter validation failed:\n" + "\n".join(f"  - {e}" for e in errors))
        
        logger.info("✅ System parameters validated (PPO-safe reward scale)")
    
    def save(self, filepath: str) -> None:
        """Save parameters to JSON."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(asdict(self), f, indent=2)
        logger.info(f"Parameters saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'SystemParameters':
        """Load parameters from JSON."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls(**data)
    
    def get_curriculum_stage_name(self) -> str:
        """Infer curriculum stage from current parameters."""
        if self.initial_battery_soc >= 0.6 and abs(self.r_event_missed) <= 5:
            return "Easy"
        elif self.initial_battery_soc >= 0.4 and abs(self.r_event_missed) <= 8:
            return "Medium"
        else:
            return "Hard"

# ============================================================================
# CURRICULUM LEARNING - ADJUSTED FOR NEW REWARD SCALE
# ============================================================================

@dataclass
class CurriculumStage:
    """Configuration for one curriculum learning stage."""
    
    name: str
    n_episodes: int
    initial_battery_soc: float
    prob_event_per_step: float
    r_event_delivered: float
    r_event_missed: float
    learning_rate: float
    entropy_coef: float
    images_per_event_mean: float = 100.0
    n_tx_img_max: int = 5
    event_timeout: int = 100
    
    def apply_to_params(self, params: SystemParameters) -> SystemParameters:
        """Apply stage configuration to system parameters."""
        params.initial_battery_soc = self.initial_battery_soc
        params.prob_event_per_step = self.prob_event_per_step
        params.r_event_delivered = self.r_event_delivered
        params.r_event_missed = self.r_event_missed
        params.images_per_event_mean = self.images_per_event_mean
        params.n_tx_img_max = self.n_tx_img_max
        return params
    
    def get_description(self) -> str:
        """Get human-readable stage description."""
        return (
            f"{self.name} Stage:\n"
            f"  Episodes: {self.n_episodes}\n"
            f"  Battery SOC: {self.initial_battery_soc:.0%}\n"
            f"  Event Probability: {self.prob_event_per_step:.2%}\n"
            f"  Images per Event: {self.images_per_event_mean:.0f}\n"
            f"  TX Capacity: {self.n_tx_img_max}\n"
            f"  Event Timeout: {self.event_timeout}\n"
            f"  Missed Penalty: {self.r_event_missed:.1f}\n"
            f"  Delivery Reward: {self.r_event_delivered:.1f}\n"
            f"  Learning Rate: {self.learning_rate:.1e}\n"
            f"  Entropy Coef: {self.entropy_coef:.2f}"
        )


def create_curriculum_stages() -> List[CurriculumStage]:
    """
    Create curriculum with PPO-SAFE reward magnitudes.
    
    🔥 KEY CHANGES:
    - Event frequency MUCH lower (better credit assignment)
    - Reward magnitudes compressed to [-10, +10] range
    - Gradual increase in difficulty
    """
    
    stages = [
        # Tutorial: Learn that delivery = good, with very sparse events
        CurriculumStage(
            name="Tutorial_Ultra_Easy",
            n_episodes=200,
            initial_battery_soc=0.99,
            prob_event_per_step=0.05,        
            r_event_delivered=6.0,
            r_event_missed=-1.0,              # Gentle
            images_per_event_mean=10.0,
            n_tx_img_max=10,
            event_timeout=500,
            learning_rate=1e-3,
            entropy_coef=0.30
        ),
        
        CurriculumStage(
            name="Tutorial",
            n_episodes=500,
            initial_battery_soc=0.95,
            prob_event_per_step=0.1,         
            r_event_delivered=6.0,
            r_event_missed=-2.0,
            images_per_event_mean=20.0,
            n_tx_img_max=9,
            event_timeout=300,
            learning_rate=1e-3,
            entropy_coef=0.20
        ),
        
        CurriculumStage(
            name="Easy",
            n_episodes=800,
            initial_battery_soc=0.90,
            prob_event_per_step=0.15,        
            r_event_delivered=6.0,
            r_event_missed=-2.5,
            images_per_event_mean=30.0,
            n_tx_img_max=9,
            event_timeout=300,
            learning_rate=8e-4,
            entropy_coef=0.18
        ),
        
        CurriculumStage(
            name="Medium",
            n_episodes=1000,
            initial_battery_soc=0.80,
            prob_event_per_step=0.2,         
            r_event_delivered=6.0,
            r_event_missed=-3.0,
            images_per_event_mean=50.0,
            n_tx_img_max=7,
            event_timeout=250,
            learning_rate=5e-4,
            entropy_coef=0.15
        ),
        
        CurriculumStage(
            name="Hard",
            n_episodes=1000,
            initial_battery_soc=0.60,
            prob_event_per_step=0.25,        
            r_event_delivered=6.0,
            r_event_missed=-3.5,
            images_per_event_mean=75.0,
            n_tx_img_max=6,
            event_timeout=200,
            learning_rate=3e-4,
            entropy_coef=0.12
        ),
        
        CurriculumStage(
            name="Expert",
            n_episodes=1200,
            initial_battery_soc=0.50,
            prob_event_per_step=0.3,         
            r_event_delivered=6.0,
            r_event_missed=-4.0,
            images_per_event_mean=100.0,
            n_tx_img_max=5,
            event_timeout=150,
            learning_rate=1e-4,
            entropy_coef=0.10
        )
    ]
    
    return stages

# ============================================================================
# GHI DATA MANAGEMENT
# ============================================================================

class GHILoader:
    """Solar irradiance (GHI) data loader with interpolation."""
    
    def __init__(self, data_path: str, use_linear_interpolation: bool = True):
        """Initialize GHI loader."""
        path = Path(data_path)
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        self.data_10min = pd.read_csv(data_path)
        
        if 'GHI' not in self.data_10min.columns:
            raise ValueError("CSV must contain 'GHI' column")
        
        self.data_10min['GHI'].fillna(0, inplace=True)
        self.data_10min['GHI'] = self.data_10min['GHI'].clip(lower=0)
        
        if use_linear_interpolation:
            GHI_1min = self._interpolate_linear()
        else:
            GHI_1min = self._repeat_values()
        
        self.data = pd.DataFrame({'GHI': GHI_1min})
        self.max_steps = len(self.data)
        
        logger.info(f"Loaded {self.max_steps} 1-minute GHI points")
    
    def _interpolate_linear(self) -> List[float]:
        """Perform linear interpolation between 10-minute points."""
        GHI_1min = []
        for i in range(len(self.data_10min)):
            current_val = self.data_10min.iloc[i]['GHI']
            
            if i < len(self.data_10min) - 1:
                next_val = self.data_10min.iloc[i + 1]['GHI']
                for j in range(10):
                    interpolated = current_val + (next_val - current_val) * (j / 10.0)
                    GHI_1min.append(interpolated)
            else:
                GHI_1min.extend([current_val] * 10)
        
        return GHI_1min
    
    def _repeat_values(self) -> List[float]:
        """Simply repeat each 10-minute value 10 times."""
        GHI_1min = []
        for value in self.data_10min['GHI']:
            GHI_1min.extend([value] * 10)
        return GHI_1min
    
    def get_GHI(self, timestep: int) -> float:
        """Get GHI value at specific timestep with wraparound."""
        idx = timestep % self.max_steps
        return float(self.data.iloc[idx]['GHI'])


def create_synthetic_ghi_data(output_path: str = "NREL.csv") -> None:
    """Create synthetic GHI data for testing."""
    logger.info("Creating synthetic GHI data...")
    Path("data").mkdir(exist_ok=True)
    
    n_points = 144
    hours = np.linspace(0, 24, n_points)
    ghi = np.maximum(0, 800 * np.sin(np.pi * (hours - 6) / 12))
    ghi += np.random.normal(0, 50, n_points)
    ghi = np.maximum(0, ghi)
    
    df = pd.DataFrame({'GHI': ghi})
    df.to_csv(output_path, index=False)
    logger.info(f"✅ Synthetic GHI data saved to {output_path}")

# ============================================================================
# METADATA CLASSES
# ============================================================================

@dataclass
class ImageMetadata:
    """Metadata for a single captured image."""
    image_id: int
    event_id: Optional[int]
    is_important: bool
    capture_time: int
    processing_model: int


@dataclass
class EventInfo:
    """Information about a surveillance event."""
    event_id: int
    is_important: bool
    start_time: int
    n_images_captured: int
    n_images_processed: int
    n_images_transmitted: int
    n_accurate_transmissions: int

# ============================================================================
# INITIALIZATION
# ============================================================================

if not Path("NREL.csv").exists():
    create_synthetic_ghi_data()

logger.info("✅ Part 1: Core Infrastructure loaded (PROFESSIONAL RL REDESIGN)")
logger.info("   🔥 Reward magnitude: 500→6 (83x compression)")
logger.info("   🔥 Event frequency: 0.25→0.01 (25x reduction)")
logger.info("   🔥 Activity reward: REMOVED")
logger.info("   🔥 Battery penalty: INCREASED")
