"""
PPO Training for Energy Harvesting Camera System - Jetson Nano
Part 2: Energy Model and Buffer Management

This module handles:
- Energy consumption calculations for all operations
- Solar energy harvesting
- Dual-buffer management with event tracking
- Battery state management

Author: Revised Implementation
Date: 2026-02-06
"""

import numpy as np
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass
import logging
"""
Part 2: Energy Model and Buffer Management
"""

import numpy as np
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass
import logging

# ============================================================================
# IMPORT FROM PART 1 (Or redefine here)
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
    n_images_captured: int = 0
    n_images_processed: int = 0
    n_images_transmitted: int = 0
    n_accurate_transmissions: int = 0
# Assuming Part 1 imports
# from src.env.core import SystemParameters, ImageMetadata, EventInfo

logger = logging.getLogger(__name__)

# ============================================================================
# ENERGY MODEL
# ============================================================================

class EnergyModel:
    """
    Energy consumption and harvesting model for Jetson Nano.
    
    All energy values in Joules, power in Watts.
    Tracks previous states to apply initialization costs correctly.
    """
    
    def __init__(self, params):  # SystemParameters
        """
        Initialize energy model.
        
        Args:
            params: SystemParameters instance
        """
        self.params = params
        
        # State tracking for initialization costs
        self.prev_capture_mode = 0
        self.prev_process_mode = 0
        self.prev_tx_mode = 0
        self.wifi_active = False
    
    def compute_harvested_energy(self, GHI: float) -> float:
        """
        Calculate solar energy harvested during time step.
        
        Args:
            GHI: Global Horizontal Irradiance (W/m²)
        
        Returns:
            Energy harvested (J)
        """
        energy = (
            self.params.eta_solar *      # Panel efficiency
            self.params.A_panel *         # Panel area (m²)
            GHI *                         # Irradiance (W/m²)
            self.params.dt                # Time duration (s)
        )
        return max(0.0, energy)
    
    def compute_capture_energy(self, capture_mode: int) -> Tuple[float, int]:
        """
        Calculate energy consumed by image capture according to documentation.
                
        E_cap = 0 if a_t = 0
        E_cap = I(prev=0) * E_init_cap + {
            N_t * e_active_low + E_log_low                     if a_t = 1
            E_start_high + (N_t - 1) * (e_cap_high + e_stby_high) + E_log_high if a_t = 2
        }
        """
        if capture_mode == 0:
            energy = 0.0
            n_captured = 0
        elif capture_mode == 1:
            n_captured = self.params.n_low
            # Energy = N * Active + Logging
            energy = n_captured * self.params.low_active_per_image + self.params.low_logging_fixed_round
        else:  # capture_mode == 2
            n_captured = self.params.n_high
            # 1. Initialization (Startup includes first image capture)
            energy = self.params.high_startup_fixed_round
            
            # 2. Steady State (Remaining images)
            # standby = delay * power_standby
            if n_captured > 1:
                energy += (n_captured - 1) * (
                    self.params.high_capture_per_image + 
                    (self.params.high_inter_capture_delay * self.params.high_standby_power)
                )
            
            # 3. Teardown (Logging)
            energy += self.params.high_logging_fixed_round

        # Add initialization energy if transitioning from System Idle (Cold Boot)
        if self.prev_capture_mode == 0 and capture_mode > 0:
            energy += self.params.E_init_cap

        # Update previous mode for the next epoch
        self.prev_capture_mode = capture_mode

        return energy, n_captured


    
    def compute_processing_energy(
        self, 
        process_mode: int, 
        n_available: int
    ) -> Tuple[float, int]:
        """
        Calculate energy consumed by image processing (inference).
        
        Args:
            process_mode: 0=Off, 1=Simple model, 2=Complex model
            n_available: Number of images available to process
        
        Returns:
            Tuple of (energy_consumed, n_images_processed)
        """
        energy = 0.0
        n_processed = 0
        
        if process_mode == 0 or n_available == 0:
            # Processing off or no images to process
            energy = 0.0
            n_processed = 0
            
        else:
            # Select processing parameters based on mode
            if process_mode == 1:  # Simple model
                max_capacity = self.params.n_proc_simple_max
                e_proc = self.params.e_proc_simple
                E_log = self.params.E_simple_log
                E_init = self.params.E_proc_init_simple
            else:  # process_mode == 2 (Complex model)
                max_capacity = self.params.n_proc_complex_max
                e_proc = self.params.e_proc_complex
                E_log = self.params.E_complex_log
                E_init = self.params.E_proc_init_complex
            
            # Determine actual throughput
            n_processed = min(n_available, max_capacity)
            
            # Calculate energy: throughput * per-image + logging
            energy = (n_processed * e_proc) + E_log
            
            # Add initialization cost if waking up from Mode 0
            if self.prev_process_mode == 0:
                energy += E_init
        
        self.prev_process_mode = process_mode
        return energy, n_processed
    
    
    
    def compute_transmission_energy(
        self, 
        n_transmit: int, 
        tx_mode: int
    ) -> float:
        """
        Calculate transmission energy with WiFi cold-start penalty.
        
        Args:
            n_transmit: Number of items to transmit
            tx_mode: 0=Off, 1=Results only, 2=Images
        
        Returns:
            Energy consumed (J)
        """
        energy = 0.0
        
        # Case 1: Transmission off
        if tx_mode == 0 or (n_transmit == 0 and tx_mode != 1):
            energy = 0.0
            
            # Jetson penalty: cost to turn WiFi OFF
            if self.prev_tx_mode > 0:
                energy += self.params.E_wifi_off
            
            self.prev_tx_mode = 0
            
        # Case 2: Transmission active
        else:
            is_cold_start = (self.prev_tx_mode == 0)
            
            # 1. Initialization / Cold start
            if is_cold_start:
                energy += self.params.E_init_tx
                # Cold start active duration
                energy += self.params.dt_proc * self.params.P_wifi
            else:
                # Steady state active duration
                energy += self.params.dt * self.params.P_wifi
            
            # 2. Workload energy
            if tx_mode == 1:  # Results only
                e_tx_single = (
                    self.params.e_tx_res_total + 
                    self.params.e_tx_log_total
                )
                energy += n_transmit * e_tx_single
                
            elif tx_mode == 2:  # Images
                e_tx_single = (
                    self.params.e_tx_img_total +
                    self.params.e_tx_res_total +
                    self.params.e_tx_log_total
                )
                energy += n_transmit * e_tx_single
            
            self.prev_tx_mode = tx_mode
        
        return energy
    
    def compute_standby_energy(self, sleep_mode: int) -> float:
        """
        Calculate standby/background energy consumption.
        
        Args:
            sleep_mode: 0=Active, 1=Idle, 2=Deep Sleep
        
        Returns:
            Energy consumed (J)
        """
        if sleep_mode == 0:      # Active
            power = self.params.P_active
        elif sleep_mode == 1:    # Idle
            power = self.params.P_idle
        else:                    # Deep Sleep (2)
            power = self.params.P_sleep
        
        energy = power * self.params.dt
        return energy
    
    def reset(self) -> None:
        """Reset internal state variables."""
        self.prev_capture_mode = 0
        self.prev_process_mode = 0
        self.prev_tx_mode = 0
        self.wifi_active = False

# ============================================================================
# BUFFER MANAGER
# ============================================================================

class EnhancedDualBufferManager:
    """
    Dual FIFO buffer manager with comprehensive event tracking.
    
    Manages two buffers:
    - Raw buffer: Unprocessed images
    - Processed buffer: Inference-processed images ready for transmission
    
    Tracks events to compute delivery statistics.
    """
    
    def __init__(
        self, 
        max_capacity_raw: int, 
        max_capacity_proc: int, 
        params  # SystemParameters
    ):
        """
        Initialize buffer manager.
        
        Args:
            max_capacity_raw: Maximum raw buffer capacity
            max_capacity_proc: Maximum processed buffer capacity
            params: SystemParameters instance
        """
        self.max_capacity_raw = max_capacity_raw
        self.max_capacity_proc = max_capacity_proc
        self.params = params
        
        # FIFO buffers
        self.raw_buffer: List = []  # List[ImageMetadata]
        self.proc_buffer: List = []  # List[ImageMetadata]
        
        # Buffer occupancy tracking
        self.occupancy_raw = 0
        self.occupancy_proc = 0
        
        # Event tracking
        self.active_events: Dict[int, any] = {}  # Dict[int, EventInfo]
        self.completed_events: List = []  # List[EventInfo]
        
        # ID generators
        self.next_image_id = 0
        self.next_event_id = 0
        self.current_timestep = 0
    


    def generate_event(self, timestep: int) -> Optional[int]:
        """Generate random surveillance event."""
        if np.random.random() < self.params.prob_event_per_step:
            event_id = self.next_event_id
            self.next_event_id += 1
            
            is_important = (
                np.random.random() < self.params.prob_important_given_event
            )
            
            # ✅ USE PROPER DATACLASS
            event_info = EventInfo(
                event_id=event_id,
                is_important=is_important,
                start_time=timestep,
                n_images_captured=0,
                n_images_processed=0,
                n_images_transmitted=0,
                n_accurate_transmissions=0
            )
            
            self.active_events[event_id] = event_info
            return event_id
        
        return None
    
    def capture_images(
        self, 
        n_captured: int, 
        current_event_id: Optional[int],
        timestep: int
    ) -> List[ImageMetadata]:
        """Create metadata for captured images."""
        captured_images = []
        
        for _ in range(n_captured):
            if current_event_id is not None:
                event_id = current_event_id
                event_info = self.active_events.get(current_event_id)
                is_important = event_info.is_important if event_info else False
                if event_info:
                    event_info.n_images_captured += 1
            else:
                event_id = None
                is_important = False
            
            # ✅ USE PROPER DATACLASS
            image = ImageMetadata(
                image_id=self.next_image_id,
                event_id=event_id,
                is_important=is_important,
                capture_time=timestep,
                processing_model=0
            )
            
            self.next_image_id += 1
            captured_images.append(image)
        
        return captured_images
    
    def update(
        self,
        n_captured: int,
        n_processed: int,
        process_mode: int,
        current_event_id: Optional[int],
        timestep: int
    ) -> Tuple:
        """
        Update buffers after capture and processing operations.
        
        Args:
            n_captured: Number of images captured
            n_processed: Number of images to process
            process_mode: Processing mode (0=off, 1=simple, 2=complex)
            current_event_id: Active event ID
            timestep: Current time step
        
        Returns:
            Tuple of (occupancy_raw, occupancy_proc, n_dropped_raw, 
                     n_dropped_proc, n_simple_processed, n_complex_processed,
                     dropped_raw_images, dropped_proc_images)
        """
        self.current_timestep = timestep
        
        n_simple_processed = 0
        n_complex_processed = 0
        dropped_raw_images = []
        dropped_proc_images = []
        
        # 1. Capture images and add to raw buffer
        captured_images = self.capture_images(n_captured, current_event_id, timestep)
        for image in captured_images:
            self.raw_buffer.append(image)
        
        # 2. Process images: raw → processed
        actual_processed = min(n_processed, len(self.raw_buffer))
        for _ in range(actual_processed):
            if process_mode > 0:
                image = self.raw_buffer.pop(0)  # FIFO
                image.processing_model = process_mode
                self.proc_buffer.append(image)
                
                # Update event statistics
                if image.event_id is not None and image.event_id in self.active_events:
                    self.active_events[image.event_id].n_images_processed += 1
                
                # Count by model type
                if process_mode == 1:
                    n_simple_processed += 1
                elif process_mode == 2:
                    n_complex_processed += 1
        
        # 3. Handle raw buffer overflow
        n_dropped_raw = 0
        if len(self.raw_buffer) > self.max_capacity_raw:
            n_dropped_raw = len(self.raw_buffer) - self.max_capacity_raw
            dropped_raw_images = self.raw_buffer[:n_dropped_raw]
            self.raw_buffer = self.raw_buffer[n_dropped_raw:]  # Keep newest
        
        # 4. Handle processed buffer overflow
        n_dropped_proc = 0
        if len(self.proc_buffer) > self.max_capacity_proc:
            n_dropped_proc = len(self.proc_buffer) - self.max_capacity_proc
            dropped_proc_images = self.proc_buffer[:n_dropped_proc]
            self.proc_buffer = self.proc_buffer[n_dropped_proc:]  # Keep newest
        
        # Update occupancy
        self.occupancy_raw = len(self.raw_buffer)
        self.occupancy_proc = len(self.proc_buffer)
        
        return (
            self.occupancy_raw, 
            self.occupancy_proc, 
            n_dropped_raw, 
            n_dropped_proc,
            n_simple_processed, 
            n_complex_processed, 
            dropped_raw_images, 
            dropped_proc_images
        )
    
    def transmit(self, n_transmit_limit: int) -> Tuple[int, int, List]:
        """
        Transmit images from processed buffer.
        
        Args:
            n_transmit_limit: Maximum number of images to transmit
        
        Returns:
            Tuple of (n_simple_tx, n_complex_tx, transmitted_images)
        """
        n_simple_tx = 0
        n_complex_tx = 0
        transmitted_images = []
        
        actual_tx = min(n_transmit_limit, len(self.proc_buffer))
        
        for _ in range(actual_tx):
            image = self.proc_buffer.pop(0)  # FIFO
            transmitted_images.append(image)
            
            # Count by model type
            if image.processing_model == 1:
                n_simple_tx += 1
            elif image.processing_model == 2:
                n_complex_tx += 1
            
            # Update event statistics
            if image.event_id is not None and image.event_id in self.active_events:
                event_info = self.active_events[image.event_id]
                event_info.n_images_transmitted += 1
                
                # Check if transmission was accurate (based on model accuracy)
                if image.processing_model == 1:
                    accuracy = self.params.acc_simple
                elif image.processing_model == 2:
                    accuracy = self.params.acc_complex
                else:
                    accuracy = 0.0
                
                # Accurate transmission if important and model succeeds
                if image.is_important and np.random.random() < accuracy:
                    event_info.n_accurate_transmissions += 1
        
        self.occupancy_proc = len(self.proc_buffer)
        return n_simple_tx, n_complex_tx, transmitted_images
    
    def finalize_old_events(
        self, 
        current_timestep: int, 
        event_timeout: int = 100
    ) -> None:
        """
        Move old events from active to completed list.
        
        Args:
            current_timestep: Current time step
            event_timeout: Steps after which event is considered complete
        """
        events_to_finalize = []
        
        for event_id, event_info in self.active_events.items():
            if current_timestep - event_info.start_time > event_timeout:
                events_to_finalize.append(event_id)
        
        for event_id in events_to_finalize:
            event_info = self.active_events.pop(event_id)
            self.completed_events.append(event_info)
    
    def get_event_delivery_stats(self) -> Dict[str, any]:
        """
        Compute event delivery statistics.
        
        Returns:
            Dictionary with delivery metrics
        """
        total_important_events = 0
        important_delivered = 0
        important_missed = 0
        total_quality_score = 0.0
        
        all_events = list(self.active_events.values()) + self.completed_events
        
        for event_info in all_events:
            if event_info.is_important:
                total_important_events += 1
                
                if event_info.n_accurate_transmissions > 0:
                    important_delivered += 1
                    # Quality: how many accurate transmissions (up to 3)
                    quality = min(event_info.n_accurate_transmissions, 3.0) / 3.0
                    total_quality_score += quality
                else:
                    important_missed += 1
        
        delivery_rate = (
            important_delivered / max(total_important_events, 1)
        )
        
        return {
            'total_important_events': total_important_events,
            'important_delivered': important_delivered,
            'important_missed': important_missed,
            'quality_score': total_quality_score,
            'delivery_rate': delivery_rate
        }
    
    def get_simple_count(self) -> int:
        """Count simple-processed images in buffer."""
        return sum(1 for img in self.proc_buffer if img.processing_model == 1)
    
    def get_complex_count(self) -> int:
        """Count complex-processed images in buffer."""
        return sum(1 for img in self.proc_buffer if img.processing_model == 2)
    
    def get_stats(self) -> Dict[str, int]:
        """Get current buffer statistics."""
        return {
            'occupancy_raw': self.occupancy_raw,
            'occupancy_proc': self.occupancy_proc,
            'simple': self.get_simple_count(),
            'complex': self.get_complex_count()
        }
    
    def reset(self) -> None:
        """Reset all buffer state."""
        self.raw_buffer = []
        self.proc_buffer = []
        self.occupancy_raw = 0
        self.occupancy_proc = 0
        self.active_events = {}
        self.completed_events = []
        self.next_image_id = 0
        self.next_event_id = 0
        self.current_timestep = 0

# ============================================================================
# BATTERY MANAGER
# ============================================================================

class BatteryManager:
    """
    Battery state management with leakage model.
    
    Tracks charge level and provides safety checks.
    """
    
    def __init__(
        self,
        capacity: float,
        min_level: float,
        leak_rate: float,
        initial_soc: float = 0.8
    ):
        """
        Initialize battery manager.
        
        Args:
            capacity: Maximum battery capacity (J)
            min_level: Minimum safe battery level (J)
            leak_rate: Leakage rate per time step
            initial_soc: Initial state of charge (0-1)
        """
        self.capacity = capacity
        self.min_level = min_level
        self.leak_rate = leak_rate
        self.initial_soc = initial_soc
        self.charge = capacity * initial_soc
    
    def update(
        self, 
        energy_harvested: float, 
        energy_consumed: float
    ) -> float:
        """
        Update battery charge.
        
        Args:
            energy_harvested: Energy gained from solar (J)
            energy_consumed: Energy used by operations (J)
        
        Returns:
            New charge level (J)
        """
        self.charge = (
            self.charge * (1 - self.leak_rate)  # Leakage
            - energy_consumed                    # Consumption
            + energy_harvested                   # Harvesting
        )
        
        # Clamp to valid range
        self.charge = max(0.0, min(self.charge, self.capacity))
        return self.charge
    
    def is_critical(self) -> bool:
        """Check if battery is critically low."""
        return self.charge < self.min_level
    
    def is_depleted(self) -> bool:
        """Check if battery is depleted."""
        return self.charge <= 0
    
    def get_normalized_level(self) -> float:
        """Get normalized battery level (0-1)."""
        return self.charge / self.capacity
    
    def get_percentage(self) -> float:
        """Get battery percentage (0-100)."""
        return 100.0 * self.get_normalized_level()
    
    def reset(self) -> None:
        """Reset battery to initial SOC."""
        self.charge = self.capacity * self.initial_soc

logger.info("✅ Part 2: Energy Model and Buffer Management loaded successfully")
