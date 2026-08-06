# Energy-Harvesting Edge Vision Systems via Reinforcement Learning

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status: Under Review](https://img.shields.io/badge/Status-Under%20Review-orange)](https://github.com/TODO)

**A unified RL framework for autonomous energy-aware camera control on energy-harvesting IoT devices. Jointly optimizes capture frequency, inference complexity, and transmission to maximize event detection under stochastic power constraints.**

## Overview

### The Problem
Energy-harvesting vision devices face conflicting objectives: limited/intermittent power, need for continuous monitoring, and unpredictable solar availability. Fragmented approaches (duty-cycling, multi-model selection, MPC) fail because they don't co-optimize across the full pipeline.

### Solution
A single PPO agent learns energy-aware policies through curriculum learning + runtime safety constraints. Achieves **82.6% event delivery** on Jetson Nano with **100% system reliability** (no battery failures).

### Key Results

| Platform | Policy | Delivery Rate | Completion |
|----------|--------|----------------|-----------|
| Jetson Nano | **PPO (safe)** | **82.6%** | **100%** |
| Jetson Nano | Smart Heuristic | 79.1% | 100% |
| Raspberry Pi | **PPO (safe)** | **71.8%** | **100%** |
| Raspberry Pi | DQN | 54.2% | 100% |

### Main Contributions
1. **Unified end-to-end learning** across capture/inference/transmission (vs. tuning each separately)
2. **Hardware-grounded energy model** with measured profiling on real devices
3. **Guaranteed safety**: 100% completion rate via curriculum + runtime wrapper
4. **Fast transfer**: 50× speedup deploying to new hardware via fine-tuning

---

## Repository Structure

```
EnergyHarvestingRL/
├── scripts/train.py                # Entry point (train/test/diagnose)
├── src/
│   ├── env/                        # Environment: MDP, energy model, solar data
│   │   ├── core.py                 # System params, curriculum, GHI loader
│   │   ├── energy.py               # Energy model, battery/buffer managers
│   │   └── environment.py          # Gymnasium environment
│   ├── agents/                     # RL agents
│   │   ├── ppo.py                  # PPO: actor-critic networks, training
│   │   └── policy_wrapper.py       # Safety wrapper (energy enforcement)
│   ├── baselines/                  # Baseline policies
│   │   ├── heuristic.py            # Threshold-based policies
│   │   ├── dqn.py                  # DQN baseline
│   │   └── mpc.py                  # MPC baseline
│   ├── training/                   # Training logic
│   │   ├── trainer.py              # Curriculum progression, checkpoints
│   │   └── mastery.py              # Transfer learning utilities
│   └── evaluation/                 # Metrics and diagnostics
│       ├── comprehensive.py        # Policy comparisons
│       ├── diagnostics.py          # Preflight checks
│       ├── computational_benchmark.py
│       └── statistics.py
├── checkpoints/                    # Trained models
├── results/                        # Evaluation outputs (CSV, JSON, plots)
└── requirements.txt
```

### Key Modules

| Module | Role |
|--------|------|
| `core.py` | System parameters, curriculum stages, solar data loading |
| `energy.py` | Parametric energy model, battery/buffer simulation |
| `environment.py` | Gymnasium-compatible MDP environment |
| `ppo.py` | PPO agent with actor-critic networks |
| `policy_wrapper.py` | Runtime safety enforcement via action masking |
| `trainer.py` | Curriculum training, checkpointing, evaluation |
| `baselines/` | Heuristic, DQN, MPC baselines for comparison |
| `evaluation/` | Comprehensive metrics and diagnostics |

---

## System Architecture

### Pipeline Overview

```
Solar Input → Environment → Observation (44D) → PPO Agent → Action
                                                     ↓
                                              Safety Wrapper
                                                     ↓
                            Capture → Inference → Transmission
```

### MDP Formulation

- **State** (44D): Battery, buffer occupancy, solar history, event flag, previous action
- **Action** (multi-discrete): `[sleep, capture, inference, tx]` ∈ {0-1} × {0-2} × {0-2} × {0-2}
- **Reward**: `R = α·delivery - β·energy - γ·buffer - Ω·safety_penalty` (clipped to [-10, +10])
- **Episode Length**: 1000 steps (~17 minutes)

---

## Installation

```bash
git clone https://github.com/your-org/EnergyHarvestingRL.git
cd EnergyHarvestingRL

python3.10 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

**Requirements**: Python ≥3.10, PyTorch, Gymnasium, NumPy, Pandas, Matplotlib, SciPy

---

## Quick Start

### Train PPO with Curriculum Learning

```bash
python scripts/train.py --mode train --save-dir checkpoints/my_experiment --num-episodes 4700
```

Outputs: trained model, training metrics, evaluation results in `results/`

### Evaluate Trained Policy

```bash
python scripts/train.py --mode test \
  --checkpoint checkpoints/my_experiment/final_policy.pth \
  --n-episodes 100
```

Computes: reward, delivery rate, battery safety, energy efficiency

### Compare All Baselines

```python
from src.evaluation.comprehensive import ComprehensiveEvaluation
from pathlib import Path

eval = ComprehensiveEvaluation(save_dir=Path('results/comparison'), n_episodes=50)
eval.run_all_comparisons()
eval.plot_results()
```

### Run Diagnostics

```bash
python scripts/train.py --mode diagnose
```

Validates: GHI data, energy model, curriculum, observation/action spaces, safety wrapper

---

## Methodology

### PPO Algorithm
- **Actor-Critic** architecture: shared feature extractor (256 hidden, 2 layers) + independent policy/value heads
- **Training**: On-policy rollouts, GAE advantage estimation (λ=0.95), clipped objective (ε=0.2), 5 epochs/update
- **Hyperparameters**: LR=3×10⁻⁴, γ=0.99, entropy=0.1 (annealed), gradient clipping=0.5

### Curriculum Learning (6 Stages)

Progressive difficulty: event rate 0.1→0.3, initial battery 70K J→10K J

| Stage | Events | Battery | Episodes | Goal |
|-------|--------|---------|----------|------|
| 1 | 0.1 | 70K J | 100 | Learn basics |
| 2 | 0.15 | 70K J | 200 | Add constraints |
| 3 | 0.2 | 50K J | 500 | Tighten slack |
| 4 | 0.25 | 35K J | 800 | Realistic pressure |
| 5 | 0.3 | 20K J | 1200 | High difficulty |
| 6 | 0.3 | 10K J | 1900 | Deployment |

**Warm-start**: Network weights carry over between stages

### Safety Wrapper
- Predicts energy cost of each action
- Overrides actions that would deplete battery below `B_min`
- Fallback hierarchy: TX → inference → capture → sleep
- **Result**: 100% episode completion (zero crashes), -6pp delivery for safety

### Transfer Learning
Train on Jetson Nano, fine-tune on Raspberry Pi in 10-50 episodes with LR=1×10⁻⁵
- **Speedup**: 50× faster (50 episodes vs. 500+)

## Training Pipeline

**Configuration** (`src/env/core.py`):
```python
from src.env.core import SystemParameters, create_curriculum_stages
params = SystemParameters(B_max=70_000, B_min=7_000, ...)  # Jetson Nano
curriculum = create_curriculum_stages(params, num_stages=6)
```

**Checkpoints**: Auto-saved every 100 episodes to `checkpoints/stage_{N}/ckpt_{step}.pth`

**Logging**: Real-time CSV with episode, reward, delivery_rate, battery_min, entropy, loss

## Baselines

All baselines evaluated under identical conditions (same environment, solar traces, seeds):

| Policy | Reward | Delivery | Completion | Notes |
|--------|--------|----------|-----------|-------|
| **PPO (safe)** | **804** | **82.6%** | **100%** | ✅ Learned, safe |
| Smart Heuristic | 696 | 79.1% | 100% | Best hand-crafted |
| DQN (safe) | 245 | 61.7% | 100% | Q-value instability |
| MPC (3-step) | -52 | 58.3% | 12.4% | Forecast errors |
| Max Throughput | 112 | 90.4% | 9.5% | Crashes (ignores energy) |
| Random | -340 | 31.2% | 68% | Lower bound |

**Smart Heuristic**: Threshold-based (battery >60%→high capture, else reduce, ≤30%→sleep)

**DQN**: Value-based RL; overestimates Q-values in large action space

**MPC**: Assumes ideal energy forecasts; fails on real transient clouds

**Max Throughput**: Lesson—ignoring energy constraints = catastrophic system failure

## Evaluation

### Run Comprehensive Evaluation

```python
from src.evaluation.comprehensive import ComprehensiveEvaluation
from pathlib import Path

evaluator = ComprehensiveEvaluation(save_dir=Path("results/full_eval"), n_eval_episodes=50)
evaluator.compare_all_baselines()
evaluator.ablate_curriculum_stages()
evaluator.ablate_safety_wrapper()
evaluator.plot_policy_comparison()
evaluator.export_to_csv("results/metrics.csv")
```

### Key Metrics

| Metric | Meaning |
|--------|---------|
| **Episode Reward** | Cumulative return |
| **Delivery Rate** | % events successfully uploaded |
| **Completion Rate** | % episodes without battery failure |
| **Energy Efficiency** | Energy per delivered event |
| **Buffer Utilization** | Peak buffer occupancy (%) |

### Diagnostics

```bash
python scripts/train.py --mode diagnose
```

Validates: GHI data, energy model consistency, curriculum, observation/action dimensions, reward clipping, safety wrapper, GPU/CPU availability

### Computational Overhead

PPO decision: **0.087 ms** (negligible vs. 60-second epoch)

## Experimental Results

### Key Findings

1. **Unified > Fragmented**: PPO (804) beats Smart Heuristic (696) by jointly optimizing all stages

2. **Curriculum is Essential**: Direct training plateaus at ~200; with curriculum reaches 487+

3. **Safety Wrapper Trade-off**:
   - Unwrapped: 82.8% delivery, 36.5% completion (crashes)
   - Wrapped: 82.6% delivery, 100% completion
   - Cost: -0.2pp delivery for guaranteed safety ✅

4. **Transfer Learning**: 50× speedup (500 episodes → 50 episodes) when fine-tuning Jetson→Raspberry Pi

5. **Learned Energy Hierarchy**: 
   - High battery: high capture + complex inference + full TX
   - Medium: low capture + simple inference + results-only
   - Low: all off (deep sleep)
   - *No explicit rules coded—discovered autonomously*

### Hardware Comparison

| Platform | PPO Reward | Delivery | Completion |
|----------|-----------|----------|-----------|
| Jetson Nano | 804 | 82.6% | 100% |
| Raspberry Pi 3 | 22 | 71.8% | 100% |

Both achieve >70% delivery with 100% system reliability

## Hardware Platforms

### Jetson Nano (Primary)
- **CPU**: 4-core ARM Cortex-A57 @ 1.43 GHz
- **Memory**: 4 GB LPDDR4
- **Power**: 1745 mW idle
- **Energy Model** (measured with INA3221 @ 100 Hz):
  - Capture: 10.5 mJ/image (high-rate), standby 830.5 mW
  - Inference: 59.4 mJ (simple), 153.6 mJ (complex)
  - TX: 3,777 mJ handshake + 252.9 mJ/image

### Raspberry Pi 3 + ESP32 (Secondary)
- **Compute**: Raspberry Pi 3B+ (1 GB RAM)
- **Sensor**: ESP32-Cam (remote)
- **Uplink**: Cellular (GSM/LTE)
- **Battery**: 250,000 J (larger due to cellular cost)
- **Energy Model** (literature-derived):
  - Capture: 111.8 mJ/image
  - Inference: 1,476 mJ (simple), 2,448 mJ (complex)
  - TX: 56.52 J/image (cellular)

### Custom Hardware

Profile subsystem energy with power monitor (INA219/INA3221), then:

```python
from src.env.core import SystemParameters
params = SystemParameters(
    B_max=your_battery_capacity,
    e_cap_hr=your_high_rate_cost,
    # ... other measured values
)
```

Run `python scripts/train.py --mode diagnose` to validate

---

## Future Work

- **Online Curriculum Adaptation**: Auto-adjust difficulty based on agent performance
- **Hierarchical RL**: Separate energy allocation (high-level) from scheduling (low-level)
- **Federated Learning**: Distribute training across multiple edge devices
- **Meta-Learning**: Learn-to-learn for hardware transfer without fine-tuning
- **Hybrid Control**: Combine MPC (long-horizon) + PPO (reactive)
- **Real Hardware**: Deploy on Jetson + solar testbed with live solar traces
- **Multi-Agent**: Networked cameras with energy trading
- **Online Safety**: Continual on-device learning with safety guarantees
- **Inverse RL**: Infer deployment objectives from user preferences
- **Sim-to-Real**: Domain randomization for real-world robustness

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'src'` | Run from repo root; `src/` in PYTHONPATH |
| CUDA OOM | Reduce `batch_size`; CPU fully supported |
| Reward plateaus | Increase `entropy_coef` in early stages |
| Battery crashes | Enable `use_safety_wrapper=True` |
| Transfer underperforms | Verify target hardware energy params |

**Debug**: `import logging; logging.basicConfig(level=logging.DEBUG)`

---

## Citation

This work is currently under peer review. For now, please cite as a preprint:

```bibtex
@misc{ElkolallyEtAl2026EnergyHarvestingRL,
  title = {Towards Sustainable Edge Intelligence: 
           A Unified Reinforcement Learning Framework for 
           Energy-Harvesting Vision Systems},
  author = {Elkolally, Ayatollah and Dorgham, Anas and 
            Etman, Abdallah and Zewail, Rami and 
            Inoue, Koji and Sayed, Mohammed},
  year = {2026},
  howpublished = {\url{https://github.com/TODO}},
  note = {Under peer review}
}
```

Preprint and publication links will be added upon acceptance.

---

## License

MIT License – See [LICENSE](LICENSE) for details. Free for academic and commercial use with attribution.

---

## Acknowledgements

**Authors**:
- Ayatollah Elkolally (Egypt-Japan University of Science & Technology)
- Anas Dorgham, Abdallah Etman, Rami Zewail, Mohammed Sayed (EJUST)
- Koji Inoue (Kyushu University)

**Data & Resources**:
- Solar irradiance data: [NREL Solar Radiation Database (NSRDB)](https://nsrdb.nrel.gov/)
- Wildlife events: [Snapshot Serengeti](https://www.zooniverse.org/projects/snapshot-kenya/snapshot-serengeti)
- RL algorithms: [OpenAI Spinning Up](https://spinningup.openai.com/)

**Funding & Institutional Support**: 
Egypt-Japan University of Science & Technology, Department of Computer Science & Engineering and Department of Electronics & Communications Engineering.

---

<div align="center">

### ⭐ Found this useful? Please consider starring the repository!

For questions or issues, please [open a GitHub issue](https://github.com/TODO).

</div>
