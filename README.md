# F1TENTH-RL

A reinforcement learning framework for autonomous racing using the F1TENTH simulator. This project includes PPO and SAC training pipelines, multi-agent evaluation, and Pure Pursuit baselines.

---

## Setup Instructions

### 1. Install `pyenv` (if not already installed)

```bash
brew update
brew install pyenv
```

### 2. Install and activate Python 3.8.12 using `pyenv`
This project is tested with Python 3.8.12, but other versions of Python 3.8 or 3.9 should also work.

```bash
pyenv install 3.8.12
pyenv local 3.8.12
```

> **Note:** If `python --version` does not show `3.8.12`, run:
```bash
pyenv init
exec "$SHELL"
```

### 3. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate
```

---

## Install Dependencies

From the project root:

```bash
cd f1tenth_gym
pip install -e .
```

> This installs the simulator and custom wrappers defined in `setup.py`.

---

## Directory Structure

```
F1TENTH-RL/
│
├── f1tenth_gym/
│   ├── algorithms/               ← PPO, SAC training & eval scripts
│   ├── wrappers/                 ← Custom Gym wrappers (PPO, SAC, multi-agent)
│   └── f1tenth_racetracks/
│       └── example/              ← Pure Pursuit baseline (waypoint_follow.py)
```

---

## Training and Evaluation

### PPO

```bash
cd algorithms
python train_ppo.py        # Train PPO
python eval_ppo.py         # Evaluate PPO
```

### SAC

```bash
python train_sac.py        # Train SAC
python eval_sac.py         # Evaluate SAC
```

---

## Multi-Agent Simulation

Train outer-line multi-agent variants:

```bash
python train_outer_line.py PPO
python train_outer_line.py SAC
```

Evaluate multi-agent scenarios:

```bash
python run_multi_agent.py --model-type PPO
python run_multi_agent.py --model-type SAC
```

---

## Modifying Reward Constants

In the training files (`train_ppo.py`, `train_sac.py`), modify these constants to tune behavior:

```python
MAX_SPEED = 8.0
CRASH_PENALTY = -10.0
ANGLE_PENALTY = -1.0
ALIVE_BONUS = 1.0
LAP_BONUS = 2.0
DISTANCE_REWARD_COEF = 500
STEERING_CONSISTENCY_COEF = -3.0
```

For **speed reward**, go to the wrapper file:
- `wrappers/f110_wrapper_ppo.py`
- `wrappers/f110_wrapper_sac.py`
- `wrappers/multi_agent_wrapper.py`

Find the line:
```python
speed_reward = 0.1 * self.obs['linear_vels_x'][0]
```
...and modify `0.1` as needed.

---

## Pure Pursuit Baseline

To run a Pure Pursuit vs Pure Pursuit simulation:

```bash
cd f1tenth_racetracks/example
python waypoint_follow.py
```