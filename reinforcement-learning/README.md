## Reinforcement Learning

Collection of small reinforcement learning experiments and notebooks.

### Contents

- `multiarmed_bandit.py` - Epsilon-greedy multi-armed bandit simulation with plots.
- `ppo_cartpole.ipynb` - PPO training on CartPole.
- `panda_pick_cube.ipynb` - Robot manipulation (Panda) pick-and-place experiment.

### Setup

Use the repo-level virtual environment and dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r ../requirements.txt
```

### Run

Bandit simulation:

```bash
python multiarmed_bandit.py
```

Notebooks:

```bash
jupyter lab
```
