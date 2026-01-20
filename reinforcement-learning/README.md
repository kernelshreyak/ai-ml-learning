## Reinforcement Learning

Collection of small reinforcement learning experiments and notebooks.

### Contents

- `multiarmed_bandit.py` - Epsilon-greedy multi-armed bandit simulation with plots.
- `ppo_cartpole.ipynb` - PPO training on CartPole.
- `panda_pick_cube.ipynb` - Robot manipulation (Panda) pick-and-place experiment.
- `panda_pick_cube.py` - Panda pick-and-place training/eval script (v2 improvements).
- `panda_reach_cube.py` - Reach-only training + scripted pick inference (v3).
- `Pandapicker_PPO_v1.md` - v1 design notes.
- `Pandapicker_PPO_v2.md` - v2 improvements (curriculum, shaping, gripper control).
- `Pandapicker_PPO_v3.md` - v3 reach-only + scripted pick pipeline.

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

### Panda Picker (overview)

The Panda picker evolved across three versions:

- **v1**: baseline PPO pick-and-place environment and training.
- **v2**: fixed cube size, curriculum on spawn/target ranges, improved reward shaping, and absolute gripper control.
- **v3**: reach-only RL training with scripted grasp in inference for faster, more reliable behavior on headless servers.

Use the versioned docs for details:

- `Pandapicker_PPO_v1.md`
- `Pandapicker_PPO_v2.md`
- `Pandapicker_PPO_v3.md`

Quick runs:

```bash
# v2: pick-and-place (train)
python panda_pick_cube.py --mode train --model ppo_panda_pick --timesteps 5000000 --episode-len 300

# v2: evaluate and save video
python panda_pick_cube.py --mode video --model ppo_panda_pick --video-dir ./rl_videos --episode-len 300

# v3: reach-only (train)
python panda_reach_cube.py --mode train --model ppo_panda_reach --timesteps 2000000 --episode-len 200

# v3: reach-only eval video
python panda_reach_cube.py --mode video --model ppo_panda_reach --video-dir ./rl_videos --episode-len 200

# v3: reach + scripted pick (headless video)
python panda_reach_cube.py --mode pick --model ppo_panda_reach --video-dir ./rl_videos --episode-len 200
```

Notebooks:

```bash
jupyter lab
```
