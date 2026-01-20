# PPO Training for Panda Pick-and-Place (v3)

This document summarizes **changes introduced after v2**. v2 details are in `Pandapicker_PPO_v2.md`.

---

## 1. What Changed vs v2

### 1.1 New Reach-Only RL Pipeline

A separate script trains **only the reaching behavior**, leaving grasping to scripted control:

- New file: `reinforcement-learning/panda_reach_cube.py`
- RL objective: move the end-effector to a **target above the cube**
- Gripper is **not part of the learning task**

This makes the RL problem easier and achievable in ~2M steps.

---

### 1.2 Scripted Grasping in Inference

Inference now supports a **reach + scripted pick** flow:

- `--mode pick` runs RL until the reach target is met
- Then scripted steps execute: **open → descend → close → lift**

This yields more reliable picking without requiring the policy to learn fine-grained gripper timing.

---

### 1.3 Headless Video Support for Scripted Pick

For server environments without GUI:

- `pick` mode records video in headless mode using `RecordVideo`
- Output stored in `--video-dir`

---

### 1.4 CLI Cube Position Override (Reach Script)

The reach script now supports fixed cube placement during evaluation:

```
--cube-pos X Y Z
```

If omitted, cube position remains randomized.

---

### 1.5 PPO Batch Size Fix

To avoid truncated minibatches during training:

- `batch_size` changed to **3072**, which evenly divides the rollout size (30720)

This removes the Stable-Baselines warning and slightly improves training consistency.

---

## 2. Expected Effects

- Faster convergence for reach-only training
- More reliable grasp success via scripted pick
- Headless inference with video output
- Cleaner PPO training batches

---

## 3. Files Updated / Added

- `reinforcement-learning/panda_reach_cube.py` (new)
- `reinforcement-learning/panda_pick_cube.py` (batch size fix)
- `reinforcement-learning/Pandapicker_PPO_v3.md`

---

End of v3.
