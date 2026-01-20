# PPO Training for Panda Pick-and-Place (v2)

This document summarizes **changes introduced after v1** to improve learning stability and grasp success. v1 details are in `Pandapicker_PPO_v1.md`.

---

## 1. What Changed vs v1

### 1.1 Environment Simplification (Curriculum + Fixed Size)

**Goal:** make the early learning phase easier while keeping cube position variable.

Changes:
- **Fixed cube size by default** (no random size range). This removes size variability while the policy learns grasping.
- **Curriculum on cube spawn and target range**:
  - Early episodes sample cube positions from a **narrow range**.
  - Over the first ~2000 episodes, the spawn range **expands linearly** to the full range.
  - Target Y-range also grows from small to the configured maximum.

These changes keep cube position **random but easier initially**.

---

### 1.2 Reward Shaping Improvements

The reward now includes additional guidance for grasping behavior:

- **Proximity bonus**: `exp(-10 * d_ee_cube)` encourages precise approach.
- **Lift bonus**: proportional to cube height above the table.
- **Gripper usage shaping**:
  - **Reward** closing when very close to the cube.
  - **Penalize** staying closed while far away.
  - **Penalize** staying open when extremely close.

The existing distance penalties and terminal success bonus remain.

---

### 1.3 Gripper Control Simplification

v1 used **incremental gripper opening**, which often left the gripper in an intermediate state.

v2 maps the gripper action **directly to an absolute opening**:

- `g = -1` → fully closed
- `g = +1` → fully open

This makes gripper behavior more learnable and stable.

---

### 1.4 Training Defaults

To give the policy enough time to learn a reliable grasp:

- **Default timesteps increased**: 2M → **5M**
- **Default episode length increased**: 200 → **300**

You can still override via CLI if needed.

---

### 1.5 Evaluation / CLI

- **Video eval stays** as the primary inference mode.
- **Cube position can be fixed via CLI** for evaluation only:

```
--cube-pos X Y Z
```

If not provided, cube position remains random.

---

## 2. Expected Effects

With these changes, the policy should:

- Learn to **open and close the gripper more reliably**
- Avoid prematurely closing far from the cube
- Learn the **reach → grasp → lift** sequence with fewer failures
- Achieve a **usable baseline** within ~3–5M timesteps

---

## 3. Notes

If training still stalls after ~3M steps, the next levers to tune are:

- `ee_delta_scale` (smaller = finer control)
- `sim_steps_per_action` (larger = smoother, slower motion)
- episode length (more time per attempt)

---

## 4. Files Updated

- `reinforcement-learning/panda_pick_cube.py`
- `reinforcement-learning/Pandapicker_PPO_v2.md`

---

End of v2.
