# PPO Training for Panda Pick-and-Place (v1)

This document explains **what Proximal Policy Optimization (PPO) is**, **how it is used in this repository**, and **how the algorithm maps to the code and environment implementation**.

This is **v1**, reflecting the current implementation. Future changes should be documented in separate markdown files.

---

## 1. Problem Overview

This repository trains a **Franka Panda robotic arm** in **PyBullet** to:

1. Reach a cube on a table
2. Grasp it using a parallel gripper
3. Move it to a target location on the table

The task is formulated as a **continuous-control reinforcement learning problem** and trained using **PPO (Proximal Policy Optimization)** from Stable-Baselines3.

---

## 2. Environment as a Markov Decision Process (MDP)

The task is modeled as an MDP defined by:

* **State** ( s_t )
* **Action** ( a_t )
* **Reward** ( r_t )
* **Termination condition**

---

### 2.1 State Space

The observation returned by the environment is a **19-dimensional continuous vector**:

| Component                | Dimension |
| ------------------------ | --------- |
| End-effector position    | 3         |
| End-effector velocity    | 3         |
| Cube position            | 3         |
| Cube linear velocity     | 3         |
| Target position          | 3         |
| Cube size (full extents) | 3         |
| Gripper opening          | 1         |
| **Total**                | **19**    |

This observation is constructed in the environment method `_get_obs()`.

---

### 2.2 Action Space

The action is a **4D continuous vector**:

```
a = [dx, dy, dz, g]
```

| Component  | Meaning                                          |
| ---------- | ------------------------------------------------ |
| dx, dy, dz | End-effector Cartesian delta (scaled internally) |
| g          | Gripper command (open / close)                   |

* Actions are normalized to `[-1, 1]`
* End-effector motion is converted to joint commands using inverse kinematics

---

### 2.3 Reward Function

The reward is **dense and shaped** to support learning.

Main components:

* Distance from end-effector to cube
* Distance from cube to target
* Height of the cube (lifting incentive)
* Binary success bonus

Key properties:

* Encourages reach → grasp → lift → place
* Penalizes inefficient or incorrect motion
* Includes a large terminal bonus for task success

Reward computation is implemented in `_compute_reward_done_info()`.

---

### 2.4 Episode Termination

An episode ends if:

* The cube reaches the target (`terminated`)
* Maximum episode length is reached (`truncated`)

Both signals are used to ensure correct return computation.

---

## 3. PPO: High-Level Algorithm

PPO is an **on-policy policy-gradient algorithm** that alternates between:

1. **Collecting rollouts** using the current policy
2. **Updating the policy and value function** using those rollouts

The objective is to maximize expected discounted reward:

[
\max_\pi ; \mathbb{E} \left[ \sum_{t=0}^T \gamma^t r_t \right]
]

---

## 4. PPO Architecture

The policy consists of:

* A **shared neural network backbone**
* A **policy head** producing a Gaussian action distribution
* A **value head** predicting the state value ( V(s) )

For continuous actions:

[
\pi(a|s) = \mathcal{N}(\mu(s), \sigma(s))
]

Both the mean and standard deviation are learned.

---

## 5. Rollout Collection (On-Policy)

Training uses vectorized environments:

* `SubprocVecEnv` with up to 32 CPU workers
* Each worker runs an independent PyBullet simulation

Key parameters:

* `n_steps = 1024`
* `n_envs ≈ 32`

Per iteration, the rollout size is approximately:

[
32 \times 1024 \approx 32{,}768 \text{ transitions}
]

All data is collected using the **same policy snapshot**.

---

## 6. Advantage Estimation (GAE)

PPO uses **Generalized Advantage Estimation (GAE)**:

[
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
]

[
A_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}
]

Defaults used by Stable-Baselines3:

* ( \gamma = 0.99 )
* ( \lambda = 0.95 )

Advantages are normalized before optimization.

---

## 7. PPO Clipping (Core Stability Mechanism)

### 7.1 Probability Ratio

PPO compares the new policy to the policy that collected the data:

[
r_t(\theta) =
\frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}
]

---

### 7.2 Clipped Objective

The PPO objective is:

[
L^{\text{CLIP}}(\theta)
=======================

\mathbb{E}
\left[
\min
\Big(
r_t(\theta) A_t,
\text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t
\Big)
\right]
]

Where:

* ( \epsilon = 0.2 ) (`clip_range`)

This prevents the policy from changing too much in a single update.

---

### 7.3 Interpretation

* Good actions cannot become **too much more likely**
* Bad actions cannot become **too much less likely**
* Once the update exceeds the clip range, **gradients are cut off**

This acts as a **soft trust region**, which is critical for stability in robotics tasks with contact dynamics.

---

## 8. Full PPO Loss

The total loss minimized during training is:

[
\mathcal{L}
===========

* L^{\text{CLIP}}

- c_v \cdot |V(s_t) - R_t|^2

* c_e \cdot H(\pi(\cdot|s_t))
  ]

Where:

* Policy loss improves action selection
* Value loss trains the critic
* Entropy bonus encourages exploration

These correspond to the logged metrics:

* `policy_gradient_loss`
* `value_loss`
* `entropy_loss`

---

## 9. Training Configuration (Current)

```python
PPO(
    n_steps=1024,
    batch_size=4096,
    n_epochs=10,
    learning_rate=3e-4,
    gamma=0.99,
    device="cpu",
)
```

Each rollout is reused for multiple epochs via minibatch SGD.

---

## 10. Why PPO is Suitable for This Task

PPO is a good fit because:

* Continuous action space (end-effector control)
* Stable updates under noisy contact dynamics
* Works well with dense reward shaping
* CPU-friendly with vectorized environments
* Robust to imperfect value estimates

---

## 11. Evaluation

Evaluation:

* Uses a **single environment**
* Runs the policy **deterministically**
* Records video using `RecordVideo`

This ensures clean and interpretable visualization of learned behavior.

---

## 12. Summary

This repository uses PPO to train a continuous-control policy for robotic manipulation by:

* Collecting large batches of on-policy experience
* Estimating advantages using a learned value function
* Updating the policy conservatively using a clipped objective
* Maintaining stability in a contact-rich physics environment

Future algorithmic or environment changes should be documented in additional versioned markdown files.
