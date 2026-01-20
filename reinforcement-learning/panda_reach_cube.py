import argparse
import math
import os
import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers import RecordVideo

import pybullet as p
import pybullet_data

import multiprocessing as mp

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor


class PandaReachCubeEnv(gym.Env):
    """
    Franka Panda end-effector control with IK.
    Task: reach above the cube (touch target), not grasp.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(
        self,
        render_mode=None,
        episode_len=200,
        sim_steps_per_action=8,
        fixed_cube_half_extent=0.04,
        target_xy_range=0.15,
        seed=None,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.episode_len = episode_len
        self.sim_steps_per_action = sim_steps_per_action
        self.fixed_cube_half_extent = fixed_cube_half_extent
        self.target_xy_range = target_xy_range

        self._rng = np.random.default_rng(seed)

        # Action: delta EE xyz + gripper command (gripper ignored during training)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        # Observation: ee_pos(3), ee_vel(3), cube_pos(3), cube_vel(3), cube_size(3)
        obs_dim = 3 + 3 + 3 + 3 + 3
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        self._client = None
        self._step_count = 0

        # IDs
        self.panda = None
        self.cube_id = None
        self.table_id = None

        # Panda joints
        self.arm_joint_indices = [0, 1, 2, 3, 4, 5, 6]
        self.finger_joint_indices = [9, 10]
        self.ee_link_index = 11  # panda_hand
        self.max_gripper_opening = 0.08

        # Internal control scaling
        self.ee_delta_scale = np.array([0.02, 0.02, 0.02], dtype=np.float32)
        self.ee_min = np.array([0.25, -0.3, 0.05], dtype=np.float32)
        self.ee_max = np.array([0.75, 0.3, 0.55], dtype=np.float32)

        # Target reach offset (above cube)
        self._reach_offset = np.array([0.0, 0.0, 0.08], dtype=np.float32)
        self._fixed_cube_pos = None

    def _connect(self):
        if self._client is not None:
            return
        if self.render_mode == "human":
            self._client = p.connect(p.GUI)
        else:
            self._client = p.connect(p.DIRECT)

        p.resetSimulation(physicsClientId=self._client)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self._client)
        p.setGravity(0, 0, -9.81, physicsClientId=self._client)
        p.setTimeStep(1.0 / 240.0, physicsClientId=self._client)

    def _load_scene(self):
        p.resetSimulation(physicsClientId=self._client)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self._client)
        p.setGravity(0, 0, -9.81, physicsClientId=self._client)
        p.setTimeStep(1.0 / 240.0, physicsClientId=self._client)

        p.loadURDF("plane.urdf", physicsClientId=self._client)

        table_urdf = os.path.join(pybullet_data.getDataPath(), "table/table.urdf")
        self.table_id = p.loadURDF(
            table_urdf,
            basePosition=[0.5, 0.0, -0.65],
            useFixedBase=True,
            physicsClientId=self._client,
        )

        self.panda = p.loadURDF(
            "franka_panda/panda.urdf",
            basePosition=[0.0, 0.0, 0.0],
            useFixedBase=True,
            physicsClientId=self._client,
        )

        for j in range(p.getNumJoints(self.panda, physicsClientId=self._client)):
            p.changeDynamics(self.panda, j, linearDamping=0.04, angularDamping=0.04, physicsClientId=self._client)

        home = [0.0, -0.6, 0.0, -2.2, 0.0, 1.6, 0.8]
        for idx, q in zip(self.arm_joint_indices, home):
            p.resetJointState(self.panda, idx, q, physicsClientId=self._client)

        self._set_gripper(opening=self.max_gripper_opening)

        hx = hy = hz = float(self.fixed_cube_half_extent)
        self.cube_half_extents = np.array([hx, hy, hz], dtype=np.float32)

        cube_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[hx, hy, hz], physicsClientId=self._client)
        cube_vis = p.createVisualShape(
            p.GEOM_BOX, halfExtents=[hx, hy, hz], rgbaColor=[0.8, 0.2, 0.2, 1.0], physicsClientId=self._client
        )

        if self._fixed_cube_pos is None:
            cube_x = float(self._rng.uniform(0.45, 0.65))
            cube_y = float(self._rng.uniform(-self.target_xy_range, self.target_xy_range))
            cube_z = 0.02
        else:
            cube_x, cube_y, cube_z = self._fixed_cube_pos
        self.cube_id = p.createMultiBody(
            baseMass=0.2,
            baseCollisionShapeIndex=cube_col,
            baseVisualShapeIndex=cube_vis,
            basePosition=[cube_x, cube_y, cube_z],
            baseOrientation=p.getQuaternionFromEuler([0, 0, float(self._rng.uniform(-math.pi, math.pi))]),
            physicsClientId=self._client,
        )
        p.changeDynamics(self.cube_id, -1, lateralFriction=1.0, spinningFriction=0.01, rollingFriction=0.01, physicsClientId=self._client)

    def _set_gripper(self, opening: float):
        opening = float(np.clip(opening, 0.0, self.max_gripper_opening))
        finger_pos = opening / 2.0
        for j in self.finger_joint_indices:
            p.setJointMotorControl2(
                self.panda,
                j,
                p.POSITION_CONTROL,
                targetPosition=finger_pos,
                force=80,
                physicsClientId=self._client,
            )

    def _get_ee_state(self):
        link = p.getLinkState(self.panda, self.ee_link_index, computeLinkVelocity=1, physicsClientId=self._client)
        pos = np.array(link[4], dtype=np.float32)
        vel = np.array(link[6], dtype=np.float32)
        return pos, vel

    def _ik_to(self, ee_pos, ee_orn=None):
        if ee_orn is None:
            ee_orn = p.getQuaternionFromEuler([math.pi, 0, 0])
        joint_poses = p.calculateInverseKinematics(
            self.panda,
            self.ee_link_index,
            ee_pos.tolist(),
            ee_orn,
            maxNumIterations=50,
            residualThreshold=1e-4,
            physicsClientId=self._client,
        )
        return joint_poses

    def _apply_action(self, action):
        action = np.asarray(action, dtype=np.float32)
        dxyz = action[:3] * self.ee_delta_scale

        ee_pos, _ = self._get_ee_state()
        target_ee = np.clip(ee_pos + dxyz, self.ee_min, self.ee_max)

        joint_poses = self._ik_to(target_ee)
        for i, j in enumerate(self.arm_joint_indices):
            p.setJointMotorControl2(
                self.panda,
                j,
                p.POSITION_CONTROL,
                targetPosition=float(joint_poses[j]),
                force=200,
                physicsClientId=self._client,
            )

    def _get_obs(self):
        ee_pos, ee_vel = self._get_ee_state()
        cube_pos, _ = p.getBasePositionAndOrientation(self.cube_id, physicsClientId=self._client)
        cube_vel_lin, _ = p.getBaseVelocity(self.cube_id, physicsClientId=self._client)

        cube_pos = np.array(cube_pos, dtype=np.float32)
        cube_vel = np.array(cube_vel_lin, dtype=np.float32)
        cube_size = (2.0 * self.cube_half_extents).astype(np.float32)

        obs = np.concatenate([ee_pos, ee_vel, cube_pos, cube_vel, cube_size], axis=0)
        return obs

    def _compute_reward_done_info(self):
        ee_pos, _ = self._get_ee_state()
        cube_pos, _ = p.getBasePositionAndOrientation(self.cube_id, physicsClientId=self._client)
        cube_pos = np.array(cube_pos, dtype=np.float32)

        target = cube_pos + self._reach_offset
        d_ee_target = float(np.linalg.norm(ee_pos - target))

        reward = -1.0 * d_ee_target + 0.5 * math.exp(-10.0 * d_ee_target)

        success = d_ee_target < 0.03
        if success:
            reward += 5.0

        terminated = bool(success)
        truncated = bool(self._step_count >= self.episode_len)

        info = {"success": success, "d_ee_target": d_ee_target}
        return reward, terminated, truncated, info

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        if options and "cube_pos" in options:
            pos = options["cube_pos"]
            if len(pos) != 3:
                raise ValueError("cube_pos must be length 3")
            self._fixed_cube_pos = (float(pos[0]), float(pos[1]), float(pos[2]))
        else:
            self._fixed_cube_pos = None
        self._connect()
        self._load_scene()
        self._step_count = 0

        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action):
        self._apply_action(action)
        for _ in range(self.sim_steps_per_action):
            p.stepSimulation(physicsClientId=self._client)
            if self.render_mode == "human":
                time.sleep(1.0 / 240.0)

        self._step_count += 1
        obs = self._get_obs()
        reward, terminated, truncated, info = self._compute_reward_done_info()
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        view = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[0.55, 0.0, 0.1],
            distance=0.8,
            yaw=45,
            pitch=-35,
            roll=0,
            upAxisIndex=2,
            physicsClientId=self._client,
        )
        proj = p.computeProjectionMatrixFOV(fov=60, aspect=16 / 9, nearVal=0.01, farVal=3.0)
        w, h, rgba, _, _ = p.getCameraImage(960, 540, viewMatrix=view, projectionMatrix=proj, physicsClientId=self._client)
        img = np.array(rgba, dtype=np.uint8).reshape(h, w, 4)
        return img

    def close(self):
        if self._client is not None:
            p.disconnect(physicsClientId=self._client)
            self._client = None

    # Scripted grasp for inference
    def scripted_pick(self, hover_height=0.08, approach_height=0.02, lift_height=0.12, steps=60):
        cube_pos, _ = p.getBasePositionAndOrientation(self.cube_id, physicsClientId=self._client)
        cube_pos = np.array(cube_pos, dtype=np.float32)

        def move_to(pos, n):
            joint_poses = self._ik_to(pos)
            for _ in range(n):
                for i, j in enumerate(self.arm_joint_indices):
                    p.setJointMotorControl2(
                        self.panda,
                        j,
                        p.POSITION_CONTROL,
                        targetPosition=float(joint_poses[j]),
                        force=200,
                        physicsClientId=self._client,
                    )
                for _ in range(self.sim_steps_per_action):
                    p.stepSimulation(physicsClientId=self._client)
                    if self.render_mode == "human":
                        time.sleep(1.0 / 240.0)

        # Open gripper, hover, descend, close, lift
        self._set_gripper(self.max_gripper_opening)
        move_to(cube_pos + np.array([0.0, 0.0, hover_height], dtype=np.float32), steps)
        move_to(cube_pos + np.array([0.0, 0.0, approach_height], dtype=np.float32), steps)
        self._set_gripper(0.0)
        move_to(cube_pos + np.array([0.0, 0.0, lift_height], dtype=np.float32), steps)


def make_env(rank: int, seed: int = 0, episode_len: int = 200):
    def _init():
        env = PandaReachCubeEnv(render_mode=None, seed=seed + rank, episode_len=episode_len)
        return env
    return _init


def train_model(total_timesteps: int, model_path: str, episode_len: int):
    mp.set_start_method("forkserver", force=True)
    n_envs = min(32, mp.cpu_count())

    env = SubprocVecEnv([make_env(i, episode_len=episode_len) for i in range(n_envs)])
    env = VecMonitor(env)

    model = PPO(
        "MlpPolicy",
        env,
        device="cpu",
        verbose=1,
        n_steps=1024,
        batch_size=3072,
        n_epochs=10,
        learning_rate=3e-4,
        gamma=0.99,
    )

    model.learn(total_timesteps=total_timesteps)
    model.save(model_path)
    env.close()


def eval_video(model_path: str, out_dir: str, episode_len: int, cube_pos):
    env = PandaReachCubeEnv(render_mode="rgb_array", episode_len=episode_len)
    env = RecordVideo(
        env,
        video_folder=out_dir,
        episode_trigger=lambda ep: ep == 0,
        name_prefix="panda-reach",
        disable_logger=False,
    )

    model = PPO.load(model_path, device="cpu")

    options = {"cube_pos": cube_pos} if cube_pos is not None else None
    obs, _ = env.reset(options=options)
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    env.close()
    print(f"Saved video(s) to {out_dir}")


def eval_scripted_pick(model_path: str, episode_len: int, gui: bool, out_dir: str, cube_pos):
    render_mode = "human" if gui else "rgb_array"
    base_env = PandaReachCubeEnv(render_mode=render_mode, episode_len=episode_len)
    env = base_env
    if not gui:
        env = RecordVideo(
            base_env,
            video_folder=out_dir,
            episode_trigger=lambda ep: ep == 0,
            name_prefix="panda-reach-pick",
            disable_logger=False,
        )
    model = PPO.load(model_path, device="cpu")

    options = {"cube_pos": cube_pos} if cube_pos is not None else None
    obs, _ = env.reset(options=options)
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            break

    # scripted grasp after reach success / timeout
    base_env.scripted_pick()

    if gui:
        try:
            while True:
                p.stepSimulation(physicsClientId=env._client)
                time.sleep(1.0 / 240.0)
        except KeyboardInterrupt:
            pass

    env.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Panda reach cube: train or run inference.")
    parser.add_argument("--mode", choices=["train", "video", "pick"], default="video")
    parser.add_argument("--model", default="ppo_panda_reach")
    parser.add_argument("--timesteps", type=int, default=2_000_000)
    parser.add_argument("--video-dir", default="./rl_videos")
    parser.add_argument("--episode-len", type=int, default=200, help="Max steps per episode.")
    parser.add_argument("--gui", action="store_true", help="Pick mode: open PyBullet GUI.")
    parser.add_argument(
        "--cube-pos",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        help="Fixed cube position for eval (e.g., 0.55 0.0 0.02)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.mode == "train":
        train_model(args.timesteps, args.model, args.episode_len)
    elif args.mode == "video":
        eval_video(args.model, args.video_dir, args.episode_len, args.cube_pos)
    elif args.mode == "pick":
        eval_scripted_pick(args.model, args.episode_len, args.gui, args.video_dir, args.cube_pos)
