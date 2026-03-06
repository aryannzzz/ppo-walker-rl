"""
Walker2D custom Gymnasium environment using PyBullet.
The robot is a 2D bipedal walker with 6 revolute joints.

Observation space: 22-dimensional vector
    [0]    torso height (m)
    [1]    torso pitch (rad)
    [2:8]  joint angles (rad), 6 joints
    [8:14] joint velocities (rad/s), 6 joints
    [14]   forward velocity vx (m/s)
    [15]   lateral velocity vy (m/s)
    [16]   vertical velocity vz (m/s)
    [17]   pitch rate (rad/s)
    [18]   left foot contact (binary)
    [19]   right foot contact (binary)
    [20]   left thigh angle (rad)
    [21]   right thigh angle (rad)

Action space: 6-dimensional vector in [-1, 1]
    Continuous joint torques for each of the 6 joints.

Episode ends when:
    - torso height < 0.5 m  (robot fell)
    - |torso pitch| > 1.2 rad  (robot tipped)
    - 1000 steps elapsed
"""

import os
import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces

try:
    import pybullet as p
    import pybullet_data
    PYBULLET_AVAILABLE = True
except ImportError:
    PYBULLET_AVAILABLE = False

from env.reward_functions import dense_reward


WALKER_URDF = os.path.join(
    os.path.dirname(__file__), "assets", "walker2d.urdf"
)


class Walker2DEnv(gym.Env):
    """
    PyBullet bipedal walker environment with a Gymnasium API.

    Parameters
    ----------
    reward_fn : callable, optional
        A function with signature (obs, action, info) -> float.
        Defaults to the dense reward function.
    render_mode : str, optional
        "human" for GUI window, "rgb_array" for pixel output,
        or None (default) for headless (fastest) mode.
    max_episode_steps : int
        Maximum number of steps per episode. Default 1000.
    """

    metadata = {"render_modes": ["human", "rgb_array"]}

    OBS_DIM = 22
    ACT_DIM = 6
    MAX_HEIGHT_FALL = 0.5      # m  -- episode ends below this
    MAX_PITCH = 1.2            # rad -- episode ends above this
    SIM_HZ = 240               # physics steps per second
    CTRL_HZ = 60               # agent control steps per second
    SIM_PER_CTRL = SIM_HZ // CTRL_HZ   # = 4
    TORQUE_SCALE = 100.0       # scale action [-1,1] to [-100, 100] Nm

    def __init__(
        self,
        reward_fn=None,
        render_mode=None,
        max_episode_steps=1000,
    ):
        if not PYBULLET_AVAILABLE:
            raise ImportError(
                "pybullet is not installed. "
                "Run:  pip install pybullet"
            )

        super().__init__()
        self.reward_fn = reward_fn if reward_fn is not None else dense_reward
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self._step_count = 0
        self._physics_client = None

        # Observation space: all floats, generous bounds
        obs_high = np.full(self.OBS_DIM, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(
            -obs_high, obs_high, dtype=np.float32
        )

        # Action space: torques scaled to [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.ACT_DIM,),
            dtype=np.float32,
        )

        self._setup_physics()

    # -----------------------------------------------------------------
    # Public Gymnasium API
    # -----------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0
        self._load_robot()
        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        torques = action * self.TORQUE_SCALE

        # Apply torques and step simulation multiple times
        for joint_idx, torque in enumerate(torques):
            p.setJointMotorControl2(
                self._robot,
                joint_idx,
                p.TORQUE_CONTROL,
                force=torque,
                physicsClientId=self._physics_client,
            )
        for _ in range(self.SIM_PER_CTRL):
            p.stepSimulation(physicsClientId=self._physics_client)

        obs = self._get_obs()
        info = self._get_info(obs)
        reward = float(self.reward_fn(obs, action, info))

        self._step_count += 1
        terminated = self._is_terminal(obs)
        truncated = self._step_count >= self.max_episode_steps

        if self.render_mode == "human":
            p.resetDebugVisualizerCamera(
                cameraDistance=3.0,
                cameraYaw=0,
                cameraPitch=-20,
                cameraTargetPosition=info["torso_pos"],
                physicsClientId=self._physics_client,
            )

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            w, h = 640, 480
            torso_pos, _ = p.getBasePositionAndOrientation(
                self._robot, physicsClientId=self._physics_client
            )
            view = p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=torso_pos,
                distance=3.0,
                yaw=0,
                pitch=-20,
                roll=0,
                upAxisIndex=2,
            )
            proj = p.computeProjectionMatrixFOV(
                fov=60, aspect=w / h, nearVal=0.1, farVal=100
            )
            _, _, px, _, _ = p.getCameraImage(
                w, h, view, proj,
                physicsClientId=self._physics_client
            )
            return np.array(px, dtype=np.uint8).reshape(h, w, 4)[:, :, :3]
        return None

    def close(self):
        if self._physics_client is not None:
            try:
                p.disconnect(physicsClientId=self._physics_client)
            except Exception:
                pass
            self._physics_client = None

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    def _setup_physics(self):
        """Connect to a PyBullet physics server."""
        if self.render_mode == "human":
            self._physics_client = p.connect(p.GUI)
            p.resetDebugVisualizerCamera(3.0, 0, -20, [0, 0, 1])
        else:
            self._physics_client = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(
            pybullet_data.getDataPath(),
            physicsClientId=self._physics_client,
        )
        p.setGravity(0, 0, -9.81, physicsClientId=self._physics_client)
        p.setTimeStep(
            1.0 / self.SIM_HZ, physicsClientId=self._physics_client
        )
        self._plane = p.loadURDF(
            "plane.urdf", physicsClientId=self._physics_client
        )
        self._robot = None

    def _load_robot(self):
        """Remove and reload the walker robot at its start pose."""
        if self._robot is not None:
            p.removeBody(self._robot, physicsClientId=self._physics_client)

        start_pos = [0, 0, 1.4]
        start_orn = p.getQuaternionFromEuler([0, 0, 0])

        # Use pybullet_data's walker2d if our local asset is not present
        urdf_path = WALKER_URDF if os.path.exists(WALKER_URDF) else "walker2d.urdf"
        self._robot = p.loadURDF(
            urdf_path,
            start_pos,
            start_orn,
            physicsClientId=self._physics_client,
        )

        # Enable torque control for all joints
        n_joints = p.getNumJoints(
            self._robot, physicsClientId=self._physics_client
        )
        for j in range(n_joints):
            p.setJointMotorControl2(
                self._robot,
                j,
                p.VELOCITY_CONTROL,
                force=0,
                physicsClientId=self._physics_client,
            )

    def _get_obs(self):
        """Build the 22-dim observation vector from PyBullet state."""
        pos, orn = p.getBasePositionAndOrientation(
            self._robot, physicsClientId=self._physics_client
        )
        euler = p.getEulerFromQuaternion(orn)
        lin_vel, ang_vel = p.getBaseVelocity(
            self._robot, physicsClientId=self._physics_client
        )

        joint_states = p.getJointStates(
            self._robot,
            range(self.ACT_DIM),
            physicsClientId=self._physics_client,
        )
        joint_angles = np.array([js[0] for js in joint_states], dtype=np.float32)
        joint_vels = np.array([js[1] for js in joint_states], dtype=np.float32)

        # Foot contacts: check if foot links are in contact with plane
        contacts_left = int(
            len(p.getContactPoints(
                self._robot, self._plane, 4, -1,
                physicsClientId=self._physics_client
            )) > 0
        )
        contacts_right = int(
            len(p.getContactPoints(
                self._robot, self._plane, 7, -1,
                physicsClientId=self._physics_client
            )) > 0
        )

        obs = np.concatenate([
            [pos[2]],              # [0]  torso height
            [euler[1]],            # [1]  torso pitch
            joint_angles,          # [2:8]
            joint_vels,            # [8:14]
            [lin_vel[0]],          # [14] forward velocity
            [lin_vel[1]],          # [15] lateral velocity
            [lin_vel[2]],          # [16] vertical velocity
            [ang_vel[1]],          # [17] pitch rate
            [contacts_left],       # [18]
            [contacts_right],      # [19]
            [joint_angles[0]],     # [20] left thigh (repeated for richer signal)
            [joint_angles[3]],     # [21] right thigh
        ]).astype(np.float32)

        return obs

    def _get_info(self, obs):
        pos, _ = p.getBasePositionAndOrientation(
            self._robot, physicsClientId=self._physics_client
        )
        return {
            "torso_pos": list(pos),
            "torso_height": float(obs[0]),
            "torso_pitch": float(obs[1]),
            "forward_velocity": float(obs[14]),
        }

    def _is_terminal(self, obs):
        height = obs[0]
        pitch = obs[1]
        return bool(height < self.MAX_HEIGHT_FALL or abs(pitch) > self.MAX_PITCH)
