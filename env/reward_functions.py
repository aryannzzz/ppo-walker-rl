"""
Reward functions for the Walker2D project.

Each function has the signature:
    reward_fn(obs: np.ndarray, action: np.ndarray, info: dict) -> float

Observation index reference
---------------------------
obs[0]    torso height (m)
obs[1]    torso pitch (rad)
obs[2:8]  joint angles (rad)
obs[8:14] joint velocities (rad/s)
obs[14]   forward velocity vx (m/s)
obs[15]   lateral velocity vy (m/s)
obs[16]   vertical velocity vz (m/s)
obs[17]   pitch rate (rad/s)
obs[18]   left foot contact (binary)
obs[19]   right foot contact (binary)
obs[20]   left thigh angle (rad)
obs[21]   right thigh angle (rad)
"""

import numpy as np


# -------------------------------------------------------------------
# Reward Function 1: Sparse Reward
# -------------------------------------------------------------------
def sparse_reward(obs: np.ndarray, action: np.ndarray, info: dict) -> float:
    """
    r_t = +1  if the robot moved forward by more than 0.01 m this step
          -1  if the robot fell (height < 0.55 m)
           0  otherwise

    This reward gives almost no learning signal because the agent needs to
    discover walking behaviour by chance before it ever receives +1.
    It illustrates the exploration problem in RL.
    """
    height = obs[0]
    vx = obs[14]

    if height < 0.55:
        return -1.0
    if vx * (1.0 / 60.0) > 0.01:   # distance moved this step > 0.01 m
        return 1.0
    return 0.0


# -------------------------------------------------------------------
# Reward Function 2: Dense Reward (default)
# -------------------------------------------------------------------
def dense_reward(obs: np.ndarray, action: np.ndarray, info: dict) -> float:
    """
    r_t = c1 * clip(vx, 0, 5)           forward velocity reward
        + c2                             alive bonus
        - c3 * ||a||_2^2                 energy penalty
        - c4 * |vy|                      lateral penalty
        + c5 * max(0, 1 - |h - h*|)     height bonus

    Default coefficients: c1=1.0, c2=1.0, c3=0.001, c4=0.5, c5=0.1, h*=1.2

    This is the primary reward function for the project.
    Each term has a specific purpose:

    - Forward velocity:  the main objective.  Clipped at 5 m/s so the agent
      cannot gain infinite reward by moving at unrealistic speeds.
    - Alive bonus:  makes staying upright worth +1 every step.  Without this,
      a struggling agent might prefer to fall quickly.
    - Energy penalty:  small penalty on large torques.  This discourages jerky,
      high-effort movements and produces a smoother gait.
    - Lateral penalty:  the agent should walk straight, not sideways.
    - Height bonus:  softly encourages maintaining upright posture near 1.2 m.
    """
    c1, c2, c3, c4, c5 = 1.0, 1.0, 0.001, 0.5, 0.1
    h_star = 1.2

    forward_reward = c1 * float(np.clip(obs[14], 0.0, 5.0))
    alive_bonus = c2
    energy_penalty = c3 * float(np.sum(np.square(action)))
    lateral_penalty = c4 * abs(float(obs[15]))
    height_error = abs(float(obs[0]) - h_star)
    height_bonus = c5 * max(0.0, 1.0 - height_error)

    return forward_reward + alive_bonus - energy_penalty - lateral_penalty + height_bonus


# -------------------------------------------------------------------
# Reward Function 3: Velocity Only
# -------------------------------------------------------------------
def velocity_only_reward(obs: np.ndarray, action: np.ndarray, info: dict) -> float:
    """
    r_t = clip(vx, 0, 5)  +  alive_bonus

    Same as dense_reward but without the energy penalty (c3 = 0).

    Expected behaviour: the agent learns to walk fast but the gait is more
    aggressive and energetically wasteful.  Comparing videos from this reward
    and dense_reward shows what the energy penalty was achieving.
    """
    forward_reward = float(np.clip(obs[14], 0.0, 5.0))
    alive_bonus = 1.0
    lateral_penalty = 0.5 * abs(float(obs[15]))
    height_error = abs(float(obs[0]) - 1.2)
    height_bonus = 0.1 * max(0.0, 1.0 - height_error)
    return forward_reward + alive_bonus - lateral_penalty + height_bonus


# -------------------------------------------------------------------
# Reward Function 4: Heavy Energy Penalty
# -------------------------------------------------------------------
def heavy_energy_reward(obs: np.ndarray, action: np.ndarray, info: dict) -> float:
    """
    Dense reward with c3 = 0.01  (10x the default energy penalty).

    Expected behaviour: one of two outcomes:
    1. A very slow, conservative walking gait that minimises joint effort.
    2. Policy collapse: the agent learns to stand still because the energy
       penalty outweighs any velocity gain.

    This demonstrates reward misspecification: an otherwise correct reward
    function with a badly calibrated coefficient produces unintended behaviour.
    """
    c1, c2, c3, c4, c5 = 1.0, 1.0, 0.01, 0.5, 0.1   # c3 is 10x larger
    h_star = 1.2

    forward_reward = c1 * float(np.clip(obs[14], 0.0, 5.0))
    alive_bonus = c2
    energy_penalty = c3 * float(np.sum(np.square(action)))
    lateral_penalty = c4 * abs(float(obs[15]))
    height_error = abs(float(obs[0]) - h_star)
    height_bonus = c5 * max(0.0, 1.0 - height_error)

    return forward_reward + alive_bonus - energy_penalty - lateral_penalty + height_bonus


# -------------------------------------------------------------------
# Registry: add your own reward functions here by name
# -------------------------------------------------------------------
REWARD_REGISTRY = {
    "sparse": sparse_reward,
    "dense": dense_reward,
    "velocity_only": velocity_only_reward,
    "heavy_energy": heavy_energy_reward,
}


def get_reward_fn(name: str):
    """Return a reward function by its registry name."""
    if name not in REWARD_REGISTRY:
        raise ValueError(
            f"Unknown reward function '{name}'. "
            f"Available: {list(REWARD_REGISTRY.keys())}"
        )
    return REWARD_REGISTRY[name]
