from env.walker_env import Walker2DEnv
import numpy as np

env = Walker2DEnv()
obs, info = env.reset()

print("Observation shape:", obs.shape)  # should be (22,)
print("First few values:", obs[:5])

for _ in range(10):
    action = env.action_space.sample()  # random action
    obs, reward, terminated, truncated, info = env.step(action)

    print(f"reward={reward:.3f}  height={obs[0]:.3f}")

    if terminated:
        break

env.close()
print("Environment test passed.")