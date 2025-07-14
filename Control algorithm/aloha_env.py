import numpy as np
import mujoco
from mujoco import viewer
import gym
from gym import spaces

class AlohaEnv(gym.Env):
    def __init__(self, xml_path):
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.viewer = None

        # Define action and observation spaces
        self.action_space = spaces.Box(
            low=self.model.actuator_ctrlrange[:, 0],
            high=self.model.actuator_ctrlrange[:, 1],
            dtype=np.float32
        )
        obs_dim = self.model.nq + self.model.nv
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Target position (example)
        self.target_pos = np.array([0.5, -0.3, 0.84])

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs()

    def step(self, action):
        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        reward, dist = self._compute_reward()
        done = False
        info = {'dist': dist}
        return obs, reward, done, info

    def render(self, mode="human"):
        if self.viewer is None:
            self.viewer = viewer.launch_passive(self.model, self.data)
        self.viewer.sync()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def _get_obs(self):
        return np.concatenate([self.data.qpos, self.data.qvel])

    def _compute_reward(self):
        # Example of site-based reward calculation (modify per your XML)
        if self.model.nsite > 0:
            ee_pos = self.data.site_xpos[0]
        else:
            # Fallback: use body position from your earlier code
            ee_pos = self.data.body("fr_link8").xpos
        dist = np.linalg.norm(ee_pos - self.target_pos)
        reward = -dist
        return reward, dist