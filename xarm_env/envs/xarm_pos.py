import numpy as np
import time

import gymnasium as gym
from gymnasium import spaces

from xarm.wrapper import XArmAPI

GRIPPER_OPEN = 800
GRIPPER_CLOSED = 0
# GRIPPER_THRESH = 800

def pulse_to_g(pulse):
    g = float((pulse - GRIPPER_OPEN) / (GRIPPER_CLOSED - GRIPPER_OPEN))

    g_clamped = min(max(g, 0.0), 1.0)
    return g_clamped

def g_to_pulse(g):
    g_clamped = min(max(g, 0.0), 1.0)
    pulse = GRIPPER_OPEN + g_clamped * (GRIPPER_CLOSED - GRIPPER_OPEN)
    return pulse

class XarmPosEnv(gym.Env):
    def __init__(self, env_data, render_mode=None):
        api = env_data["api"]

        # Setup arm
        self._is_connected = False
        self._setup_arm(api)

        # Setup gym
        self.observation_space = spaces.Dict({
            "ee_pose": spaces.Box(low=-np.inf,
                                      high=np.inf,
                                      shape=(6,),
                                      dtype=float),
            # "ee_gripper": spaces.Discrete(2),
            "ee_gripper": spaces.Box(low=-np.inf, high=np.inf, dtype=float),
        })
        
        self.action_space = spaces.Dict({
            "ee_pose": spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=float),
            # "gripper": spaces.Discrete(2),
            "ee_gripper": spaces.Box(low=-np.inf, high=np.inf, dtype=float)
        })

        assert render_mode is None

    def _setup_arm(self, api):
        self.api = api
        self._is_connected = False

        # Instantiate API
        self.arm = XArmAPI(api)
        time.sleep(0.5)

        self._is_connected = True

        self.arm.clean_error()
        self.arm.clean_warn()
        self.arm.motion_enable(True)
        time.sleep(1)

        # Set mode and state
        self.arm.set_mode(0)
        time.sleep(1)
        self.arm.set_collision_sensitivity(2, wait=True)
        time.sleep(1)
        self.arm.set_state(0)
        time.sleep(1)

        # Gripper
        self.arm.set_gripper_mode(0)
        self.arm.set_gripper_enable(True)
        self.arm.set_gripper_speed(5000)
        time.sleep(1)
        self.arm.set_gripper_position(GRIPPER_OPEN, wait=True)
        time.sleep(1)
        
    def _disconnect(self):
        try:
            print("Disconnecting from xArm...")
            self.arm.disconnect()
        except Exception as e:
            # Avoid exceptions during interpreter shutdown
            print(f"Error during disconnect: {e}")

        self._is_connected = False

    # TODO handle failed gets
    def _get_obs(self):
        _, pose = self.arm.get_position_aa(is_radian=True)

        _, gripper_pos = self.arm.get_gripper_position()
        is_closed = pulse_to_g(gripper_pos)

        pose = np.array(pose)

        pose[0:3] /= 1000.0

        # if gripper_pos < GRIPPER_THRESH:
        #     ee_gripper = 1.0
        # else:
        #     ee_gripper = 0.0

        self._latest_obs = {
            'ee_pose': pose,
            'ee_gripper': is_closed,
        }

        return self._latest_obs
    
    def _get_info(self):
        return {
        }
    
    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        observation = self._get_obs()
        info = self._get_info()

        return observation, info
    
    def step(self, action):
        ee_actions = np.copy(action['ee_pose'])
        gripper_action = np.copy(action['ee_gripper'])

        # first move arm
        # need to scale action
        ee_actions[0:3] *= 1000
        self.arm.set_position_aa(ee_actions, is_radian=True, wait=False)
        # self.arm.set_servo_cartesian_aa(ee_actions, is_radian=True)

        # now gripper
        pulse = g_to_pulse(gripper_action)
        self.arm.set_gripper_position(pulse, wait=False)

        # if gripper_action > 0.5:
        # if gripper_action > 0.5:
        #     self.arm.set_gripper_position(GRIPPER_CLOSED, wait=False)
        # else:
        #     self.arm.set_gripper_position(GRIPPER_OPEN, wait=False)

        terminated = False
        reward = 0
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, False, info
    
    def render(self):
        pass

    def close(self):
        if self._is_connected:
            self._disconnect()

    def __del__(self):
        if self._is_connected:
            self._disconnect()
        




