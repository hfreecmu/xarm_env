import numpy as np
import time

import gymnasium as gym
from gymnasium import spaces

from xarm_env.xarm_multiproc_servo import (
    XArmServoProcessController,
    # GRIPPER_OPEN_PULSE as GRIPPER_OPEN,
    # GRIPPER_CLOSED_PULSE as GRIPPER_CLOSED,
    # GRIPPER_THRESH_PULSE as GRIPPER_THRESH,
    # pulse_to_g,
)

from xarm_env.gripper_controller import (
    XArmGripperController,
)

class XarmServoEnv(gym.Env):
    def __init__(self, env_data, render_mode=None):
        assert render_mode is None

        ip = env_data["ip"]

        ee_controller_frequency = env_data["ee_controller_frequency"]
        ee_log_freq = env_data.get("ee_log_freq", None)
        ee_max_pos_speed = env_data.get("ee_max_pos_speed", 0.25)
        ee_max_rot_speed = env_data.get("ee_max_rot_speed", 0.6)
        ee_history_len = env_data.get("ee_history_len", 240)
        ee_queue_len = env_data.get("ee_queue_len", 2000)

        gripper_controller_frequency = env_data.get("gripper_controller_frequency", 30)
        gripper_log_freq = env_data.get("gripper_log_freq", None)
        gripper_history_len = env_data.get("gripper_history_len", 60)
        gripper_queue_len = env_data.get("gripper_queue_len", 2000)

        self.receive_robot_latency = env_data.get("receive_robot_latency", 0.0001)
        self.receive_gripper_latency = env_data.get("receive_gripper_latency", 0.01)

        self.compensate_latency = env_data.get("compensate_latency", False)
        self.robot_action_latency = env_data.get("robot_action_latency", 0.1)
        self.gripper_action_latency = env_data.get("gripper_action_latency", 0.1)

        # self.is_single = env_data.get("is_single", False)

        # self.receive_robot_latency = env_data.get("receive_robot_latency", 0.0001)
        # self.receive_gripper_latency = env_data.get("receive_gripper_latency", 0.01)
        # self.robot_action_latency = env_data.get("robot_action_latency", 0.1)

        # Setup EE Controller
        self.ee_controller = XArmServoProcessController(
            ip=ip,
            control_frequency=ee_controller_frequency,
            log_freq=ee_log_freq,
            max_pos_speed=ee_max_pos_speed,
            max_rot_speed=ee_max_rot_speed,
            history_len=ee_history_len,
            queue_len=ee_queue_len,
        )

        # Setup Gripper Controller
        self.gripper_controller = XArmGripperController(
            ip=ip,
            control_frequency=gripper_controller_frequency,
            log_freq=gripper_log_freq,
            history_len=gripper_history_len,
            queue_len=gripper_queue_len,
        )

        ee_start_succ = self.ee_controller.start()
        if not ee_start_succ:
            self.ee_controller.stop()
            raise RuntimeError('ee controller did not start succesfully')
        
        gripper_start_succ = self.gripper_controller.start()
        if not gripper_start_succ:
            self.gripper_controller.stop()
            raise RuntimeError('gripper controller did not start succesfully')
                
        # sleep extra half second just for safety
        time.sleep(0.5)

        # Setup gym
        # self.observation_space = spaces.Dict({
        #     "ee_pose": spaces.Box(low=-np.inf,
        #                               high=np.inf,
        #                               shape=(6,),
        #                               dtype=float),
        #     "ee_gripper": spaces.Discrete(2),
        # })

        self.observation_space = spaces.Dict({
            "ee_pose": spaces.Space(),   # accepts anything
            "robot_receive_timestamp": spaces.Space(),
            "robot_timestamp": spaces.Space(),

            "ee_gripper": spaces.Space(),
            "gripper_receive_timestamp": spaces.Space(),
            "gripper_timestamp": spaces.Space(),
        })
        
        self.action_space = spaces.Dict({
            "ee_pose": spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=float),
            "ee_gripper": spaces.Box(low=-np.inf, high=np.inf, dtype=float)
            #"gripper": spaces.Discrete(2),
        })

        self._latest_obs = None
        # self.close_thresh = pulse_to_g(GRIPPER_THRESH)

    def _get_obs(self):
        _, ee_history = self.ee_controller.get_ee_state()
        _, gripper_history = self.gripper_controller.get_gripper_state()

        ee_poses = ee_history[:, 0:6]
        robot_timestamps = ee_history[:, 6]

        is_closed_vals = gripper_history[:, 0]
        gripper_timestamps = gripper_history[:, 1]

        self._latest_obs = {
            'ee_pose': ee_poses,
            'robot_receive_timestamp': robot_timestamps,
            'robot_timestamp': robot_timestamps - self.receive_robot_latency,
            
            'ee_gripper': is_closed_vals,
            'gripper_receive_timestamp': gripper_timestamps,
            'gripper_timestamp': gripper_timestamps - self.receive_gripper_latency,
        }

        # if self.is_single:
        #     state = self.controller.get_ee_state()
    
        #     pose = np.array(
        #         [state.x, state.y, state.z, state.rx, state.ry, state.rz],
        #         dtype=float,
        #     )
        #     # ee_gripper = 0 if state.gripper < self.close_thresh else 1

        #     ee_gripper = state.gripper

        #     robot_timestamp = state.robot_timestamp
        #     gripper_timestamp = state.gripper_timestamp

        #     self._latest_obs = {
        #         'ee_pose': pose,
        #         'ee_gripper': ee_gripper,
        #         'robot_receive_timestamp': robot_timestamp,
        #         'robot_timestamp': robot_timestamp - self.receive_robot_latency,
        #         'gripper_receive_timestamp': gripper_timestamp,
        #         'gripper_timestamp': gripper_timestamp - self.receive_gripper_latency
        #     }

        # else:
        #     state = self.controller.get_ee_state_history_array()
        #     poses = state[:, 0:6]
        #     ee_grippers = np.where(state[:, 6] < self.close_thresh, 
        #                         np.zeros_like(state[:, 6]), 
        #                         np.ones_like(state[:, 6]))

        #     robot_timestamps = state[:, 7]
        #     gripper_timestamps = state[:, 8]

        #     self._latest_obs = {
        #         'ee_pose': poses,
        #         'ee_gripper': ee_grippers,
        #         'robot_receive_timestamp': robot_timestamps,
        #         'robot_timestamp': robot_timestamps - self.receive_robot_latency,
        #         'gripper_receive_timestamp': gripper_timestamps,
        #         'gripper_timestamp': gripper_timestamps - self.receive_gripper_latency,
        #     }

        return self._latest_obs
    
    def _get_info(self):
        return {
        }
    
    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # if not self.controller.is_alive():
        #     start_succ = self.controller.start()
        #     if not start_succ:
        #         raise RuntimeError("controller did not start successfully")

        #     time.sleep(0.5)   # safety pause

        observation = self._get_obs()
        info = self._get_info()

        return observation, info
    
    def step(self, action):
        ee_actions = np.copy(action['ee_pose'])
        gripper_actions = np.copy(action['ee_gripper'])
        timestamps = np.copy(action['timestamps'])

        receive_time = time.time()
        is_new = timestamps > receive_time
        new_ee_actions = ee_actions[is_new]
        new_gripper_actions = gripper_actions[is_new]
        new_timestamps = timestamps[is_new]

        r_latency = self.robot_action_latency if self.compensate_latency else 0.0
        g_latency = self.gripper_action_latency if self.compensate_latency else 0.0
        
        for ee_pose, ee_gripper, timestamp in zip(new_ee_actions, new_gripper_actions, new_timestamps):
            self.ee_controller.schedule_waypoint(ee_pose, 
                                              timestamp - r_latency)
            self.gripper_controller.schedule_waypoint(ee_gripper,
                                                      timestamp - g_latency)

        # gripper_actions = np.copy(action['ee_gripper'])

        # if not self.is_single:
        #     timestamps = np.copy(action['timestamps'])

        #     receive_time = time.time()
        #     is_new = timestamps > receive_time
        #     new_ee_actions = ee_actions[is_new]
        #     new_ee_gripper_actions = gripper_actions[is_new]
        #     new_timestamps = timestamps[is_new]

        #     r_latency = self.robot_action_latency if compensate_latency else 0.0
            
        #     target_timestamps = new_timestamps - r_latency

        #     self.controller.schedule_ee_targets(target_timestamps,
        #                                         new_ee_actions,
        #                                         new_ee_gripper_actions)
        # else:
        #     self.controller.set_ee_target(ee_actions, gripper_actions)

        terminated = False
        reward = 0
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, False, info

        # ee_actions = np.copy(action['ee_pose'])
        # gripper_action = np.copy(action['ee_gripper'])

        # gripper_norm = 1.0 if gripper_action > 0.5 else 0.0

        # self.controller.set_ee_target(ee_actions, gripper=gripper_norm)

        # terminated = False
        # reward = 0
        # observation = self._get_obs()
        # info = self._get_info()

        # return observation, reward, terminated, False, info
    
    def render(self):
        pass

    def close(self):
        self.ee_controller.stop()
        self.gripper_controller.stop()

    def __del__(self):
        self.ee_controller.stop()
        self.gripper_controller.stop()
        




