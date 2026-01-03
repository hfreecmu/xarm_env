import time
import math
import numpy as np
import multiprocessing as mp
from dataclasses import dataclass
from typing import Optional, Tuple

from xarm_env.pose_trajectory_interpolator import PoseTrajectoryInterpolator

from xarm.wrapper import XArmAPI

# GRIPPER_OPEN_PULSE = 800
# GRIPPER_CLOSED_PULSE = 0
# GRIPPER_THRESH_PULSE = 800
# GRIPPER_THRESH_PULSE = 700 

EE_STATE_SIZE = 7
EE_CMD_SIZE = 6

@dataclass
class EEState:
    x: float
    y: float
    z: float
    rx: float
    ry: float
    rz: float
    timestamp: float

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "EEState":
        return cls(*arr.tolist())

    def as_array(self) -> np.ndarray:
        return np.array(
            [self.x, self.y, self.z, self.rx, self.ry, self.rz, self.timestamp],
            dtype=float,
        )

# def _rate_sleep(start_t: float, period: float, slack_time: float=0.001) -> None:
#     t_end = start_t + period
#     remaining = t_end - time.monotonic()

#     if remaining > 0:
#         t_sleep = remaining - slack_time
#         if t_sleep > 0:
#             time.sleep(t_sleep)
#         while time.monotonic() < t_end:
#             pass

#     return

def precise_wait(t_end: float, slack_time: float=0.001, time_func=time.monotonic):
    t_start = time_func()
    t_wait = t_end - t_start
    if t_wait > 0:
        t_sleep = t_wait - slack_time
        if t_sleep > 0:
            time.sleep(t_sleep)
        while time_func() < t_end:
            pass
    return

def _clear_error_and_setup_servo_mode(arm) -> None:
    arm.clean_error()
    arm.clean_warn()
    arm.motion_enable(True)
    time.sleep(1)

    # arm.set_mode(1)
    ##
    arm.set_mode(7)
    ##
    time.sleep(1)

    arm.set_collision_sensitivity(2, wait=True)
    # arm.set_collision_sensitivity(1, wait=True)
    # arm.set_collision_sensitivity(0, wait=True)
    time.sleep(1)

    arm.set_state(state=0)
    time.sleep(1)

    # arm.set_gripper_enable(True)
    # time.sleep(1)

    # arm.set_gripper_mode(0)
    # time.sleep(1)

    # arm.set_gripper_speed(3000)
    # time.sleep(1)

    # arm.set_gripper_position(GRIPPER_OPEN_PULSE, wait=True)
    # time.sleep(1)

def _get_gripper_norm(arm: XArmAPI) -> float:
    """Read gripper pulse and convert to [0, 1] (0=open, 1=closed)."""
    code, gpos = arm.get_gripper_position()
    
    if code != 0:
        arm.set_state(4)
        raise RuntimeError('invalid code received get gripper pos')


    return pulse_to_g(gpos)

def _set_gripper_from_norm(arm: XArmAPI, g: float) -> None:
    """Command gripper using normalized input [0,1]."""
    #g = np.round(g)
    pulse = g_to_pulse(g)
    arm.set_gripper_position(int(pulse), wait=False)

def pulse_to_g(pulse):
    denom = (GRIPPER_CLOSED_PULSE - GRIPPER_OPEN_PULSE)
    g = float((pulse - GRIPPER_OPEN_PULSE) / denom)

    g_clamped = min(max(g, 0.0), 1.0)
    return g_clamped

def g_to_pulse(g):
    g_clamped = min(max(g, 0.0), 1.0)
    pulse = GRIPPER_OPEN_PULSE + g_clamped * (GRIPPER_CLOSED_PULSE - GRIPPER_OPEN_PULSE)
    return pulse

def _read_ee_state(arm: XArmAPI) -> EEState:
    code, pose = arm.get_position_aa(is_radian=True)
    robot_time = time.time()

    if code != 0:
        arm.set_state(4)
        raise RuntimeError('invalid code received read ee state')
    
    pose = np.array(pose, dtype=float)
    x_m, y_m, z_m = pose[0] / 1000.0, pose[1] / 1000.0, pose[2] / 1000.0
    rx, ry, rz = pose[3], pose[4], pose[5]
    
    return EEState(x_m, y_m, z_m, rx, ry, rz, robot_time)

def _xarm_servo_worker(
    ip,

    # ee_cmd_array: mp.Array,
    ee_state_array: mp.Array,

    running_flag: mp.Value,
    ready_flag: mp.Value,
    control_frequency: float,

    # max_cart_delta: float,

    history_len: int,
    history_array: mp.Array,
    history_index: mp.Value,

    wp_queue_len: int,
    wp_queue: mp.Array,
    wp_head: mp.Value,
    wp_tail: mp.Value,
    wp_count: mp.Value,

    max_pos_speed: float,
    max_rot_speed: float,

    log_freq: int,
):
    arm = None

    period = 1.0 / control_frequency

    def pop_ready_waypoint():
        with wp_queue.get_lock():
            if wp_count.value == 0:
                return None

            h = wp_head.value
            base = h * EE_STATE_SIZE
            
            wp = wp_queue[base:base+EE_STATE_SIZE] 

            wp_head.value = (wp_head.value + 1) % wp_queue_len
            wp_count.value -= 1

            return wp

    try:
        arm = XArmAPI(ip, is_radian=True)
        _clear_error_and_setup_servo_mode(arm)

        ee_state = _read_ee_state(arm)
        initial = ee_state.as_array()
        # initial_time = initial[-1]

        # set initial ee commands. all but timestep
        # with ee_cmd_array.get_lock():
        #     ee_cmd_array[:EE_CMD_SIZE] = initial[:EE_CMD_SIZE]

        # set initial ee state
        with ee_state_array.get_lock():
            ee_state_array[:] = initial[:]

            # with history_array.get_lock(), history_index.get_lock():
            history_idx = history_index.value
            history_start = history_idx * EE_STATE_SIZE

            history_array[history_start:history_start+EE_STATE_SIZE] = initial[:]
            history_index.value = (history_idx + 1) % history_len

        print(f"[xarm_servo_worker] Started servo loop at {control_frequency} Hz")

        iter_idx = 0
        t_start = time.monotonic()
        last_waypoint_time = t_start

        pose_interp = PoseTrajectoryInterpolator(
                times=[t_start],
                poses=[initial[0:EE_CMD_SIZE]]
        )

        print(f'Pose start is: {initial[0:EE_CMD_SIZE]}')
        
        ready_flag.value = True

        while running_flag.value:
            t_now = time.monotonic()

            pose_command = pose_interp(t_now)
            if log_freq is not None and ((iter_idx % log_freq == 0) and (iter_idx != 0)):
                print(f'Arm Average Frequency after {iter_idx} iterations: {(t_now - t_start) / iter_idx}')
                #print(f'Arm Pose command is: {pose_command}')

            # set the command
            axis_angle_pose_mm = [
                float(pose_command[0] * 1000.0),
                float(pose_command[1] * 1000.0),
                float(pose_command[2] * 1000.0),
                float(pose_command[3]),
                float(pose_command[4]),
                float(pose_command[5]),
            ]
            # code = arm.set_servo_cartesian_aa(
            #         axis_angle_pose_mm,
            #         is_radian=True,
            #         # wait=True,
            #         # I am not sure if wait does anything,
            #         # I don't think it does
            # )
            code = arm.set_position_aa(
                axis_angle_pose_mm,
                speed=300,
                mvacc=2000,
            )
            if code != 0:
                arm.set_state(4)
                raise RuntimeError(f'Invalid code: {code}')

            # update current state
            # TODO this reading of state can be put in another process
            # same with gripper when we get there
            ee_state = _read_ee_state(arm)
            current = ee_state.as_array()
            with ee_state_array.get_lock():
                ee_state_array[:] = current[:]

                # with history_array.get_lock(), history_index.get_lock():
                history_idx = history_index.value
                history_start = history_idx * EE_STATE_SIZE

                # for j in range(STATE_SIZE):
                #     history_array[start + j] = current[j]
                history_array[history_start:history_start+EE_STATE_SIZE] = current[:]
                history_index.value = (history_idx + 1) % history_len

            # fetch commands from queue
            wp = pop_ready_waypoint()
            if wp is not None:
                command = np.array(wp)
                target_pose = command[0:6]
                target_time = command[6]
                #delta_start_time = target_time - initial_time
                #delta_now_time = target_time - time.time()
                # print(f'got waypoint at time: {target_time}, {target_time-time.time()}')
                # print(f'received pose: {target_pose} at {delta_start_time} since start and {delta_now_time} since now')

                # translate global time to monotonic time
                target_time = time.monotonic() - time.time() + target_time
                curr_time = t_now + period

                pose_interp = pose_interp.schedule_waypoint(
                    pose=target_pose,
                    time=target_time,
                    max_pos_speed=max_pos_speed,
                    max_rot_speed=max_rot_speed,
                    curr_time=curr_time,
                    last_waypoint_time=last_waypoint_time
                )
                last_waypoint_time = target_time

            t_wait_util = t_start + (iter_idx + 1) * period
            iter_idx += 1
            precise_wait(t_wait_util, time_func=time.monotonic)
            
            continue

            # wp = pop_ready_waypoint()
            # continue

            # if wp is not None:
            #     command = np.array(wp)
            #     target_pose = command[0:7]
            #     target_time = command[7]

            #     # translate global time to monotonic time
            #     target_time = time.monotonic() - time.time() + target_time
            #     curr_time = loop_start + period

            #     pose_interp = pose_interp.schedule_waypoint(
            #         pose=target_pose,
            #         time=target_time,
            #         max_pos_speed=max_pos_speed,
            #         max_rot_speed=max_rot_speed,
            #         curr_time=curr_time,
            #         last_waypoint_time=last_waypoint_time
            #     )
            #     last_waypoint_time = target_time

            #     desired = np.array(wp[1:8], dtype=float)
            #     with cmd_array.get_lock():
            #         # for i in range(CMD_SIZE):
            #         #     cmd_array[i] = desired[i]
            #         cmd_array[:] = desired[:]

            # else:
            #     with cmd_array.get_lock():
            #         desired = np.array(cmd_array[:CMD_SIZE], dtype=float)

            # ee_state = _read_ee_state(arm)
            # current = ee_state.as_array()

            # with state_array.get_lock():
            #     # for i, v in enumerate(current):
            #     #     state_array[i] = v
            #     state_array[:] = current[:]

            # with history_array.get_lock(), history_index.get_lock():
            #     idx = history_index.value
            #     start = idx * STATE_SIZE

            #     # for j in range(STATE_SIZE):
            #     #     history_array[start + j] = current[j]
            #     history_array[start:start+STATE_SIZE] = current[:]

            #     history_index.value = (idx + 1) % history_len

            # cur_xyz = current[0:3]
            # des_xyz = desired[0:3]
            # delta_xyz = des_xyz - cur_xyz

            # #curr_grip = current[6]
            # #des_grip = desired[6]
            # #delta_grip = des_grip - curr_grip
            
            # dist = float(np.linalg.norm(delta_xyz))

            # next_gripper = desired[6]
            # if dist > max_cart_delta and dist > 1e-9:
            #     delta_xyz = delta_xyz * (max_cart_delta / dist)
            #     #delta_grip = delta_grip * (max_cart_delta / dist)
            #     # next_gripper = current[6]
            #     # print('why why why', dist)

            # next_xyz = cur_xyz + delta_xyz
            # next_rx, next_ry, next_rz = desired[3:6]

            # # next_gripper = desired[6]
            # #next_gripper = curr_grip + delta_grip
            # # TODO THIS SHOULD NO BE HERE
            # # next_gripper = np.round(next_gripper)

            # axis_angle_pose_mm = [
            #     float(next_xyz[0] * 1000.0),
            #     float(next_xyz[1] * 1000.0),
            #     float(next_xyz[2] * 1000.0),
            #     float(next_rx),
            #     float(next_ry),
            #     float(next_rz),
            # ]

            # ret = arm.set_servo_cartesian_aa(
            #     axis_angle_pose_mm,
            #     is_radian=True,
            # )

            # if ret !=0:
            #     arm.set_state(4)
            #     raise RuntimeError('set servo failed')
            
            # _set_gripper_from_norm(arm, float(next_gripper))

            # # iter_idx += 1
            # # precise_wait(t_cycle_end, slack_time=0)
            # _rate_sleep(loop_start, period)

    except KeyboardInterrupt:
        print("[xarm_servo_worker] Interrupted by user")
    except Exception as e:
        print(f"[xarm_servo_worker] Exception: {e}")
    finally:
        if arm is not None:
            try:
                arm.set_state(4)
                arm.disconnect()
            except Exception:
                pass
        print("[xarm_servo_worker] Exiting")

class XArmServoProcessController:
    def __init__(
            self,
            ip: str,
            control_frequency: float = 50.0,
            # max_cart_delta: float = 0.0005,
            #max_cart_delta: float = 0.002,
            #hisotry_len: int = 60,
            history_len: int = 240,
            queue_len: int = 2000,
            log_freq: int = None,
            max_pos_speed: float = 0.25,
            max_rot_speed: float = 0.6,
        ):

        self.ip = ip
        self._control_frequency = control_frequency
        #self.max_cart_delta = max_cart_delta
        self._log_freq = log_freq

        ctx = mp.get_context("spawn") 

        # self._ee_cmd_array = ctx.Array("d", EE_CMD_SIZE, lock=True)
        self._ee_state_array = ctx.Array("d", EE_STATE_SIZE, lock=True)

        self._running_flag = ctx.Value("b", False)
        self._ready_flag = ctx.Value("b", False)

        self.history_len = history_len
        self._history_array = ctx.Array("d", self.history_len * EE_STATE_SIZE, lock=True)
        self._history_index = ctx.Value("i", 0)

        self._wp_queue_len = queue_len
        self._wp_queue = ctx.Array("d", self._wp_queue_len * EE_STATE_SIZE, lock=True)
        self._wp_head = ctx.Value("i", 0) 
        self._wp_tail = ctx.Value("i", 0) 
        self._wp_count = ctx.Value("i", 0) 

        self._max_pos_speed = max_pos_speed
        self._max_rot_speed = max_rot_speed
        cube_diag = np.linalg.norm([1,1,1])

        self._process = ctx.Process(
            target=_xarm_servo_worker,
            args=(
                self.ip,

                # self._ee_cmd_array,
                self._ee_state_array,

                self._running_flag,
                self._ready_flag,
                self._control_frequency,

                # self.max_cart_delta,

                self.history_len,
                self._history_array,
                self._history_index,

                self._wp_queue_len,
                self._wp_queue,
                self._wp_head,
                self._wp_tail,
                self._wp_count,

                self._max_pos_speed*cube_diag,
                self._max_rot_speed*cube_diag,

                self._log_freq,
            ),
        )

        self._process.daemon = True

    def start(self, timeout: Optional[float] = 20.0) -> bool:
        if self.is_alive():
            return
        self._running_flag.value = True
        self._process.start()

        start_time = time.monotonic()
        while not self._ready_flag.value:
            time.sleep(0.1)

            if time.monotonic() - start_time > timeout:
                break
        
        if not self._ready_flag.value:
            print('Process did not ready up.')

            self.stop()
            return False
        
        return True

    def stop(self, timeout: Optional[float] = 5.0) -> None:
        if not self.is_alive():
            return
        self._running_flag.value = False
        self._process.join(timeout=timeout)
        if self.is_alive():
            print("[XArmServoProcessController] Warning: process still alive after stop timeout")

    def schedule_waypoint(self, pose, target_time):
        pose = np.array(pose)
        assert pose.shape == (6,)

        with self._wp_queue.get_lock():
            if self._wp_count.value >= self._wp_queue_len:
                raise RuntimeError('Queue exceeded limits')
                # self._wp_head.value = (self._wp_head.value + 1) % self._wp_queue_len
                # self._wp_count.value -= 1

            #print(f'scheduling waypoiont at time: {target_time}, {target_time - time.time()}')

            tail = self._wp_tail.value  
            base = tail * EE_STATE_SIZE

            wp = pose.tolist() + [target_time]
            self._wp_queue[base:base+EE_STATE_SIZE] = wp[:]

            self._wp_tail.value = (self._wp_tail.value + 1) % self._wp_queue_len
            self._wp_count.value += 1

    # def set_ee_target(self, pose_aa: np.ndarray, gripper: float) -> None:
    #     """
    #     Set desired EE axis-angle pose and gripper from main process.

    #     pose_aa: np.ndarray with shape (6,)
    #              [x(m), y(m), z(m), rx(rad), ry(rad), rz(rad)]
    #     gripper: normalized 0.0(open) .. 1.0(closed)
    #     """
    #     pose_aa = np.array(pose_aa, dtype=float)

    #     g = float(gripper)
    #     with self._cmd_array.get_lock():
    #         # for i in range(6):
    #         #     self._cmd_array[i] = pose_aa[i]
    #         self._cmd_array[0:6] = pose_aa[:]
    #         self._cmd_array[6] = g

    # def schedule_ee_targets(self, timestamps: np.ndarray, poses: np.ndarray, grippers: np.ndarray):
    #     with self._wp_queue.get_lock():
    #         for i in range(len(timestamps)):
    #             if self._wp_count.value >= self._wp_queue_len:
    #                 self._wp_head.value = (self._wp_head.value + 1) % self._wp_queue_len
    #                 self._wp_count.value -= 1

    #             t = self._wp_tail.value
    #             base = t * 8

    #             wp = [
    #                 float(timestamps[i]),
    #                 float(poses[i,0]), float(poses[i,1]), float(poses[i,2]),
    #                 float(poses[i,3]), float(poses[i,4]), float(poses[i,5]),
    #                 float(grippers[i]),
    #             ]

    #             # for j in range(8):
    #             #     self._wp_queue[base + j] = wp[j]
    #             self._wp_queue[base:base+8] = wp[:]

    #             self._wp_tail.value = (self._wp_tail.value + 1) % self._wp_queue_len
    #             self._wp_count.value += 1

    def get_ee_state(self):# -> Tuple[EEState | np.ndarray]:
        """Return latest measured EEState (non-blocking)."""
        with self._ee_state_array.get_lock():
            arr = np.array(self._ee_state_array[:EE_STATE_SIZE], dtype=float)

            history_idx = self._history_index.value
            raw = np.array(self._history_array[:], dtype=float).reshape(self.history_len, EE_STATE_SIZE)
        
            ordered = np.concatenate([raw[history_idx:], raw[:history_idx]], axis=0)

            history_result = []
            for row in ordered:
                if not np.allclose(row, 0.0):
                    # result.append(EEState.from_array(row))
                    history_result.append(row)

            history_result = np.array(history_result)

        return EEState.from_array(arr), history_result
    
    # def get_ee_state_array(self) -> np.ndarray:
    #     """Return latest state as np.ndarray of shape (7,)."""
    #     with self._state_array.get_lock():
    #         return np.array(self._state_array[:STATE_SIZE], dtype=float)
        
    # def get_ee_state_history_array(self):
    #     with self._history_array.get_lock(), self._history_index.get_lock():
    #         idx = self._history_index.value
    #         raw = np.array(self._history_array[:], dtype=float).reshape(self.history_len, STATE_SIZE)

    #     ordered = np.concatenate([raw[idx:], raw[:idx]], axis=0)

    #     result = []
    #     for row in ordered:
    #         if not np.allclose(row, 0.0):
    #             # result.append(EEState.from_array(row))
    #             result.append(row)

    #     result = np.array(result)

    #     return result
        
    def is_alive(self) -> bool:
        return self._process.is_alive()

def run_demo():
    radius = 0.05
    duration = 100
    total_iters = 1000
    dt = 0.05
    time_offset = 5
    time_wait_at_end = 5

    assert total_iters % duration == 0

    # Simple test: move in a small circle in front of robot
    ctrl = XArmServoProcessController(
        ip="192.168.1.212",
        control_frequency=200.0,
        #max_cart_delta=0.005,
        log_freq=800,
        max_pos_speed=2.0,
        max_rot_speed=6.0,
    )

    # important not to exceed queue
    assert total_iters < ctrl._wp_queue_len

    start_succ = ctrl.start()

    if not start_succ:
        ctrl.stop()
        raise RuntimeError('did not start succesfully')

    state, history = ctrl.get_ee_state()
    assert np.isclose(state.as_array(), history[-1]).all()

    center = np.array([state.x, state.y - radius, state.z])

    t0 = time.time() + time_offset

    for ind in range(total_iters + 1):
        ts = t0 + ind * dt 
        phase = ind % duration

        angle = 2 * math.pi * (phase / duration) 
        offset = np.array([0.0, radius * math.cos(angle), radius * math.sin(angle)])
        target_xyz = center + offset

        pose = np.array(
            [target_xyz[0], target_xyz[1], target_xyz[2], state.rx, state.ry, state.rz],
            dtype=float,
        )

        ctrl.schedule_waypoint(pose, ts)

    time_to_sleep = time_wait_at_end + ts - time.time()
    if time_to_sleep > 0:
        print('waiting for commands')
        time.sleep(time_to_sleep)
    
    print('done')

    ctrl.stop()

if __name__ == "__main__":
    run_demo()

    # try:
    #     MODE = "scheduled"

    #     # Get initial position
    #     state = ctrl.get_ee_state()
    #     center = np.array([state.x, state.y, state.z])
    #     radius = 0.05
    #     duration = 10
    #     total_iters = 100
    #     dt = 0.1

    #     if MODE == "direct":
    #         for ind in range(total_iters):
    #             phase = ind % duration

    #             angle = 2 * math.pi * (phase / duration)
    #             offset = np.array([0.0, radius * math.cos(angle), radius * math.sin(angle)])
    #             target_xyz = center + offset

    #             pose = np.array(
    #                 [target_xyz[0], target_xyz[1], target_xyz[2], state.rx, state.ry, state.rz],
    #                 dtype=float,
    #             )

    #             ctrl.set_ee_target(pose, gripper=state.gripper)

    #             time.sleep(dt)

    #         pose = np.array(
    #             [center[0], center[1], center[2], state.rx, state.ry, state.rz],
    #             dtype=float,
    #         )
        
    #         ctrl.set_ee_target(pose, gripper=state.gripper)

    #     elif MODE == "scheduled":

    #         timestamps = []
    #         poses = []
    #         grippers = []

    #         t0 = time.time()

    #         for ind in range(total_iters):
    #             ts = t0 + ind * dt 
    #             phase = ind % duration

    #             angle = 2 * math.pi * (phase / duration) 
    #             offset = np.array([0.0, radius * math.cos(angle), radius * math.sin(angle)])
    #             target_xyz = center + offset

    #             pose = np.array(
    #                 [target_xyz[0], target_xyz[1], target_xyz[2], state.rx, state.ry, state.rz],
    #                 dtype=float,
    #             )

    #             timestamps.append(ts)
    #             poses.append(pose)
    #             grippers.append(state.gripper)

    #         ctrl.schedule_ee_targets(
    #             timestamps=np.array(timestamps),
    #             poses=np.array(poses),
    #             grippers=np.array(grippers),
    #         )

    #         time.sleep(total_iters * dt + 0.25)

    #         pose = np.array(
    #                 [center[0], center[1], center[2], state.rx, state.ry, state.rz],
    #                 dtype=float,
    #             )
            
    #         ctrl.set_ee_target(pose, gripper=state.gripper)
    #         time.sleep(dt + 0.25)

    # finally:
    #     ctrl.stop()

    