import time
import numpy as np
import multiprocessing as mp
from dataclasses import dataclass
from typing import Optional, Tuple

GRIPPER_STATE_SIZE = 2

GRIPPER_OPEN = 800
GRIPPER_CLOSED = 0

from xarm_env.pose_trajectory_interpolator import PoseTrajectoryInterpolator

from xarm.wrapper import XArmAPI

@dataclass
class GripperState:
    is_closed: float
    timestamp: float

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "GripperState":
        return cls(*arr.tolist())

    def as_array(self) -> np.ndarray:
        return np.array(
            [self.is_closed, self.timestamp],
            dtype=float,
        )
    
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

def _configure_gripper(arm) -> None:
    arm.set_gripper_mode(0)
    arm.set_gripper_enable(True)
    # arm.set_gripper_speed(3000)
    arm.set_gripper_speed(5000)
    time.sleep(1)

    arm.set_gripper_position(GRIPPER_OPEN, wait=True)
    time.sleep(1)

def pulse_to_g(pulse):
    g = float((pulse - GRIPPER_OPEN) / (GRIPPER_CLOSED - GRIPPER_OPEN))

    g_clamped = min(max(g, 0.0), 1.0)
    return g_clamped

def g_to_pulse(g):
    g_clamped = min(max(g, 0.0), 1.0)
    pulse = GRIPPER_OPEN + g_clamped * (GRIPPER_CLOSED - GRIPPER_OPEN)
    return pulse

def _read_gripper_state(arm: XArmAPI) -> GripperState:
    code, gripper_pulse = arm.get_gripper_position()
    gripper_time = time.time()

    if code != 0:
        raise RuntimeError('invalid gripper code received read gripper state')
    
    is_closed = pulse_to_g(gripper_pulse)

    return GripperState(is_closed, gripper_time)

def _xarm_gripper_worker(
    ip,

    gripper_state_array: mp.Array,

    running_flag: mp.Value,
    ready_flag: mp.Value,
    control_frequency: float,

    history_len: int,
    history_array: mp.Array,
    history_index: mp.Value,

    wp_queue_len: int,
    wp_queue: mp.Array,
    wp_head: mp.Value,
    wp_tail: mp.Value,
    wp_count: mp.Value,

    # max_move_speed: float,

    log_freq: int,
):
    
    arm = None

    period = 1.0 / control_frequency

    def pop_ready_waypoint():
        with wp_queue.get_lock():
            if wp_count.value == 0:
                return None

            h = wp_head.value
            base = h * GRIPPER_STATE_SIZE
            
            wp = wp_queue[base:base+GRIPPER_STATE_SIZE] 

            wp_head.value = (wp_head.value + 1) % wp_queue_len
            wp_count.value -= 1

            return wp

    try:
        arm = XArmAPI(ip, is_radian=True)
        _configure_gripper(arm)

        gripper_state = _read_gripper_state(arm)
        initial = gripper_state.as_array()

        with gripper_state_array.get_lock():
            gripper_state_array[:] = initial[:]

            history_idx = history_index.value
            history_start = history_idx * GRIPPER_STATE_SIZE

            history_array[history_start:history_start+GRIPPER_STATE_SIZE] = initial[:]
            history_index.value = (history_idx + 1) % history_len

        print(f"[_xarm_gripper_worker] Started servo loop at {control_frequency} Hz")

        iter_idx = 0
        t_start = time.monotonic()
        last_waypoint_time = t_start

        pose_interp = PoseTrajectoryInterpolator(
                times=[t_start],
                poses=[[initial[0], 0, 0, 0, 0, 0]]
        )

        print(f'Gripper start is: {initial[0]}')

        ready_flag.value = True

        while running_flag.value:
            t_now = time.monotonic()

            pos_command = pose_interp(t_now)[0]
            # target_vel = (target_pos - pose_interp(t_now - period)[0]) / period

            if log_freq is not None and ((iter_idx % log_freq == 0) and (iter_idx != 0)):
                print(f'Gripper Average Frequency after {iter_idx} iterations: {(t_now - t_start) / iter_idx}')
                # print(f'Gripper command (pos, speed) is: {target_pos, target_vel}')
                # print(f'Gripper command (pos) is: {pos_command}')

            # set the command
            gripper_pulse = g_to_pulse(pos_command)
            code = arm.set_gripper_position(gripper_pulse,
                                            #speed=target_vel, need to convert units
                                            )

            if code != 0:
                raise RuntimeError(f'Invalid code: {code}')
            
            # update current state
            # TODO this reading of state can be put in another process
            gripper_state = _read_gripper_state(arm)
            current = gripper_state.as_array()
            with gripper_state_array.get_lock():
                gripper_state_array[:] = current[:]

                history_idx = history_index.value
                history_start = history_idx * GRIPPER_STATE_SIZE
                history_array[history_start:history_start+GRIPPER_STATE_SIZE] = current[:]
                history_index.value = (history_idx + 1) % history_len

            # fetch commands from queue
            wp = pop_ready_waypoint()
            if wp is not None:
                command = np.array(wp)
                target_is_closed = command[0]
                target_time = command[1]

                # translate global time to monotonic time
                target_time = time.monotonic() - time.time() + target_time
                # curr_time = t_now + period
                curr_time = t_now # for some reason gripper doesn't add period

                pose_interp = pose_interp.schedule_waypoint(
                    pose=[target_is_closed, 0, 0, 0, 0, 0],
                    time=target_time,
                    #max_pos_speed=max_move_speed,
                    #max_rot_speed=max_move_speed,
                    curr_time=curr_time,
                    last_waypoint_time=last_waypoint_time
                )
                last_waypoint_time = target_time

            t_wait_util = t_start + (iter_idx + 1) * period
            iter_idx += 1
            precise_wait(t_wait_util, time_func=time.monotonic)

    except KeyboardInterrupt:
        print("[_xarm_gripper_worker] Interrupted by user")
    except Exception as e:
        print(f"[_xarm_gripper_worker] Exception: {e}")
    finally:
        if arm is not None:
            try:
                arm.disconnect()
            except Exception:
                pass
        print("[_xarm_gripper_worker] Exiting")

class XArmGripperController:
    def __init__(
            self,
            ip: str,
            control_frequency: float = 30.0,
            history_len: int = 60,
            queue_len: int = 2000,
            log_freq: int = None,
            # max_move_speed: float = 200,
        ):

        self.ip = ip
        self._control_frequency = control_frequency
        self._log_freq = log_freq

        ctx = mp.get_context("spawn") 

        self._gripper_state_array = ctx.Array("d", GRIPPER_STATE_SIZE, lock=True)

        self._running_flag = ctx.Value("b", False)
        self._ready_flag = ctx.Value("b", False)

        self.history_len = history_len
        self._history_array = ctx.Array("d", self.history_len * GRIPPER_STATE_SIZE, lock=True)
        self._history_index = ctx.Value("i", 0)

        self._wp_queue_len = queue_len
        self._wp_queue = ctx.Array("d", self._wp_queue_len * GRIPPER_STATE_SIZE, lock=True)
        self._wp_head = ctx.Value("i", 0) 
        self._wp_tail = ctx.Value("i", 0) 
        self._wp_count = ctx.Value("i", 0) 

        self._process = ctx.Process(
            target=_xarm_gripper_worker,
            args=(
                self.ip,

                self._gripper_state_array,

                self._running_flag,
                self._ready_flag,
                self._control_frequency,

                self.history_len,
                self._history_array,
                self._history_index,

                self._wp_queue_len,
                self._wp_queue,
                self._wp_head,
                self._wp_tail,
                self._wp_count,

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
            print("[XArmGripperController] Warning: process still alive after stop timeout")

    def schedule_waypoint(self, is_closed, target_time):
        with self._wp_queue.get_lock():
            if self._wp_count.value >= self._wp_queue_len:
                raise RuntimeError('Queue exceeded limits')
                # self._wp_head.value = (self._wp_head.value + 1) % self._wp_queue_len
                # self._wp_count.value -= 1

            #print(f'scheduling waypoiont at time: {target_time}, {target_time - time.time()}')

            tail = self._wp_tail.value  
            base = tail * GRIPPER_STATE_SIZE

            wp = [float(is_closed)] + [target_time]
            self._wp_queue[base:base+GRIPPER_STATE_SIZE] = wp[:]

            self._wp_tail.value = (self._wp_tail.value + 1) % self._wp_queue_len
            self._wp_count.value += 1

    def get_gripper_state(self): #-> Tuple[GripperState|np.ndarray]:
        """Return latest measured EEState (non-blocking)."""
        with self._gripper_state_array.get_lock():
            arr = np.array(self._gripper_state_array[:GRIPPER_STATE_SIZE], dtype=float)

            history_idx = self._history_index.value
            raw = np.array(self._history_array[:], dtype=float).reshape(self.history_len, GRIPPER_STATE_SIZE)
        
            ordered = np.concatenate([raw[history_idx:], raw[:history_idx]], axis=0)

            history_result = []
            for row in ordered:
                if not np.allclose(row, 0.0):
                    # result.append(EEState.from_array(row))
                    history_result.append(row)

            history_result = np.array(history_result)

        return GripperState.from_array(arr), history_result

    def is_alive(self) -> bool:
        return self._process.is_alive()

def run_demo():
    duration = 100
    total_iters = 500
    dt = 0.025
    time_offset = 5
    time_wait_at_end = 5

    assert duration % 2 == 0
    assert total_iters % duration == 0

    ctrl = XArmGripperController(
        ip="192.168.1.212",
        log_freq=240,
    )

    # important not to exceed queue
    assert total_iters < ctrl._wp_queue_len

    start_succ = ctrl.start()

    if not start_succ:
        ctrl.stop()
        raise RuntimeError('gripper did not start succesfully')
    
    # give gripper time to populate as not as frequent
    time.sleep(1)

    state, history = ctrl.get_gripper_state()
    assert np.isclose(state.as_array(), history[-1]).all()

    closing_vals = np.linspace(0, 1.0, duration // 2, endpoint=True)
    opening_vals = np.linspace(1.0, 0.0, duration // 2, endpoint=True)
    single_duration_vals = np.concatenate((closing_vals, opening_vals))
    total_vals = np.concatenate([single_duration_vals]*(total_iters//duration))

    t0 = time.time() + time_offset
    for ind in range(total_iters):
        ts = t0 + ind * dt 
        is_closed = total_vals[ind]

        ctrl.schedule_waypoint(is_closed, ts)

    time_to_sleep = time_wait_at_end + ts - time.time()
    if time_to_sleep > 0:
        print('waiting for commands')
        time.sleep(time_to_sleep)
    
    print('done')

    ctrl.stop()

if __name__ == "__main__":
    run_demo()
