from gymnasium.envs.registration import register

register(
    id="xarm_servo_env/Xarm-v0",
    entry_point="xarm_env.envs:XarmServoEnv",
)

register(
    id="xarm_pos_env/Xarm-v0",
    entry_point="xarm_env.envs:XarmPosEnv",
)