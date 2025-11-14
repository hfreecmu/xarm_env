from gymnasium.envs.registration import register

register(
    id="xarm_env/Xarm-v0",
    entry_point="xarm_env.envs:XarmEnv",
)