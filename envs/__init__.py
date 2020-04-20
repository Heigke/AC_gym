from gym.envs.registration import register
register(id=’CustomEnv-v0',
    entry_point=’envs.AC_man_dir:AC_man’
)