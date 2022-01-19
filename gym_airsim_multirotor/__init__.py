from importlib.metadata import entry_points
from gym.envs.registration import register

register(
    id='airsim-multirotor-v0',
    entry_point='gym_airsim_multirotor.envs:AirsimMultirotor'
)

register(
    id = 'airsim-simple-dynamics-v0',
    entry_point = 'gym_airsim_multirotor.envs:SimpleDynamicEnv'
)