from gym.envs.registration import register

register(
    id='airsim-multirotor-v0',
    entry_point='gym_airsim_multirotor.envs:AirsimMultirotor',
    max_episode_steps=400
)