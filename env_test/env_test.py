import gym
import gym_airsim_multirotor

env = gym.make("airsim-multirotor-v0")
env.read_config('gym_airsim_multirotor/envs/config.ini')