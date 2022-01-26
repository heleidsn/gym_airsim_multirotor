import gym
from gym import spaces
import airsim
from configparser import ConfigParser

import torch as th
import numpy as np
import math
import cv2

class FixedWingEnv(gym.Env):
    def __init__(self) -> None:
        super().__init__()

        env_name = 'City'

        # set start and goal position
        start_position = [0, 0, 5]
        goal_distance = 90
        self.dynamic_model.set_start([0, 0, 5], random_angle=math.pi*2)
        self.dynamic_model.set_goal(distance = 90, random_angle=math.pi*2)
        self.work_space_x = [start_position[0] - goal_distance - 10, start_position[0] + goal_distance + 10]
        self.work_space_y = [start_position[1] - goal_distance - 10, start_position[1] + goal_distance + 10]
        self.work_space_z = [0.5, 10]

        self.client = self.dynamic_model.client
        self.state_feature_length = self.dynamic_model.state_feature_length

        # training state
        self.episode_num = 0
        self.total_step = 0
        self.step_num = 0
        self.cumulated_episode_reward = 0
        self.previous_distance_from_des_point = 0

        # other settings
        self.distance_to_obstacles_accept = 2
        self.accept_radius = 2
        self.max_episode_steps = 600

        self.screen_height = 80
        self.screen_width = 100

        np.set_printoptions(formatter={'float': '{: 4.2f}'.format}, suppress=True)
        th.set_printoptions(profile="short", sci_mode=False, linewidth=1000)

        self.observation_space = spaces.Box(low=0, high=255, \
                                            shape=(self.screen_height, self.screen_width, 2),\
                                            dtype=np.uint8)

        self.action_space = self.dynamic_model.action_space