from random import random
from tracemalloc import start
import gym
from gym import spaces
import airsim
from configparser import ConfigParser

import torch as th
import numpy as np
import math
import cv2

from .dynamics_simple import SimpleDynamics
from .dynamics_multirotor import MultirotorDynamics

class AirsimGymEnv(gym.Env):
    def __init__(self) -> None:
        super().__init__()

        # select dynamics models
        self.dynamic_model = MultirotorDynamics()

        env_name = 'NH'
        # set start and goal position
        if env_name == 'NH':
            start_position = [110, 180, 5]
            goal_distance = 90
            self.dynamic_model.set_start(start_position, random_angle=0)
            self.dynamic_model.set_goal(distance = 90, random_angle=0)
            self.work_space_x = [start_position[0], start_position[0] + goal_distance + 10]
            self.work_space_y = [start_position[1] - 30, start_position[1] + 30]
            self.work_space_z = [0.5, 10]
        else:
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

    def step(self, action):
        # set action
        self._set_action(action)

        # get new obs
        obs = self._get_obs()
        done = self._is_done()
        info = {
            'is_success': self.is_in_desired_pose(),
            'is_crash': self.is_crashed(),
            'is_not_in_workspace': self.is_not_inside_workspace(),
            'step_num': self.step_num
        }
        if done:
            print(info)
        
        reward = self._compute_reward(done, action)
        self.cumulated_episode_reward += reward

        self._print_train_info(action, reward, info)

        self.step_num += 1
        self.total_step += 1

        return obs, reward, done, info

    def reset(self):
        # reset state  
        self.dynamic_model.reset()

        self.episode_num += 1
        self.step_num = 0
        self.cumulated_episode_reward = 0
        self.previous_distance_from_des_point = self.dynamic_model.goal_distance

        obs = self._get_obs()

        return obs

    def _set_action(self, action):

        self.dynamic_model.set_action(action)

    def _get_obs(self):
        '''
        @description: get depth image and target features for navigation
        @param {type}
        @return:
        '''
        # 1. get current depth image and transfer to 0-255  0-20m 255-0m
        image = self._get_depth_image()
        image_resize = cv2.resize(image, (self.screen_width, self.screen_height))
        image_scaled = image_resize * 100
        self.min_distance_to_obstacles = image_scaled.min()
        image_scaled = -np.clip(image_scaled, 0, 20) / 20 * 255 + 255  
        image_uint8 = image_scaled.astype(np.uint8)

        assert image_uint8.shape[0] == self.screen_height and image_uint8.shape[1] == self.screen_width, 'image size not match'
        
        # 2. get current state (relative_pose, velocity)
        state_feature_array = np.zeros((self.screen_height, self.screen_width))
        state_feature = self.dynamic_model._get_state_feature()

        assert (self.state_feature_length == state_feature.shape[0]), 'state_length {0} is not equal to state_feature_length {1}' \
                                                                    .format(self.state_feature_length, state_feature.shape[0])
        state_feature_array[0, 0:self.state_feature_length] = state_feature

        # 3. generate image with state
        image_with_state = np.array([image_uint8, state_feature_array])
        image_with_state = image_with_state.swapaxes(0, 2)
        image_with_state = image_with_state.swapaxes(0, 1)
        
        return image_with_state

    def _compute_reward(self, done, action):
        reward = 0
        reward_reach = 10
        reward_crash = -10
        reward_outside = -10

        if not done:
            distance_now = self.get_distance_to_goal_3d()
            reward_distance = (self.previous_distance_from_des_point - distance_now)
            self.previous_distance_from_des_point = distance_now

            reward_obs = 0
            action_cost = 0

            # add yaw_rate cost
            yaw_speed_cost = 0.2 * abs(action[-1]) / self.dynamic_model.max_vel_yaw_rad

            if self.dynamic_model.control_acc:
                acc_cost = 0.2 * abs(action[0]) / self.dynamic_model.max_acc_xy
                action_cost += acc_cost

            if self.dynamic_model.navigation_3d:
                v_z_cost = 0.2 * abs(action[1]) / self.dynamic_model.max_vel_z
                action_cost += v_z_cost
            
            action_cost += yaw_speed_cost

            reward = reward_distance - reward_obs - action_cost
        else:
            if self.is_in_desired_pose():
                reward = reward_reach
            if self.is_crashed():
                reward = reward_crash
            if self.is_not_inside_workspace():
                reward = reward_outside

        return reward

    def _is_done(self):
        episode_done = False

        is_not_inside_workspace_now = self.is_not_inside_workspace()
        has_reached_des_pose    = self.is_in_desired_pose()
        too_close_to_obstable   = self.is_crashed()

        # We see if we are outside the Learning Space
        episode_done = is_not_inside_workspace_now or\
                        has_reached_des_pose or\
                        too_close_to_obstable or\
                        self.step_num >= self.max_episode_steps
    
        return episode_done

    def _get_depth_image(self):

        responses = self.client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.DepthVis, True)
            ])

        # check observation
        while responses[0].width == 0:
            print("get_image_fail...")
            responses = self.client.simGetImages([
                airsim.ImageRequest("0", airsim.ImageType.DepthVis)
                ])

        depth_img = airsim.list_to_2d_float_array(responses[0].image_data_float, responses[0].width, responses[0].height)

        return depth_img

    def get_distance_to_goal_3d(self):
        current_pose = self.dynamic_model.get_position()
        goal_pose = self.dynamic_model.goal_position
        relative_pose_x = current_pose[0] - goal_pose[0]
        relative_pose_y = current_pose[1] - goal_pose[1]
        relative_pose_z = current_pose[2] - goal_pose[2]

        return math.sqrt(pow(relative_pose_x, 2) + pow(relative_pose_y, 2) + pow(relative_pose_z, 2))

    def is_not_inside_workspace(self):
        """
        Check if the Drone is inside the Workspace defined
        """
        is_not_inside = False
        current_position = self.dynamic_model.get_position()

        if current_position[0] < self.work_space_x[0] or current_position[0] > self.work_space_x[1] or \
            current_position[1] < self.work_space_y[0] or current_position[1] > self.work_space_y[1] or \
                current_position[2] < self.work_space_z[0] or current_position[2] > self.work_space_z[1]:
            is_not_inside = True

        return is_not_inside

    def is_in_desired_pose(self):
        in_desired_pose = False
        if self.get_distance_to_goal_3d() < self.accept_radius:
            in_desired_pose = True

        return in_desired_pose

    def is_crashed(self):
        is_crashed = False
        collision_info = self.client.simGetCollisionInfo()
        if collision_info.has_collided or self.min_distance_to_obstacles < self.distance_to_obstacles_accept:
            is_crashed = True

        return is_crashed

    def _print_train_info(self, action, reward, info):
        feature_all = self.model.actor.features_extractor.feature_all
        self.client.simPrintLogMessage('feature_all: ', str(feature_all))
        msg_train_info = "EP: {} Step: {} Total_step: {}".format(self.episode_num, self.step_num, self.total_step)

        self.client.simPrintLogMessage('Train: ', msg_train_info)
        self.client.simPrintLogMessage('Action: ', str(action))
        self.client.simPrintLogMessage('reward: ', "{:4.4f} total: {:4.4f}".format(reward, self.cumulated_episode_reward))
        self.client.simPrintLogMessage('Info: ', str(info))
        self.client.simPrintLogMessage('Feature_all: ', str(feature_all))
        self.client.simPrintLogMessage('Feature_norm: ', str(self.dynamic_model.state_norm))
        self.client.simPrintLogMessage('Feature_raw: ', str(self.dynamic_model.state_raw))
