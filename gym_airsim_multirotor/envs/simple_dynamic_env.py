from msilib.schema import Class
from os import curdir
from turtle import position
import gym
from gym import spaces
import airsim
from configparser import ConfigParser

import torch as th
import numpy as np
import math
import cv2

class SimpleDynamicEnv(gym.Env):
    def __init__(self, 
                 x_init=0, 
                 y_init=0, 
                 z_init=5, 
                 yaw_rad_init=0, 
                 dt=0.1) -> None:
        super().__init__()
        print('welcome to SimpleDynamicEnv')

        # init airsim client
        self.client = airsim.VehicleClient()
        self.client.confirmConnection()
        
        # for stable baselines policy
        self.model = None

        # UAV state
        self.x = x_init
        self.y = y_init
        self.z = z_init
        self.yaw_rad = yaw_rad_init # -180~180
        self.v_xy = 0
        self.v_z = 0
        self.yaw_rate = 0
        self.dt = dt

        # goal
        self.start_position = [x_init, y_init, z_init, yaw_rad_init]
        self.goal_angle_noise_degree = 180
        self.goal_position = np.zeros(3)
        self.goal_distance = 100

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

        self.work_space_xy_max = self.goal_distance + 10
        self.work_space_z_max =  10
        self.work_space_z_min = 0.5
        self.max_vertical_difference = 5

        self.navigation_3d = False
        self.control_acc = True
        self.max_acc_xy = 5
        self.max_vel_x = 5
        self.min_vel_x = 1
        self.max_vel_z = 1
        self.max_vel_yaw_deg = 50
        self.max_vel_yaw_rad = math.radians(self.max_vel_yaw_deg)
        self.screen_height = 80
        self.screen_width = 100

        np.set_printoptions(formatter={'float': '{: 4.2f}'.format}, suppress=True)
        th.set_printoptions(profile="short", sci_mode=False, linewidth=1000)

        self.observation_space = spaces.Box(low=0, high=255, \
                                            shape=(self.screen_height, self.screen_width, 2),\
                                            dtype=np.uint8)

        if self.navigation_3d:
            self.state_feature_length = 6
            self.action_space = spaces.Box(low=np.array([self.min_vel_x , -self.max_vel_z, -self.max_vel_yaw_rad]), \
                                            high=np.array([self.max_vel_x, self.max_vel_z, self.max_vel_yaw_rad]), \
                                            dtype=np.float32)
        else:
            self.state_feature_length = 4
            if self.control_acc:
                self.action_space = spaces.Box(low=np.array([-self.max_acc_xy, -self.max_vel_yaw_rad]), \
                                               high=np.array([self.max_acc_xy, self.max_vel_yaw_rad]) 
                                    )
            else:
                self.action_space = spaces.Box(low=np.array([self.min_vel_x , -self.max_vel_yaw_rad]), \
                                                high=np.array([self.max_vel_x, self.max_vel_yaw_rad]), \
                                                dtype=np.float32)



    def step(self, action):
        # set action
        self._set_action(action)

        # get new obs
        obs = self._get_obs()
        done = self._is_done()
        info = {
            'is_success': self.is_in_desired_pose(),
            # 'is_crash': self.is_crashed() or not self.is_inside_workspace(),
            'is_crash': self.is_crashed(),
            'step_num': self.step_num
        }
        reward = self._compute_reward(done, action)
        self.cumulated_episode_reward += reward

        self._print_train_info(action, reward, info)

        self.step_num += 1
        self.total_step += 1

        return obs, reward, done, info

    def reset(self):
        # reset state
        yaw_noise = math.radians(360) * np.random.random() - math.pi
        self.x = self.start_position[0]
        self.y = self.start_position[1]
        self.z = self.start_position[2]
        self.yaw_rad = self.start_position[3] + yaw_noise# -180~180
        self.v_xy = 0
        self.v_z = 0
        self.yaw_rate = 0

        # reset pose
        pose = self.client.simGetVehiclePose()
        pose.position.x_val = self.x
        pose.position.y_val = self.y
        pose.position.z_val = -self.z
        
        pose.orientation = airsim.to_quaternion(0, 0, self.yaw_rad)
        self.client.simSetVehiclePose(pose, False)

        self.episode_num += 1
        self.step_num = 0
        self.cumulated_episode_reward = 0
        self.previous_distance_from_des_point = self.goal_distance

        self._set_goal_pose()
        obs = self._get_obs()

        return obs

    def _set_action(self, action):

        if self.control_acc:
            acc_xy = action[0]
            v_xy = self.v_xy + acc_xy * self.dt
            if v_xy > self.max_vel_x:
                v_xy = self.max_vel_x
            elif v_xy < self.min_vel_x:
                v_xy = self.min_vel_x
        else:
            v_xy = action[0]
        
        yaw_rate = action[-1]

        self.v_xy = v_xy
        self.yaw_rate = yaw_rate

        self.yaw_rad += yaw_rate * self.dt
        if self.yaw_rad > math.radians(180):
            self.yaw_rad -= math.pi * 2
        elif self.yaw_rad < math.radians(-180):
            self.yaw_rad += math.pi * 2
        
        self.x += v_xy * math.cos(self.yaw_rad) * self.dt
        self.y += v_xy * math.sin(self.yaw_rad) * self.dt

        if len(action) == 3:
            self.z += action[1] * self.dt

        position = [self.x, self.y, self.z]
        pose = self.client.simGetVehiclePose()
        pose.position.x_val = position[0]
        pose.position.y_val = position[1]
        pose.position.z_val = - position[2]
        pose.orientation = airsim.to_quaternion(0, 0, self.yaw_rad)
        self.client.simSetVehiclePose(pose, False)

    def _get_obs(self):
        '''
        @description: get depth image and target features for navigation
        @param {type}
        @return:
        '''
        # 1. get current depth image 0-255 uint8 
        image = self._get_depth_image()
        image_resize = cv2.resize(image, (self.screen_width, self.screen_height))
        # np.save('1-img_ori', image)
        image_scaled = image_resize * 100
        self.min_distance_to_obstacles = image_scaled.min()
        # print("[get_obs] min_dist: {:.2f}".format(self.min_distance_to_obstacles))
        image_scaled = -np.clip(image_scaled, 0, 20) / 20 * 255 + 255  # 0-255  0-20m 255-0m

        # np.save('2-img_scaled', image_scaled)

        image_uint8 = image_scaled.astype(np.uint8)

        # cv2.imshow('test', image_uint8)
        # cv2.waitKey(1)
        # np.save('3-img_gray', image_uint8)
        assert image_uint8.shape[0] == self.screen_height and image_uint8.shape[1] == self.screen_width, 'image size not match'
        # image_uint8 = image_uint8.reshape(self.screen_height, self.screen_width, 1)
        state_feature_array = np.zeros((self.screen_height, self.screen_width))

        # 2. get current state (position, goal_pose, velocity)
        state_feature = self._get_state_feature()

        assert (self.state_feature_length == state_feature.shape[0]), 'state_lenght {0} is not equal to state_feature_length {1}' \
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

        if not done:
            distance_now = self.get_distance_from_desired_point_3d()
            reward_distance = (self.previous_distance_from_des_point - distance_now)
            self.previous_distance_from_des_point = distance_now

            reward_obs = 0

            # add yaw_rate cost
            yaw_speed_cost = 0.2 * abs(action[-1]) / self.max_vel_yaw_rad
            action_cost = yaw_speed_cost

            reward = reward_distance - reward_obs - action_cost
        else:
            if self.is_in_desired_pose():
                reward = reward_reach
            if self.is_crashed():
                reward = reward_crash

        return reward

    def _is_done(self):
        episode_done = False

        is_inside_workspace_now = self.is_inside_workspace()
        has_reached_des_pose    = self.is_in_desired_pose()
        too_close_to_obstable   = self.is_crashed()

        # We see if we are outside the Learning Space
        episode_done = not(is_inside_workspace_now) or\
                        has_reached_des_pose or\
                        too_close_to_obstable or\
                        self.step_num >= self.max_episode_steps
    
        return episode_done

    def _get_depth_image(self):

        # png_image = self.client.simGetImage("0", airsim.ImageType.Scene)

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
        # depth_img = np.zeros([200, 200])

        return depth_img

    def _get_state_feature(self):
        '''
        @description: update and get current uav state and state_norm 
        @param {type} 
        @return: state_norm
                    normalized state range 0-255
        '''
        distance = self._get_2d_distance_to_goal()
        relative_yaw = self._get_relative_yaw()  # return relative yaw -pi to pi 
        relative_pose_z = self.z - self.goal_position[2]  # current position z is positive
        vertical_distance_norm = (relative_pose_z / self.max_vertical_difference / 2 + 0.5) * 255

        distance_norm = distance / self.goal_distance * 255
        relative_yaw_norm = (relative_yaw / math.pi / 2 + 0.5 ) * 255

        # current speed and angular speed
        linear_velocity_xy = self.v_xy
        linear_velocity_norm = linear_velocity_xy / self.max_vel_x * 255
        linear_velocity_z = self.v_z
        linear_velocity_z_norm = (linear_velocity_z / self.max_vel_z / 2 + 0.5) * 255
        angular_velocity_norm = (self.yaw_rate / self.max_vel_yaw_rad / 2 + 0.5) * 255

        if self.navigation_3d:
            # state: distance_h, distance_v, relative yaw, velocity_x, velocity_z, velocity_yaw
            self.state_raw = np.array([distance, relative_pose_z,  math.degrees(relative_yaw), linear_velocity_xy, linear_velocity_z,  math.degrees(self.yaw_rate)])
            state_norm = np.array([distance_norm, vertical_distance_norm, relative_yaw_norm, linear_velocity_norm, linear_velocity_z_norm, angular_velocity_norm])
            state_norm = np.clip(state_norm, 0, 255)
            self.state_norm = state_norm
        else:
            self.state_raw = np.array([distance, math.degrees(relative_yaw), linear_velocity_xy,  math.degrees(self.yaw_rate)])
            state_norm = np.array([distance_norm, relative_yaw_norm, linear_velocity_norm, angular_velocity_norm])
            state_norm = np.clip(state_norm, 0, 255)
            self.state_norm = state_norm
    
        return state_norm
    
    def _set_goal_pose(self):
        distance = self.goal_distance
        noise = np.random.random() * 2 - 1
        angle = noise * math.radians(self.goal_angle_noise_degree)
        goal_x = distance * math.cos(angle)
        goal_y = distance * math.sin(angle)
        goal_z = 5

        self.goal_position = np.array([goal_x, goal_y, goal_z])

    def _get_2d_distance_to_goal(self):
        current_pose = np.array([self.x, self.y, self.z])
        relative_pose = current_pose - self.goal_position
        return math.sqrt(pow(relative_pose[0], 2) + pow(relative_pose[1], 2))

    def get_distance_from_desired_point_3d(self):
        current_pose = [self.x, self.y, self.z]
        relative_pose = current_pose - self.goal_position
        return math.sqrt(pow(relative_pose[0], 2) + pow(relative_pose[1], 2) + pow(relative_pose[2], 2))

    def _get_relative_yaw(self):
        '''
        @description: get relative yaw from current pose to goal in radian
        @param {type} 
        @return: 
        '''
        # get relative angle
        relative_pose_x = self.goal_position[0] - self.x
        relative_pose_y = self.goal_position[1] - self.y
        angle = math.atan2(relative_pose_y, relative_pose_x)

        # get current yaw
        yaw_current = self.yaw_rad

        # get yaw error
        yaw_error = angle - yaw_current
        if yaw_error > math.pi:
            yaw_error -= 2*math.pi
        elif yaw_error < -math.pi:
            yaw_error += 2*math.pi

        return yaw_error

    def is_inside_workspace(self):
        """
        Check if the Drone is inside the Workspace defined
        """
        is_inside = True
        current_position = [self.x, self.y, self.z]

        if max(abs(current_position[0]), abs(current_position[1])) > self.work_space_xy_max or \
                                               current_position[2] > self.work_space_z_max or \
                                               current_position[2] < self.work_space_z_min:
            is_inside = False

        return is_inside

    def is_in_desired_pose(self):
        in_desired_pose = False
        if self.get_distance_from_desired_point_3d() < self.accept_radius:
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
        self.client.simPrintLogMessage('Feature_norm: ', str(self.state_norm))
        self.client.simPrintLogMessage('Feature_raw: ', str(self.state_raw))
