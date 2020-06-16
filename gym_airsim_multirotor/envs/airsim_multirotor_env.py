'''
@Author: Lei He
@Date: 2020-05-29 16:54:52
@LastEditTime: 2020-06-09 22:25:20
@FilePath: \gym_airsim_multirotor\gym_airsim_multirotor\envs\airsim_multirotor_env.py
@Description: Gym like environemnt for AirSim. Using for 3D navigation.
@Github: https://github.com/heleidsn
'''
import math
import time
import keyboard

import airsim
import cv2
import gym
import numpy as np

from configparser import ConfigParser

from gym import error, spaces, utils
from gym.utils import seeding

from PyQt5 import QtCore
from PyQt5.QtCore import pyqtSignal

class AirsimMultirotor(gym.Env, QtCore.QThread):
    # metadata = {'render.modes': ['human']}
    action_signal = pyqtSignal(int, np.ndarray, np.ndarray, np.ndarray) # vel_x, vel_z, vel_yaw
    state_signal = pyqtSignal(int, np.ndarray) # state_raw
    state_norm_signal = pyqtSignal(int, np.ndarray)
    reward_signal = pyqtSignal(int, float, float) # step_reward, total_reward
    depth_image_signal = pyqtSignal(np.ndarray) # depth_image
    feature_signal = pyqtSignal(np.ndarray)

    def __init__(self):
        super(AirsimMultirotor, self).__init__()
        print('init AirsimMultirotor environment')

        # init airsim client
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        # training state
        self.episode_num = 0
        self.step_num = 0
        self.total_step = 0
        self.cumulated_episode_reward = 0
        self.last_obs = 0
        self.previous_distance_from_des_point = 0
        self.feature_before_fc = 0

        # uav state
        self.goal_position = np.zeros(3)
        self.state_current_pose = np.zeros(3)
        self.state_current_attitude = np.zeros(3)
        self.state_current_velocity = np.zeros(3)
        self.min_distance_to_obstacles = 0
        self.vel_cmd_final = np.zeros(3)
        print('finish init environment')


    def set_config(self, cfg):

        self.config_file_name = cfg.get('config', 'config_file_name')
        print('Using config file: ', self.config_file_name)

        # config
        self.reward_decompose = cfg.getboolean('config', 'reward_decompose')
        self.keyboard_debug = cfg.getboolean('config', 'keyboard_debug')

        # environment
        self.max_episode_steps = cfg.getint('environment', 'max_episode_steps')

        # uav_model
        self.acc_lim_x = cfg.getfloat('uav_model', 'acc_lim_x')
        self.acc_lim_z = cfg.getfloat('uav_model', 'acc_lim_z')
        self.acc_lim_yaw_rad = math.radians(cfg.getfloat('uav_model', 'acc_lim_yaw_deg'))

        self.max_vel_x = cfg.getfloat('uav_model', 'max_vel_x')
        self.min_vel_x = cfg.getfloat('uav_model', 'min_vel_x')
        self.max_vel_z = cfg.getfloat('uav_model', 'max_vel_z')
        self.max_vel_yaw_rad = math.radians(cfg.getfloat('uav_model', 'max_vel_yaw_deg'))

        # reward
        self.reward_coeff_obstacle_distance = cfg.getfloat('reward', 'reward_coeff_obstacle_distance')
        self.reward_coeff_action_cost = cfg.getfloat('reward', 'reward_coeff_action_cost')
        self.reward_coeff_position_cost_up = cfg.getfloat('reward', 'reward_coeff_positon_cost_up')
        self.reward_coeff_position_cost_down = cfg.getfloat('reward', 'reward_coeff_positon_cost_down')
        self.reward_coeff_position_cost_yaw = cfg.getfloat('reward', 'reward_coeff_position_cost_yaw')

        # goal
        self.goal_distance = cfg.getint('goal', 'goal_distance')
        self.goal_angle_noise_degree = cfg.getint('goal', 'goal_angle_noise_degree')
        
        # work space
        self.work_space_xy_max = self.goal_distance + cfg.getint('work_space', 'work_space_xy_padding')
        self.work_space_z_max = cfg.getint('work_space', 'work_space_z_max')
        self.work_space_z_min = cfg.getint('work_space', 'work_space_z_min')

        self.takeoff_hight = cfg.getint('control', 'takeoff_hight')
        self.accept_radius = cfg.getint('goal', 'goal_accept_radius')

        # input image
        self.screen_height = cfg.getint('input_image', 'image_height')
        self.screen_width = cfg.getint('input_image', 'image_width')
        self.fov_h_degrees = cfg.getint('input_image', 'fov_horizontal_degrees')
        self.fov_v_degrees = cfg.getint('input_image', 'fov_vertical_degrees')

        # control
        self.time_for_control_second = cfg.getfloat('control', 'control_time_interval')
        self.max_vertical_difference = cfg.getfloat('control', 'max_vertical_difference')
        self.rotate_speed_max = math.radians(cfg.getfloat('control', 'rotate_speed_max_degrees'))
        self.distance_to_obstacles_accept = cfg.getint('control', 'distance_to_obstacles_accept')
        self.distance_to_obstacles_punishment = cfg.getint('control', 'distance_to_obstacles_punishment')

        self.debug_mode = cfg.getboolean('control', 'debug_mode')

        self.navigation_3d = cfg.getboolean('config', 'navigation_3d')


        # observation and action space
        self.observation_space = spaces.Box(low=0, high=255, \
                                            shape=(self.screen_height, self.screen_width, 2),\
                                            dtype=np.uint8)
 
        if self.navigation_3d:
            self.state_feature_lenght = 6
            self.state_raw = np.zeros(self.state_feature_lenght)
            self.state_norm  = np.zeros(self.state_feature_lenght)
            self.action_space = spaces.Box(low=np.array([self.min_vel_x , -self.max_vel_z, -self.max_vel_yaw_rad]), \
                                        high=np.array([self.max_vel_x, self.max_vel_z, self.max_vel_yaw_rad]), \
                                        dtype=np.float32)
        else:
            self.state_feature_lenght = 4
            self.state_raw = np.zeros(self.state_feature_lenght)
            self.state_norm = np.zeros(self.state_feature_lenght)
            self.action_space = spaces.Box(low=np.array([self.min_vel_x , -self.max_vel_yaw_rad]), \
                                        high=np.array([self.max_vel_x, self.max_vel_yaw_rad]), \
                                        dtype=np.float32)


    def step(self, action):
        # set action
        if self.navigation_3d:
            self._set_action_3d(action)
        else:
            self._set_action_2d(action)

        self.step_num += 1
        self.total_step += 1
        
        # get observation
        obs = self._get_obs()
        done = self._is_done(obs)
        info = {
            'is_success': self.is_in_desired_pose(),
            'is_crash': self.is_crashed() or not self.is_inside_workspace(),
            'step_num': self.step_num
        }
        if self.navigation_3d:
            reward, reward_split = self._compute_reward_norm(obs, done, action)
        else:
            reward, reward_split = self._compute_reward(obs, done, action)

        self.cumulated_episode_reward += reward
        self.info_display(action, obs, done, reward)

        self.action_signal.emit(int(self.total_step), action, self.vel_cmd_final, self.state_raw)
        self.state_signal.emit(int(self.total_step), self.state_raw)
        self.state_norm_signal.emit(int(self.total_step), self.state_norm)
        self.reward_signal.emit(int(self.total_step), reward, self.cumulated_episode_reward)
        

        if self.keyboard_debug:
            action_deg = np.copy(action)
            action_deg[2] = math.degrees(action_deg[2])
            state_deg = np.copy(self.state_raw)
            state_deg[2] = math.degrees(state_deg[2])
            state_deg[5] = math.degrees(state_deg[5])
            np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
            print('=============================================================================')
            print('episode', self.episode_num, 'step', self.step_num, 'total step', self.total_step)
            print('action', action_deg)
            print('state', state_deg)
            print('state_norm', self.state_norm)
            print('reward {:.3f} {:.3f}'.format(reward, self.cumulated_episode_reward))
            print('done', done)
            keyboard.wait('a')

        self.last_obs = obs

        if self.reward_decompose:
            return obs, reward, done, info, reward_split
        else:
            return obs, reward, done, info      

    def reset(self):
        self._reset_sim()
        self._init_env_variables()
        self._update_episode()
        obs = self._get_obs()
        self.last_obs = obs
        return obs

    def render(self, mode='human'):
        print('render')

    def close(self):
        print('close')

# Methods for costum environment
    # -----------------------------------
    def _reset_sim(self):
        self.client.simPause(False)
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)

        self.client.moveToZAsync(-self.takeoff_hight, 2).join()
        current_pos = self.client.simGetVehiclePose().position
        while current_pos.z_val > -2:
            print("reset fail...")
            self.client.reset()
            self.client.enableApiControl(True)
            self.client.armDisarm(True)
            self.client.moveToZAsync(-self.takeoff_hight, 2).join()
            current_pos = self.client.simGetVehiclePose().position

        self.client.simPause(True)

    def _init_env_variables(self):
        self._change_goal_pose_random()
        # self._change_goal_for_NH_env()
        self.step_num = 0
        self.previous_distance_from_des_point = self.goal_distance

    def _update_episode(self):
        self.episode_num += 1
        self.cumulated_episode_reward = 0

    def _get_obs(self):
        '''
        @description: get depth image and target features for navigation
        @param {type}
        @return:
        '''
        # 1. get current depth image 0-255 uint8 
        
        image = self.get_depth_image()
        image_scaled = image * 100
        self.min_distance_to_obstacles = image_scaled.min()
        image_scaled = -np.clip(image_scaled, 0, 20) / 20 * 255 + 255  # 0-255  0-20m 255-0m

        image_uint8 = image_scaled.astype(np.uint8)
        assert image_uint8.shape[0] == self.screen_height and image_uint8.shape[1] == self.screen_width, 'image size not match'
        # image_uint8 = image_uint8.reshape(self.screen_height, self.screen_width, 1)
        state_feature_array = np.zeros((self.screen_height, self.screen_width))

        # 2. get current state (position, goal_pose, velocity)
        state_feature = self._get_state_feature()

        assert (self.state_feature_lenght == state_feature.shape[0]), 'state_lenght {0} is not equal to state_feature_length {1}' \
                                                                    .format(self.state_feature_lenght, state_feature.shape[0])
        state_feature_array[0, 0:self.state_feature_lenght] = state_feature

        # 3. generate image with state
        image_with_state = np.array([image_uint8, state_feature_array])
        image_with_state = image_with_state.swapaxes(0, 2)
        image_with_state = image_with_state.swapaxes(0, 1)
        
        return image_with_state

    def _set_action_3d(self, action):
        '''
        @description: set action for 3D navigation
        @param {type}:  action[0]: forward velocity
                        action[1]: vertical velocity
                        action[2]: yaw velocity
        @return: 
        '''
        self.client.simPause(False)

        # get actions
        cmd_vel_x = float(action[0])
        cmd_vel_z = float(action[1])
        cmd_yaw_rate = float(action[2])
        
        # get current state
        current_vel_x = self.state_raw[3]
        current_vel_z = self.state_raw[4]
        current_vel_yaw = self.state_raw[5]

        current_yaw = self.get_current_attitude_radian()[2]

        # acc limitation
        vel_x_range = np.array([current_vel_x - self.acc_lim_x * self.time_for_control_second, current_vel_x + self.acc_lim_x * self.time_for_control_second])
        vel_z_range = np.array([current_vel_z - self.acc_lim_z * self.time_for_control_second, current_vel_z + self.acc_lim_z * self.time_for_control_second])
        vel_yaw_range = np.array([current_vel_yaw - self.acc_lim_yaw_rad * self.time_for_control_second, current_vel_yaw + self.acc_lim_yaw_rad * self.time_for_control_second])
        cmd_vel_x_new = np.clip(cmd_vel_x, vel_x_range[0], vel_x_range[1])
        cmd_vel_z_new = np.clip(cmd_vel_z, vel_z_range[0], vel_z_range[1])
        cmd_yaw_rate_new = np.clip(cmd_yaw_rate, vel_yaw_range[0], vel_yaw_range[1])

        # velocity limitation
        cmd_vel_x_final = np.clip(cmd_vel_x_new, self.min_vel_x, self.max_vel_x)
        cmd_vel_z_final = np.clip(cmd_vel_z_new, -self.max_vel_z, self.max_vel_z)
        cmd_yaw_rate_final = np.clip(cmd_yaw_rate_new, -self.max_vel_yaw_rad, self.max_vel_yaw_rad)

        # print('action set===============')
        # print('action {:.3f} {:.3f} {:.3f}'.format(cmd_vel_x, cmd_vel_z, math.degrees(cmd_yaw_rate)))
        # print('current vel {:.3f} {:.3f} {:.3f}'.format(current_vel_x, current_vel_z, math.degrees(current_vel_yaw)))
        # print('acc limit {:.3f} {:.3f} {:.3f}'.format(cmd_vel_x_new, cmd_vel_z_new, math.degrees(cmd_yaw_rate_new)))
        # print('cmd final {:.3f} {:.3f} {:.3f}'.format(cmd_vel_x_final, cmd_vel_z_final, math.degrees(cmd_yaw_rate_final)))

        self.vel_cmd_final = np.array([cmd_vel_x_final, cmd_vel_z_final, cmd_yaw_rate_final])

        '''
        # FoV limitation
        speed_scale = (self.fov_h_degrees - 2*math.degrees(abs(yaw_rate))) / self.fov_h_degrees
        speed_scale = max(0, speed_scale)
        speed_scale = 1
        if self.debug_mode:
            print('[_set_action]', 'yaw_rate_cmd: ', yaw_rate, 'speed_scale:', speed_scale) 
        '''
        
        # transfer dx dy from body frame to local frame
        vx_body = cmd_vel_x_final
        vy_body = 0
        vz_body = cmd_vel_z_final
        vx_local, vy_local = self.point_transfer(vx_body, vy_body, -current_yaw)

        # set 3d navigation action      
        self.client.moveByVelocityAsync(vx_local, vy_local, -vz_body, self.time_for_control_second,
                                            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                                            yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=math.degrees(cmd_yaw_rate_final))).join()
                          
        self.client.simPause(True)

    def _set_action_2d(self, action):
        '''
        @description: set action to airsim client
        @param {type}:
            action[0]: forward velocity
            action[1]: yaw angular velocity
        @return: None
        '''
        self.client.simPause(False)

        # get actions
        target_forward_speed = float(action[0])
        yaw_rate = float(action[-1])

        # add FoV speed constraint
        speed_scale = (self.fov_h_degrees - 2*math.degrees(abs(yaw_rate))) / self.fov_h_degrees
        speed_scale = max(0, speed_scale)
        speed_scale = 1
        if self.debug_mode:
            print('[_set_action]', 'yaw_rate_cmd: ', yaw_rate, 'speed_scale:', speed_scale)
        
        # get current states
        current_yaw = self.get_current_attitude_radian()[2]
        yaw_setpoint = current_yaw + yaw_rate

        # transfer dx dy from body frame to local frame
        vx_body = target_forward_speed * speed_scale
        vy_body = 0
        vx_local, vy_local = self.point_transfer(vx_body, vy_body, -yaw_setpoint)
                 
        # set 2d navigation action
        self.client.moveByVelocityZAsync(vx_local, vy_local, -self.takeoff_hight, self.time_for_control_second,
                                             drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
                                             yaw_mode=airsim.YawMode(is_rate=True, yaw_or_rate=math.degrees(yaw_rate))).join()                    

        self.client.simPause(True)

    def _is_done(self, obs):
        """
        The done can be done due to three reasons:
        1) It went outside the workspace
        2) It crashed
        3) It has reached the desired point
        """

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

    def _compute_reward(self, obs, done, action):
        '''
        reward consists of:
            1. sparse part: reach, crash, step punishment
            2. continuous part: 
                distance to reward
                distance to obstacle punishment
        '''
        reward = 0
        reward_split = [0, 0, 0]

        # reward_reach = 10
        reward_reach = 0
        reward_crash = 0
        # reward_crash = -10
        reward_coeff_distance = 20
        reward_coeff_obstacle_distance = 0.5  # set punishment for too close to the obstacles
        # reward_step_punishment = 0.05
        # get distance from goal position
        current_pose = self.get_current_pose()
        distance_now = self.get_distance_from_desired_point(current_pose)

        if not done:
            # normalized distance to goal reward  
            reward_distance = (self.previous_distance_from_des_point - distance_now) / self.goal_distance * reward_coeff_distance
            # distance to obstacle punishment
            reward_obs = 0
            if self.min_distance_to_obstacles < self.distance_to_obstacles_punishment:
                reward_obs = reward_coeff_obstacle_distance * (self.min_distance_to_obstacles-self.distance_to_obstacles_punishment) \
                         / (self.distance_to_obstacles_accept - self.distance_to_obstacles_punishment)

            # step punishment
            # reward = reward_distance - reward_obs - reward_step_punishment
            reward = reward_distance - reward_obs
            self.previous_distance_from_des_point = distance_now
            reward_split = [reward_distance, -reward_obs, reward]
            # reward_split = [reward, 0, reward]

            # msg = 'reward_distanc: {} reward_obs: {} reward_step: {}'.format(reward_distance, reward_obs, reward_step_punishment)
            # self.client.simPrintLogMessage('debug: ', msg)

        else:
            if self.is_in_desired_pose():
                reward = reward_reach
                reward_split = [10, 0, 10]
                # reward_split = [reward, 0, 0]
            if self.is_crashed():
                reward = reward_crash
                reward_split = [0, -10, -10]
                # reward_split = [reward, -10, 0]

        return reward, reward_split

    def _compute_reward_3d(self, obs, done, action):
        '''
        reward consists of:
            1. sparse part: reach, crash, step punishment
            2. continuous part: 
                distance to reward
                distance to obstacle punishment
        '''
        reward = 0
        reward_split = [0, 0, 0]

        # reward_reach = 10
        reward_reach = 0
        reward_crash = 0
        # reward_crash = -10
        reward_coeff_distance = 40
        reward_coeff_obstacle_distance = 0.5  # set punishment for too close to the obstacles
        # reward_step_punishment = 0.05
        # get distance from goal position
        current_pose = self.get_current_pose()
        distance_now = self.get_distance_from_desired_point(current_pose)

        if not done:
            # normalized distance to goal reward  
            # reward_distance = (self.previous_distance_from_des_point - distance_now) / self.goal_distance * reward_coeff_distance
            reward_distance = (self.previous_distance_from_des_point - distance_now)

            # distance to obstacle cost
            reward_obs = 0
            if self.min_distance_to_obstacles < self.distance_to_obstacles_punishment:
                reward_obs = reward_coeff_obstacle_distance * (self.min_distance_to_obstacles-self.distance_to_obstacles_punishment) \
                         / (self.distance_to_obstacles_accept - self.distance_to_obstacles_punishment)

            # step punishment
            # reward = reward_distance - reward_obs - reward_step_punishment

            # action cost
            action_cost = 0

            if action[1] < 0:
                vertical_speed_cost = 0.1 * abs(action[1]) / self.max_vel_z
            else:
                vertical_speed_cost = 0.05 * abs(action[1]) / self.max_vel_z

            yaw_speed_cost = 0.2 * abs(action[2]) / self.max_vel_yaw_rad
            action_cost = vertical_speed_cost + yaw_speed_cost

            reward = reward_distance - reward_obs - action_cost

            self.previous_distance_from_des_point = distance_now
            reward_split = [reward_distance, -reward_obs, reward]
        else:
            if self.is_in_desired_pose():
                reward = reward_reach
                reward_split = [10, 0, 0]
            if self.is_crashed():
                reward = reward_crash
                reward_split = [0, -10, 0]

        if self.debug_mode:
            print('action: {:.4f} {:.4f} {:.4f}'.format(action[0], action[1], action[2]))
            print('reward:{:.4f} action_cost:{:.4f} vertical_speed_cost:{:.4f}'.format(reward, action_cost, vertical_speed_cost))

        return reward, reward_split

    def _compute_reward_norm(self, obs, done, action):
        '''
        @description: get a normalized reward function
        @param {type} 
        @return: 
        '''
        reward = 0
        reward_split = [0, 0, 0]
        
        current_pose = self.get_current_pose()
        distance_now = self.get_distance_from_desired_point(current_pose)
        
        if not done:
            # 1. distance reward
            reward_distance = np.clip((self.previous_distance_from_des_point - distance_now) / self.time_for_control_second / self.max_vel_x, 0, 1)
            reward_distance = math.sqrt(reward_distance)

            # 2. distance to obstacle cost
            obs_cost = 0
            if self.min_distance_to_obstacles < self.distance_to_obstacles_punishment:
                obs_cost = self.reward_coeff_obstacle_distance * (self.min_distance_to_obstacles-self.distance_to_obstacles_punishment) \
                         / (self.distance_to_obstacles_accept - self.distance_to_obstacles_punishment)

            # 3. action cost
            action_cost = 0
            vertical_speed_cost = self.reward_coeff_action_cost * abs(action[1]) / self.max_vel_z
            yaw_speed_cost = self.reward_coeff_action_cost * abs(action[2]) / self.max_vel_yaw_rad
            action_cost = vertical_speed_cost + yaw_speed_cost

            # 4. position cost
            position_cost = 0
            vertical_distance = self.state_raw[1]
            relative_yaw = self.state_raw[2]

            if vertical_distance < 0:
                vertical_cost = self.reward_coeff_position_cost_down * abs(vertical_distance)
            else:
                vertical_cost = self.reward_coeff_position_cost_down * abs(vertical_distance)
                
            yaw_cost = self.reward_coeff_position_cost_yaw * np.clip((abs(relative_yaw) / math.radians(90)), 0, 1)
            position_cost = vertical_cost + yaw_cost

            reward = reward_distance - obs_cost - action_cost - position_cost

            reward = np.clip(reward, 0, 1)

            self.previous_distance_from_des_point = min(distance_now, self.previous_distance_from_des_point)
        else:
            if self.is_in_desired_pose():
                reward = 30
            else:
                reward = 0

        return reward, reward_split

    def _change_goal_pose(self, goal_x, goal_y, goal_z):
        self.goal_position = np.array([goal_x, goal_y, goal_z])

    def _get_state_feature(self):
        '''
        @description: update and get current uav state and state_norm 
        @param {type} 
        @return: state_norm
                    normalized state range 0-255
        '''
        current_pose = self.get_current_pose()
        distance = self.get_distance_from_desired_point(current_pose)
        relative_yaw = self._get_ralative_yaw(current_pose, self.goal_position)

        relative_pose_z = -(self.goal_position[2] + current_pose[2])  # current position z is negative
        vertical_distance_norm = (relative_pose_z / self.max_vertical_difference / 2 + 0.5) * 255

        distance_norm = distance / self.goal_distance * 255
        relative_yaw_norm = (relative_yaw / math.pi / 2 + 0.5 ) * 255

        # current speed and angular speed
        groudtruth = self.client.simGetGroundTruthKinematics()
        linear_velocity = groudtruth.linear_velocity
        angular_velocity = groudtruth.angular_velocity
        linear_velocity_xy = math.sqrt(linear_velocity.x_val ** 2 + linear_velocity.y_val ** 2)
        linear_velocity_norm = linear_velocity_xy / self.max_vel_x * 255
        linear_velocity_z = -linear_velocity.z_val
        linear_velocity_z_norm = (linear_velocity_z / self.max_vel_z / 2 + 0.5) * 255
        angular_velocity_norm = (angular_velocity.z_val / self.max_vel_yaw_rad / 2 + 0.5) * 255

        if self.navigation_3d:
            # state: distance_h, distance_v, relative yaw, velocity_x, velocity_z, velocity_yaw
            self.state_raw = np.array([distance, relative_pose_z, relative_yaw, linear_velocity_xy, linear_velocity_z, angular_velocity.z_val])
            state_norm = np.array([distance_norm, vertical_distance_norm, relative_yaw_norm, linear_velocity_norm, linear_velocity_z_norm, angular_velocity_norm])
            state_norm = np.clip(state_norm, 0, 255)
            self.state_norm = state_norm
        else:
            self.state_raw = np.array([distance, relative_yaw, linear_velocity_xy, angular_velocity.z_val])
            state_norm = np.array([distance_norm, relative_yaw_norm, linear_velocity_norm, angular_velocity_norm])
            state_norm = np.clip(state_norm, 0, 255)
            self.state_norm = state_norm
    
        return state_norm
        

# Some useful methods
    # --------------------------------------------
    def get_distance_from_desired_point(self, current_pose):
        relative_pose = current_pose - self.goal_position
        return math.sqrt(pow(relative_pose[0], 2) + pow(relative_pose[1], 2))

    def get_distance_between_points(self, p_start, p_end):
        """
        Given a Vector3 Object, get distance from current position
        :param p_end:
        :return:
        """
        return np.linalg.norm(p_start, p_end)

    def is_inside_workspace(self):
        """
        Check if the Drone is inside the Workspace defined
        """
        is_inside = True
        current_position = self.get_current_pose()

        if max(abs(current_position[0]), abs(current_position[1])) > self.work_space_xy_max or \
            -current_position[2] > self.work_space_z_max:
            is_inside = False

        return is_inside

    def is_in_desired_pose(self):
        in_desired_pose = False
        current_position = self.get_current_pose()
        if self.get_distance_from_desired_point(current_position) < self.accept_radius:
            in_desired_pose = True

        return in_desired_pose

    def is_crashed(self):
        is_crashed = False
        collision_info = self.client.simGetCollisionInfo()
        if collision_info.has_collided or self.min_distance_to_obstacles < self.distance_to_obstacles_accept:
            is_crashed = True

        return is_crashed

    def point_transfer(self, x, y, theta):
        '''
        @description: transfer x y from one frame to another frame
        @param {type}
        @return:
        '''
        # transfer x, y to another frame
        new_x = x * math.cos(theta) + y * math.sin(theta)
        new_y = - x * math.sin(theta) + y * math.cos(theta)

        return new_x, new_y


    def _get_ralative_yaw(self, current_pose, goal_pose):
        '''
        @description: get relative yaw from current pose to goal in radian
        @param {type} 
        @return: 
        '''
        # get relative angle
        relative_pose_x = goal_pose[0] - current_pose[0]
        relative_pose_y = goal_pose[1] - current_pose[1]
        angle = math.atan2(relative_pose_y, relative_pose_x)

        # get current yaw
        yaw_current = self.get_current_attitude_radian()[2]

        # get yaw error
        yaw_error = angle - yaw_current
        if yaw_error > math.pi:
            yaw_error -= 2*math.pi
        elif yaw_error < -math.pi:
            yaw_error += 2*math.pi

        return yaw_error
        
    def _change_goal_pose_random(self):
        distance = self.goal_distance

        noise = np.random.random() * 2 - 1
        angle = noise * math.radians(self.goal_angle_noise_degree)

        goal_x = distance * math.cos(angle)
        goal_y = distance * math.sin(angle)

        self.goal_position = np.array([goal_x, goal_y, self.takeoff_hight])
        # print(self.goal_position)

    def _change_goal_for_NH_env(self):
        goal_list = [[120, 0, 5],
                     [0, 120, 5],
                     [-110, 0, 5],
                     [0, -120, 5],
                     [-175, 42, 5],
                     [-82, 80, 5],
                     [62, -10, 5],
                     [-30, 48, 10],
                     [35, -102, 10],
                     [-145, -110, 10]]

        # index = random.randint(0, 9)
        index = self.episode_num % 10
        print('episode: ', self.episode_num + 1, 'goal: ', goal_list[index])
        self.goal_position = np.array(goal_list[index])


# Gets method
    def get_current_attitude_radian(self):
        self.state_current_attitude = self.client.simGetVehiclePose().orientation
        return airsim.to_eularian_angles(self.state_current_attitude)

    def get_current_pose(self):
        pose = self.client.simGetVehiclePose().position
        self.state_current_pose = np.array([pose.x_val, pose.y_val, pose.z_val])
        return self.state_current_pose

    # def get_current_velocity(self):
    #     self.state_current_velocity = 

    def get_depth_image(self):
        responses = self.client.simGetImages([
            airsim.ImageRequest("0", airsim.ImageType.DepthVis, True, False)
            ])

        # depth_img = airsim.list_to_2d_float_array(responses[0].image_data_float, responses[0].width, responses[0].height)

        # check observation
        while responses[0].width == 0:
            print("get_image_fail...")
            responses = self.client.simGetImages([
                airsim.ImageRequest("0", airsim.ImageType.DepthVis, True, False)
                ])

        depth_img = airsim.list_to_2d_float_array(responses[0].image_data_float, responses[0].width, responses[0].height)

        return depth_img

    def info_display(self, action, obs, done, reward):
        '''
        display information in airsim
        '''
        # current episode and step
        msg = "Episode: {} Step: {} Totoal step: {}".format(self.episode_num, self.step_num, self.total_step)
        self.client.simPrintLogMessage('1- ', msg)

        # state: obs, min_distance
        msg = 'distance: {:.2f} relative_yaw: {:.2f} v_x: {:.2f} yaw_speed: {:.2f} distance to obs: {:.2f}'.format(self.state_raw[0], self.state_raw[1], self.state_raw[2], self.state_raw[3], self.min_distance_to_obstacles)
        self.client.simPrintLogMessage('2-: ', msg)
        msg = 'State_norm: ' + np.array2string(self.state_norm, precision=2)
        self.client.simPrintLogMessage('2-: ', msg)

        # action
        msg = 'linear speed: {:.2f} angular speed: {:.2f}'.format(action[0], math.degrees(action[1]))
        self.client.simPrintLogMessage('3-Action: ', msg)
        # reward
        msg = ':{:.2f} total reward: {:.2f}'.format(reward, self.cumulated_episode_reward)
        self.client.simPrintLogMessage('4-Reward: ', msg)

    def info_display_RD(self, out, reward_split):
        '''
        display the debug variable for reward decomposition
        '''
        # current q
        q_value = out[2]

        # feature
        feature = out[0][0]

        # action
        action = out[1][0]

        # policy dense bias
        policy_dense_bias = out[3][0]

        # network parameters
        msg = 'q-value: {:.2f} speed: {:.2f} angle: {:.2f}'.format(q_value, action[0], action[1])
        self.client.simPrintLogMessage('5-RD-debug- ', msg)
        # feature_show = np.around(feature, decimals=1)
        msg = np.array2string(feature, precision=2)
        # msg = np.fromstring(feature_show, dtype=int)
        self.client.simPrintLogMessage('6-RD-debug-feature: ', msg)

        msg = 'feature_max: {:.2f}'.format(feature.max())
        self.client.simPrintLogMessage('6-RD-debug-: ', msg)

        msg = np.array2string(policy_dense_bias, precision=5)
        self.client.simPrintLogMessage('7-RD-debug-bias: ', msg)

        min_target_q = out[4]
        q1_target = out[5]
        q2_target = out[6]
        msg = '{:.2f} Min_target_q: {:.2f} Total reward: {:.2f} q1_t: {:.2F} q2_t:{:.2f}'.format(q_value, min_target_q, self.cumulated_episode_reward, q1_target, q2_target)
        self.client.simPrintLogMessage('8-q:', msg )
        
        Q1 = out[7][0][0]
        Q2 = out[8][0][0]
        Q3 = out[9][0][0]
        Q_sum = Q1 + Q2 + Q3
        msg = 'Q1: {:.2f} Q2: {:.2f} Q3: {:.2f} Sum: {:.2f} Total: {:.2f}'.format(Q1, Q2, Q3, Q_sum, q_value)
        self.client.simPrintLogMessage('9-:', msg )

        msg = 'reward split: {:.2f} {:.2f} {:.2f}'.format(reward_split[0], reward_split[1], reward_split[2])
        self.client.simPrintLogMessage('10-', msg)

    def set_feature_before_fc(self, feature):
        self.feature_before_fc = feature
        self.feature_signal.emit(self.feature_before_fc)
        # print(feature) 
