import airsim
import numpy as np
import math
from gym import spaces

class SimpleDynamics():
    '''
    A simple dynamics used for vision based navigation
    It has position (x, y, z, yaw) in local frame and v_xy v_z yaw_rate as states
    '''
    def __init__(self) -> None:
        
        # config
        self.navigation_3d = True
        self.control_acc = True
        self.dt = 0.1

        # AirSim Client
        self.client = airsim.VehicleClient()
        self.client.confirmConnection()

        # start and goal position
        self.start_position = None
        self.start_random_angle = None
        self.goal_position = None
        self.goal_distance = None
        self.goal_random_angle = None
        
        # states
        self.x = 0
        self.y = 0
        self.z = 0
        self.v_xy = 0
        self.v_z = 0
        self.yaw = 0
        self.yaw_rate = 0

        self.max_acc_xy = 5.0
        self.max_vel_x = 5.0
        self.min_vel_x = 1.0
        self.max_vel_z = 2.0
        self.max_vel_yaw_deg = 50.0
        self.max_vel_yaw_rad = math.radians(self.max_vel_yaw_deg)
        self.max_vertical_difference = 5
        
        if self.navigation_3d:
            self.state_feature_length = 6
            if self.control_acc:
                self.action_space = spaces.Box(low=np.array([-self.max_acc_xy, -self.max_vel_z, -self.max_vel_yaw_rad]), \
                                               high=np.array([self.max_acc_xy, self.max_vel_z, self.max_vel_yaw_rad]), dtype=np.float32)
            else:
                self.action_space = spaces.Box(low=np.array([self.min_vel_x , -self.max_vel_z, -self.max_vel_yaw_rad]), \
                                            high=np.array([self.max_vel_x, self.max_vel_z, self.max_vel_yaw_rad]), dtype=np.float32)
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


    def reset(self):
        # reset goal
        self._set_goal_pose()
        # reset start
        yaw_noise = self.start_random_angle
        self.x = self.start_position[0]
        self.y = self.start_position[1]
        self.z = self.start_position[2]
        self.yaw_rad = yaw_noise  # TODO: test if 360 works
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

    def set_action(self, action):
        # update dynamics
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
            self.v_z = action[1]

        # set airsim pose
        position = self.get_position()
        pose = self.client.simGetVehiclePose()
        pose.position.x_val = position[0]
        pose.position.y_val = position[1]
        pose.position.z_val = - position[2]
        pose.orientation = airsim.to_quaternion(0, 0, self.yaw_rad)
        self.client.simSetVehiclePose(pose, False)

        return 0

    def _set_goal_pose(self):
        distance = self.goal_distance
        noise = np.random.random() * 2 - 1
        angle = noise * self.goal_random_angle
        goal_x = distance * math.cos(angle) + self.start_position[0]
        goal_y = distance * math.sin(angle) + self.start_position[1]
        goal_z = 5

        self.goal_position = [goal_x, goal_y, goal_z]
        print(self.goal_position)

    def set_start(self, position, random_angle):
        self.start_position = position
        self.start_random_angle = random_angle
    
    def set_goal(self, distance, random_angle):
        self.goal_distance = distance
        self.goal_random_angle = random_angle

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

    def _get_relative_yaw(self):
        '''
        @description: get relative yaw from current pose to goal in radian
        @param {type} 
        @return: 
        '''
        current_position = self.get_position()
        # get relative angle
        relative_pose_x = self.goal_position[0] - current_position[0]
        relative_pose_y = self.goal_position[1] - current_position[1]
        angle = math.atan2(relative_pose_y, relative_pose_x)

        # get current yaw
        yaw_current = self.get_attitude()[2]

        # get yaw error
        yaw_error = angle - yaw_current
        if yaw_error > math.pi:
            yaw_error -= 2*math.pi
        elif yaw_error < -math.pi:
            yaw_error += 2*math.pi

        return yaw_error

    def get_states_all(self):
        return [self.x, self.y, self.z, self.v_xy, self.v_z, self.yaw, self.yaw_rate]

    def get_position(self):
        return [self.x, self.y, self.z]

    def get_velocity(self):
        return [self.v_xy, self.v_z, self.yaw_rate]

    def get_attitude(self):
        # return current euler angle
        return [0, 0, self.yaw]

    def _get_2d_distance_to_goal(self):
        return math.sqrt(pow(self.get_position()[0] - self.goal_position[0], 2) + pow(self.get_position()[1] - self.goal_position[1], 2))