#!/usr/bin/python3.6

import os
import time
import numpy as np
import copy
import csv
from math import sin, cos, tan
import rospy
import math
import matplotlib
from geometry_msgs.msg import Pose,Point, PoseStamped, Quaternion,PoseWithCovarianceStamped
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Path, Odometry
from std_msgs.msg import Duration, Header, String,Bool
from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Int32MultiArray

class SimInterface:
    def __init__(self):
        cmd_vel_topic = rospy.get_param('cmd_vel_topic_name', '/nav')
        odom_topic = rospy.get_param('odom_topic_name', '/odom')
        goal_topic = rospy.get_param('goal_topic_name', '/move_base_simple/goal')
        key_topic = rospy.get_param('key_topic_name', '/key')
        self.car_frame = rospy.get_param('car_frame', 'base_link')
        self.GOAL_THRESHOLD = rospy.get_param('goal_threshold', 0.50)
        self.DEBUG_MODE = rospy.get_param('debug_mode', True)

        dirname = os.path.dirname(__file__)
        path_folder_name = rospy.get_param('path_folder_name', 'track')
        self.OUTPUT_FILE_PATH = os.path.join(dirname, path_folder_name)
        self.CENTER_TRACK_FILENAME = os.path.join(dirname, path_folder_name + '/centerline_waypoints.csv')
        self.preprocess_track_data()

        self.min_vel = rospy.get_param('min_speed', -2.5)
        self.max_vel = rospy.get_param('max_speed', 2.5)
        self.max_steer_limit = rospy.get_param('max_steering', 0.418)
        self.min_steer_limit = rospy.get_param('min_steering', -0.418)
        self.L = rospy.get_param('car_length', 0.325)
        self.SAMPLING_SIZE = rospy.get_param('sample_size', 100)
        self.HORIZON_LENGTH = rospy.get_param('horizon_length', 20)
        self.ANGLE_STEP = rospy.get_param('angle_step', 18)
        self.dT = rospy.get_param('dT', 0.05)
        self.downsampled_ranges = None
        self.original_ranges = None
        self.LIDAR_DELTA_ANGLE_INCR = None
        self.LIDAR_ANGLE_MIN = None
        self.GRID_SIZE = 0.05
        self.Origin_X=75
        self.Origin_Y=50

        self.PROGRESS_REWARD_FACTOR = 1.0
        self.DEVIATION_REWARD_FACTOR = 2.0
        self.lap_count = 0
        self.COUNT_FLAG = False
        self.TRAINING_LAPS = 30
        self.STEP_TIME = 0.1

        self.current_pose = np.array([0.0, 0.0, 0.0])
        self.current_vel = None
        self.current_steering = None
        self.train_data = []
        self.trajectories = []
        self.train_images=[]
        self.test_images=[]

        self.current_state = None
        self.current_action = None
        self.current_obs = None
        self.current_full_obs=None
        self.reset_pose= None
        self.reset_image_state = None

        # Goal status related variables
        self.goal_pose = None
        self.goal_reached = False
        self.goal_received = False
        self.LIDAR_INIT = False
        self.NAVIGATION_ENABLE = False
        self.ACTION_INIT = False
        self.ODOM_INIT = False
        self.APPEND_MODE= False
        self.PROG_COMPLETE_FLAG = False
        self.COLLISION_FLAG= False
        self.NAVIGATION_MODE = 0

        # Publishers
        self.cmd_vel_pub = rospy.Publisher(cmd_vel_topic, AckermannDriveStamped, queue_size=10)
        self.keyboard_pub = rospy.Publisher(key_topic, String, queue_size=10)
        self.lap_pub = rospy.Publisher("turning_mode", String, queue_size=1)
        self.pose_pub = rospy.Publisher("initialpose",PoseWithCovarianceStamped,queue_size=1)

        # Subscribers
        rospy.Subscriber(odom_topic, Odometry, self.odom_callback, queue_size=1)
        rospy.Subscriber(goal_topic, PoseStamped, self.goal_callback, queue_size=1)
        rospy.Subscriber('/scan', LaserScan, self.lidarCB, queue_size=1)
        rospy.Subscriber('/action', AckermannDriveStamped, self.action_callback, queue_size=1)
        rospy.Subscriber('/collision_status', Bool, self.collision_callback, queue_size=1)
        rospy.Subscriber('/mux',Int32MultiArray,self.mux_callback,queue_size=1)


        # Timer callback function for the control loop
        # rospy.Timer(rospy.Duration(1.0 / self.CONTROLLER_FREQ), self.controlLoopCB)

    def collision_callback(self,msg):
        self.COLLISION_FLAG = msg.data

    def mux_callback(self,msg):
        self.NAVIGATION_MODE = msg.data[4]


    def create_header(self, frame_id):
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = frame_id
        return header

    def heading(self, yaw):
        qdata = Quaternion()
        qdata.x,qdata.y,qdata.z,qdata.w= self.euler_to_quaternion(0, 0, yaw)
        return qdata

    def quaternion_to_euler_yaw(self, orientation):
        _, _, yaw = self.quaternion_to_euler(orientation.x, orientation.y, orientation.z, orientation.w)
        return yaw

    def euler_to_quaternion(self,roll, pitch, yaw):
        qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(roll / 2) * np.sin(pitch / 2) * np.sin(
            yaw / 2)
        qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.cos(pitch / 2) * np.sin(
            yaw / 2)
        qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(roll / 2) * np.sin(pitch / 2) * np.cos(
            yaw / 2)
        qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.sin(pitch / 2) * np.sin(
            yaw / 2)

        return qx, qy, qz, qw

    def quaternion_to_euler(self,x, y, z, w):
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(t0, t1)
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch = math.asin(t2)
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(t3, t4)
        return yaw, pitch, roll

    def generate_rollout_trajectory(self, MODE='random'):
        '''generate random trajectories sampled from real world-like or random distribution so that points can be sampled from them to train the model'''
        if (MODE == 'random'):
            '''generate random trajectories'''
            trajectory = []
            vel_samples = np.random.uniform(self.min_vel, self.max_vel, self.SAMPLING_SIZE)
            steer_angle_samples = np.random.uniform(self.min_steer_limit, self.max_steer_limit, self.SAMPLING_SIZE)
            for v in vel_samples:
                for delta in steer_angle_samples:
                    current_state = [0.0, 0.0, 0.0]
                    trajectory.append(current_state)
                    for i in range(self.HORIZON_LENGTH):
                        next_state = self.update_state_with_kinematics(current_state, v, delta)
                        trajectory.append(next_state)
                        self.train_data.append(
                            [current_state[0], current_state[1], current_state[2], v, delta, next_state[0],
                             next_state[1], next_state[2]])
                        # print([current_state,[v,delta],next_state])
                        current_state = copy.deepcopy(next_state)
                    self.trajectories.append(trajectory)

        elif (MODE == 'real'):
            '''generate real world like trajectories'''

    def write_data_to_file(self, filename):
        '''write generated training data to file'''
        header = ['current_lidar_data','x', 'y', 'theta', 'vel', 'delta','next_lidar_data', 'x_next', 'y_next', 'theta_next','reward']
        if not self.APPEND_MODE:
            with open(self.OUTPUT_FILE_PATH + '/' + filename, "w") as f:
                writer = csv.writer(f, delimiter=',')
                writer.writerow(header)
                for row in self.train_data:
                    writer.writerow(row)
        if self.APPEND_MODE:
            with open(self.OUTPUT_FILE_PATH + '/' + filename, "a") as f:
                writer = csv.writer(f, delimiter=',')
                for row in self.train_data:
                    writer.writerow(row)
        rospy.loginfo("Training data generation completed.")

    def generate_training_data(self):
        rospy.loginfo("Generating Training data....")
        self.generate_rollout_trajectory(MODE='random')
        self.write_data_to_file('train_data.csv')
        rospy.loginfo("Training data generation completed.")

    def is_goal_reached(self):
        '''return true if goal is reached'''
        return self.goal_reached

    def update_state_with_kinematics(self, current_state, vel, delta):
        state = copy.deepcopy(current_state)
        state[0] += vel * cos(state[2]) * self.dT
        state[1] += vel * sin(state[2]) * self.dT
        state[2] += (vel / self.L) * tan(delta) * self.dT
        return state

    def reset_environment(self):
        self.COLLISION_FLAG = False
        pose_info = PoseWithCovarianceStamped()
        pose_info.pose.pose.position = Point(self.reset_pose[0],self.reset_pose[1],0.0)
        pose_info.pose.pose.orientation = self.heading(self.reset_pose[2])
        self.pose_pub.publish(pose_info)
        if self.NAVIGATION_MODE:
            self.NAVIGATION_MODE=0
        return self.reset_image_state
    
    def get_reset_state(self):
        return self.reset_image_state

    def step(self,vel,delta):
        self.current_state = copy.deepcopy(self.current_pose)
        self.send_motor_command(vel,delta)
        start_time = time.time()
        # wait for 100ms to see effect of action
        while not rospy.is_shutdown():
            if time.time()- start_time>=self.STEP_TIME:
                break
        next_state = copy.deepcopy(self.current_pose)
        next_state_image = self.get_local_image(self.original_ranges)
        reward, lap_complete_status = self.calculate_reward(next_state[0:2], self.current_state[0:2])
        return next_state_image,reward,lap_complete_status


    def toggle_navigation(self):
        '''send navigation toggle key data 'n' using keyboard topic'''
        msg = String()
        self.NAVIGATION_ENABLE = not self.NAVIGATION_ENABLE
        print("Navigation enable=", self.NAVIGATION_ENABLE)
        msg.data = 'n'
        self.keyboard_pub.publish(msg)

    def goal_callback(self, msg):
        '''callback function for goal given from rviz'''
        self.goal_pose = msg.pose.position
        self.goal_received = True
        self.goal_reached = False
        # self.generate_training_data()
        self.toggle_navigation()
        if self.DEBUG_MODE:
            print("Goal pos=", self.goal_pose)

    def lidarCB(self, msg):
        '''Initializes reused buffers, and stores the relevant laser scanner data for later use.'''
        if not self.LIDAR_INIT:
            self.LIDAR_DELTA_ANGLE_INCR = msg.angle_increment
            self.LIDAR_ANGLE_MIN = msg.angle_min
            self.reset_image_state = self.get_local_image(np.array(msg.ranges))
            self.LIDAR_INIT = True

        self.downsampled_ranges = np.array(msg.ranges[::self.ANGLE_STEP])
        self.original_ranges = np.array(msg.ranges)


    def construct_local_image(self,mode,idx,lidar_ranges):
        '''construct 2d image from lidar observation in local coordinate system'''
        img = self.get_local_image(lidar_ranges)
        if mode =='train':
            filename="./images/train/img"+str(idx)+".png"
        else:
            filename = "./images/test/img" + str(idx) + ".png"
        # imsave(filename,img)

    def get_local_image(self,lidar_ranges):
        img = 100.0 * np.ones((150, 150),dtype='float32')
        for i in range(len(lidar_ranges)):
            scan_range = lidar_ranges[i]  # range measured for the particular scan
            scan_angle = self.LIDAR_ANGLE_MIN + i * self.LIDAR_DELTA_ANGLE_INCR  # bearing measured
            if scan_range > 15.0:
                continue
            # find position of cells in the local frame
            occupied_x = scan_range * cos(scan_angle)
            occupied_y = scan_range * sin(scan_angle)
            p_x = int(occupied_x / self.GRID_SIZE) + self.Origin_X
            p_y = int(occupied_y / self.GRID_SIZE) + self.Origin_Y
            if (0 <= p_x < 150 and 0 <= p_y < 150):
                # img[p_x,p_y]=255
                img[149 - p_y, p_x] = 255.0
        img[149 - self.Origin_Y - 1:149 - self.Origin_Y + 2, self.Origin_X - 1:self.Origin_X + 2] = 255.0
        return img



    def construct_lidar_observation_images(self):
        '''construct 2d images from lidar observations in global coordinate system'''
        for i,train_img in enumerate(self.train_images):
            self.construct_local_image('train',i,train_img)
        for j,test_img in enumerate(self.test_images):
            self.construct_local_image('test', j, test_img)
        rospy.loginfo("Images writing completed.")

    def odom_callback(self, msg):
        '''odometry callback from simulator'''
        self.current_pose[0] = msg.pose.pose.position.x
        self.current_pose[1] = msg.pose.pose.position.y
        self.current_pose[2] = self.quaternion_to_euler_yaw(msg.pose.pose.orientation)
        if not self.ODOM_INIT:
            self.reset_pose = copy.deepcopy(self.current_pose)
            self.ODOM_INIT = True

        self.current_vel = msg.twist.twist.linear.x
        if self.goal_received:
            dist2goal = np.linalg.norm(
                np.array([self.goal_pose.x, self.goal_pose.y]) - np.array([self.current_pose[0:2]]))
            if dist2goal < self.GOAL_THRESHOLD:
                self.goal_reached = True
                self.goal_received = False
                self.toggle_navigation()
                self.write_data_to_file('train_data.csv')
                self.APPEND_MODE = True
                rospy.loginfo("Goal Reached !")

    def action_callback(self, msg):
        '''get action from car controller'''
        # vel = copy.deepcopy(self.current_vel)
        vel = msg.drive.speed
        vel = max(0.5,0.25*np.around(vel/0.25))
        steer = msg.drive.steering_angle
        if self.NAVIGATION_ENABLE:
            if self.ACTION_INIT:
                next_obs = copy.deepcopy(self.downsampled_ranges)
                next_full_obs = copy.deepcopy(self.original_ranges)
                next_state = copy.deepcopy(self.current_pose)
                reward,prog_complete_status = self.calculate_reward(next_state[0:2], self.current_state[0:2])
                if prog_complete_status:
                    return
                # append data(current state,current_action,current_obs,next_obs,next_state,reward) to train_data array
                data_chunk = [self.current_obs,self.current_state[0], self.current_state[1], self.current_state[2],
                              self.current_action[0], self.current_action[1],next_obs, next_state[0], next_state[1],
                              next_state[2], reward]
                self.train_images.append(self.current_full_obs)
                self.test_images.append(next_full_obs)
                self.train_data.append(data_chunk)
                self.current_state = next_state
                self.current_action = np.array([vel, steer])
                self.current_obs = next_obs
                self.current_full_obs = next_full_obs
            else:
                self.current_state = copy.deepcopy(self.current_pose)
                self.current_obs = copy.deepcopy(self.downsampled_ranges)
                self.current_full_obs = copy.deepcopy(self.original_ranges)
                self.current_action = np.array([vel, steer])
                self.ACTION_INIT = True

    def get_current_pose(self):
        '''return current pose of the car'''
        return self.current_pose

    def get_arc_lengths(self, waypoints):
        d = np.diff(waypoints, axis=0)
        consecutive_diff = np.sqrt(np.sum(np.power(d, 2), axis=1))
        dists_cum = np.cumsum(consecutive_diff)
        dists_cum = np.insert(dists_cum, 0, 0.0)
        return dists_cum

    def find_nearest_index(self, car_pos):
        distances_array = np.linalg.norm(self.center_lane - car_pos, axis=1)
        min_dist_idx = np.argmin(distances_array)
        return min_dist_idx, distances_array[min_dist_idx]

    def find_current_arc_length(self, car_pos):
        nearest_index, minimum_dist = self.find_nearest_index(car_pos)
        current_s = self.element_arc_lengths[nearest_index]
        return current_s, minimum_dist

    def calculate_reward(self, next_car_pos,current_car_pos):  # car pos is np.array[x,y]
        next_arc_length_travelled, next_lateral_deviation = self.find_current_arc_length(next_car_pos)
        current_arc_length_travelled, current_lateral_deviation = self.find_current_arc_length(current_car_pos)
        progress = next_arc_length_travelled - current_arc_length_travelled
        if(0<next_arc_length_travelled<self.element_arc_lengths[int(len(self.element_arc_lengths)/3)] and self.COUNT_FLAG):
            self.lap_count +=1
            rospy.loginfo("Lap count=%s",self.lap_count)
            self.lap_pub.publish('incr')
            self.COUNT_FLAG = False
            if(self.lap_count>=2):
                self.toggle_navigation()
                self.PROG_COMPLETE_FLAG=True
                return 1, self.PROG_COMPLETE_FLAG

        if(next_arc_length_travelled>self.element_arc_lengths[int(len(self.element_arc_lengths)/2)]):
            self.COUNT_FLAG = True

        if progress<0:
            progress += self.element_arc_lengths[-1]
        else:
            progress = next_arc_length_travelled - current_arc_length_travelled
        progress_reward = progress * self.PROGRESS_REWARD_FACTOR
        lateral_deviation_reward = -next_lateral_deviation * self.DEVIATION_REWARD_FACTOR
        total_reward = progress_reward + lateral_deviation_reward
        return total_reward,self.PROG_COMPLETE_FLAG

    def read_waypoints_array_from_csv(self, filename):
        '''read waypoints from given csv file and return the data in the form of numpy array'''
        if filename == '':
            raise ValueError('No any file path for waypoints file')
        with open(filename) as f:
            path_points = [tuple(line) for line in csv.reader(f, delimiter=',')]
        path_points = np.array([[float(point[0]), float(point[1])] for point in path_points])
        return path_points

    def preprocess_track_data(self):
        self.center_lane = self.read_waypoints_array_from_csv(self.CENTER_TRACK_FILENAME)
        self.element_arc_lengths = self.get_arc_lengths(self.center_lane)

    def send_motor_command(self, velocity, steer_angle):
        '''publish motor commands to drive and steering servo'''
        ackermann_cmd = AckermannDriveStamped()
        ackermann_cmd.header = self.create_header(self.car_frame)
        ackermann_cmd.drive.steering_angle = steer_angle
        self.current_steering = steer_angle
        ackermann_cmd.drive.speed = velocity
        # ackermann_cmd.drive.acceleration = 1.0
        self.cmd_vel_pub.publish(ackermann_cmd)


# if __name__ == '__main__':
#     sim_node = SimInterface()
#     while not rospy.is_shutdown():
#         if sim_node.PROG_COMPLETE_FLAG:
#             sim_node.write_data_to_file('train_data.csv')
#             sim_node.construct_lidar_observation_images()
#             sim_node.PROG_COMPLETE_FLAG =False
#     rospy.spin()
