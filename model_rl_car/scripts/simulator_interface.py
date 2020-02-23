#!/usr/bin/env python

import matplotlib.pyplot as plt
import os
import time
import numpy as np
import copy
import csv
from math import sin,cos,tan

import rospy
from geometry_msgs.msg import Pose, PoseStamped, Quaternion
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Path, Odometry
from std_msgs.msg import Duration, Header,String
from tf.transformations import quaternion_from_euler, euler_from_quaternion
from visualization_msgs.msg import Marker, MarkerArray


class SimInterface:
    def __init__(self):
        rospy.init_node('sim_interface_node')
        cmd_vel_topic = rospy.get_param('cmd_vel_topic_name', '/nav')
        odom_topic = rospy.get_param('odom_topic_name', '/odom')
        goal_topic = rospy.get_param('goal_topic_name', '/move_base_simple/goal')
        key_topic = rospy.get_param('key_topic_name','/key')
        self.car_frame = rospy.get_param('car_frame', 'base_link')
        self.GOAL_THRESHOLD = rospy.get_param('goal_threshold', 0.10)

        self.DEBUG_MODE = rospy.get_param('debug_mode', True)

        dirname = os.path.dirname(__file__)
        path_folder_name = rospy.get_param('path_folder_name', 'track')
        self.OUTPUT_FILE_PATH = os.path.join(dirname, path_folder_name)

        self.min_vel = rospy.get_param('min_speed',-2.5)
        self.max_vel = rospy.get_param('max_speed', 2.5)
        self.max_steer_limit= rospy.get_param('max_steering', 0.418)
        self.min_steer_limit = rospy.get_param('min_steering', -0.418)
        self.L = rospy.get_param('car_length', 0.325)
        self.SAMPLING_SIZE =rospy.get_param('sample_size',100)
        self.HORIZON_LENGTH = rospy.get_param('horizon_length',20)
        self.dT = rospy.get_param('dT',0.05)

        self.current_pose = np.array([0.0,0.0,0.0])
        self.current_vel = None
        self.current_steering= None
        self.train_data = []
        self.trajectories=[]

        # Goal status related variables
        self.goal_pose = None
        self.goal_reached = False
        self.goal_received = False

        # Publishers
        self.cmd_vel_pub = rospy.Publisher(cmd_vel_topic, AckermannDriveStamped, queue_size=10)
        self.keyboard_pub = rospy.Publisher(key_topic, String, queue_size=10)

        # Subscribers
        rospy.Subscriber(odom_topic, Odometry, self.odom_callback, queue_size=1)
        rospy.Subscriber(goal_topic, PoseStamped, self.goal_callback, queue_size=1)
        # Timer callback function for the control loop
        # rospy.Timer(rospy.Duration(1.0 / self.CONTROLLER_FREQ), self.controlLoopCB)

    def create_header(self, frame_id):
        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = frame_id
        return header

    def heading(self, yaw):
        q = quaternion_from_euler(0, 0, yaw)
        return Quaternion(*q)

    def quaternion_to_euler_yaw(self, orientation):
        _, _, yaw = euler_from_quaternion((orientation.x, orientation.y, orientation.z, orientation.w))
        return yaw

    def generate_rollout_trajectory(self,MODE='random'):
        '''generate random trajectories sampled from real world-like or random distribution so that points can be sampled from them to train the model'''
        if(MODE == 'random'):
            '''generate random trajectories'''
            trajectory=[]
            vel_samples = np.random.uniform(self.min_vel,self.max_vel,self.SAMPLING_SIZE)
            steer_angle_samples = np.random.uniform(self.min_steer_limit,self.max_steer_limit,self.SAMPLING_SIZE)
            for v in vel_samples:
                for delta in steer_angle_samples:
                    current_state = [0.0,0.0,0.0]
                    trajectory.append(current_state)
                    for i in range(self.HORIZON_LENGTH):
                        next_state = self.update_state_with_kinematics(current_state,v,delta)
                        trajectory.append(next_state)
                        self.train_data.append([current_state[0],current_state[1],current_state[2],v,delta,next_state[0],next_state[1],next_state[2]])
                        # print([current_state,[v,delta],next_state])
                        current_state = copy.deepcopy(next_state)
                    self.trajectories.append(trajectory)

        elif(MODE == 'real'):
            '''generate real world like trajectories'''


    def write_data_to_file(self,filename,data):
        '''write generated training data to file'''
        header = ['x', 'y', 'theta','vel','delta','x_next','y_next','theta_next']
        with open(self.OUTPUT_FILE_PATH +'/'+ filename, "w") as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(header)
            for row in self.train_data:
                writer.writerow(row)

    def generate_training_data(self):
        rospy.loginfo("Generating Training data....")
        self.generate_rollout_trajectory(MODE='random')
        self.write_data_to_file('train_data.csv', self.train_data)
        rospy.loginfo("Training data generation completed.")

    def is_goal_reached(self):
        '''return true if goal is reached'''
        return self.goal_reached

    def update_state_with_kinematics(self,current_state,vel,delta):
        state = copy.deepcopy(current_state)
        state[0] += vel*cos(state[2])*self.dT
        state[1] += vel*sin(state[2])*self.dT
        state[2] += (vel/self.L)*tan(delta)*self.dT
        return state

    def enable_navigation(self):
        '''send navigation enable key data 'n' using keyboard topic'''
        msg = String()
        msg.data = 'n'
        self.keyboard_pub.publish(msg)

    def goal_callback(self,msg):
        '''callback function for goal given from rviz'''
        self.goal_pose = msg.pose.position
        self.goal_received = True
        self.goal_reached = False
        self.generate_training_data()
        if self.DEBUG_MODE:
            print("Goal pos=", self.goal_pose)

    def odom_callback(self,msg):
        '''odometry callback from simulator'''
        self.current_pose[0] = msg.pose.pose.position.x
        self.current_pose[1] = msg.pose.pose.position.y
        self.current_pose[2] = self.quaternion_to_euler_yaw(msg.pose.pose.orientation)
        self.current_vel = msg.twist.twist.linear.x
        if self.goal_received:
            dist2goal = np.linalg.norm(np.array([self.goal_pose.x,self.goal_pose.y])-np.array([self.current_pose[0:2]]))
            if dist2goal < self.GOAL_THRESHOLD:
                self.goal_reached = True
                self.goal_received = False
                rospy.loginfo("Goal Reached !")

    def get_current_pose(self):
        '''return current pose of the car'''
        return self.current_pose

    def send_motor_command(self,velocity,steer_angle):
        '''publish motor commands to drive and steering servo'''
        ackermann_cmd = AckermannDriveStamped()
        ackermann_cmd.header = self.create_header(self.car_frame)
        ackermann_cmd.drive.steering_angle = steer_angle
        self.current_steering= steer_angle
        ackermann_cmd.drive.speed = velocity
        # ackermann_cmd.drive.acceleration = 1.0
        self.cmd_vel_pub.publish(ackermann_cmd)

if __name__ == '__main__':
    sim_node = SimInterface()
    rospy.spin()




