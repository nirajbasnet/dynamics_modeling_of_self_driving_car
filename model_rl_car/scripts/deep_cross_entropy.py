#!/usr/bin/python3.6
import time
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from skorch import NeuralNetClassifier
from skorch.callbacks import EpochScoring
import torch
import matplotlib.pyplot as plt
from torch import nn
import pickle
import rospy
import cem_simulator_interface
from cem_simulator_interface import SimInterface

class CNN_ClassifierModule(nn.Module):
    def __init__(self):
        super(CNN_ClassifierModule, self).__init__()
        self.layer1 =nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),  # 150X150
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=3)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=24, kernel_size=3, stride=1, padding=1),#50X50
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.drop_out = nn.Dropout()
        self.out = nn.Sequential(
            nn.Linear( 25* 25 * 24, 1300),
            nn.ReLU(),
            nn.Linear(1300, 85),
            nn.Softmax(dim=-1)
        )

    def forward(self, X, **kwargs):
        X = self.layer1(X)
        X=self.layer2(X)
        X=X.reshape(X.size(0), -1)
        X= self.drop_out(X)
        X = self.out(X)
        return X

class MLP_module(nn.Module):
    def __init__(self, num_units=340):
        super(MLP_module, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dense_block1 = nn.Sequential(
            nn.Linear(75*75,35*35),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(35*35,400),
            nn.ReLU()
        )
        self.out_block = nn.Sequential(
            nn.Linear(400, 85),
            nn.Softmax(dim=-1)
        )

    def forward(self, X, **kwargs):
        X= self.pool(X)
        X= X.reshape(-1,75*75)
        X = self.dense_block1(X)
        X = self.out_block(X)
        return X

class DeepCEM:
    def __init__(self):
        rospy.init_node("deep_cem")
        self.n_steer = 17
        self.n_vel = 5
        self.total_actions = self.n_steer * self.n_vel
        self.velocity_range = [0.5, 0.75, 1.0, 1.25, 1.5]  #unitL: m/s
        self.min_steering_angle = -0.4188   #unit: radians. Equivalent to -24 degrees
        self.steer_range = [self.min_steering_angle + 0.0523 * i for i in range(self.n_steer)]
        self.model_file_name = '/CEMmodel.pkl'
        self.lrate = 0.1
        self.device = 'cpu'

        #Deep MLP module
        self.net = NeuralNetClassifier(
            MLP_module,
            max_epochs=20,
            lr=self.lrate,
            iterator_train__shuffle=False,
        )
        #Deep CNN module
        self.cnet = NeuralNetClassifier(
            CNN_ClassifierModule,
            max_epochs=30,
            lr=self.lrate,
            device=self.device,
            optimizer=torch.optim.Adam,
        )

        # Simulator interface variables
        self.sim_node = SimInterface()
        rospy.sleep(2.0)   #sleeping for 2 seconds for simulator to start properly
        self.reset_state_sim = self.sim_node.get_reset_state()
        self.cnet.initialize_module()
        self.net.initialize_module()

        #Training related variables
        self.NN_MODULE_TYPE = 'deepCNN'   #'deepMLP' and 'deepCNN
        self.N_SESSIONS = 100
        self.N_TSTEPS_HORIZON = 25
        self.ELITE_PERCENTILE = 70
        self.N_GENERATIONS = 100
        self.log=[]
        self.FIRST_RUN = True

        self.train_CEM()



    def test_initialize_neural_modules(self):
        plt.imshow(self.reset_state_sim)
        plt.show()
        torch.manual_seed(0)
        data = np.tile(self.reset_state_sim,(self.total_actions,1))
        X = torch.tensor(data.reshape((self.total_actions,1,150,150)))
        # Y = torch.tensor(np.arange(self.total_actions).astype(int))
        Y = torch.tensor(30*np.ones(self.total_actions).astype(int))
        self.net.partial_fit(X, Y, range(self.total_actions))
        self.cnet.partial_fit(X, Y, range(self.total_actions))


    def test_MLP(self):
        torch.manual_seed(0)
        X = torch.ones([10, 1, 150, 150], dtype=torch.float32)
        test = torch.ones([1, 1, 150, 150], dtype=torch.float32)
        Y = torch.tensor(np.ones(10).astype(int))
        self.net.partial_fit(X, Y, range(85))
        y_proba = self.net.predict_proba(X[1].reshape(1, 1, 150, 150))
        print(y_proba, y_proba.shape)

    def test_CNN(self):
        torch.manual_seed(0)
        X=torch.ones([10,1,150, 150], dtype=torch.float32)
        test = torch.ones([1,1,150, 150], dtype=torch.float32)
        Y=torch.tensor(np.ones(10).astype(int))
        print(X.shape, Y.shape)
        # self.cnet.partial_fit(X,Y,range(85))
        y_proba = self.cnet.predict_proba(X[1].reshape(1,1,150,150))
        print(y_proba,y_proba.shape)

    def encode(self,vel, delta):
        data = np.zeros(self.n_steer * self.n_vel)
        vel_idx = int(vel / 0.25) - 2
        delta_idx = int(delta /0.05) + 8
        total_idx = vel_idx * self.n_steer + delta_idx
        data[total_idx] = 1
        return data

    def decode(self,enc_action):
        t_idx = np.argmax(enc_action)
        v_idx = int(t_idx / self.n_steer)
        s_idx = t_idx - v_idx * self.n_steer
        return self.velocity_range[v_idx], self.steer_range[s_idx]

    def decode_action_idx(self,t_idx):
        v_idx = int(t_idx / self.n_steer)
        s_idx = t_idx - v_idx * self.n_steer
        return self.velocity_range[v_idx], self.steer_range[s_idx]

    def dump_model_to_file(self,filename,network):
        with open(filename, 'wb') as f:
            pickle.dump(network, f)

    def load_model_from_file(self,filename):
        with open(filename, 'rb') as f:
            new_net = pickle.load(f)
            return new_net

    def generate_session(self, t_max=100):
        """
        Play a single episode using agent neural network.
        Terminate when game finishes or after :t_max: steps
        """
        states, actions = [], []
        total_reward = 0
        probs= None
        sampled_action=None
        time_penalty = 0.1
        current_state = self.sim_node.reset_environment()   #reset environment before generating sessions
        self.sim_node.toggle_navigation()
        rospy.sleep(0.1)

        if self.FIRST_RUN:
            probs = (1. / self.total_actions) * np.ones(self.total_actions)
            sampled_action = np.random.choice(np.arange(self.total_actions), p=probs.reshape(self.total_actions, ))
        # use the probabilities to pick an action
        # print(probs)

        # print("sampled action=",sampled_action)
        for t in range(t_max):
            try:
                # use agent to predict a vector of action probabilities for state s
                curent_state_tensor = torch.tensor(current_state.reshape(1, 1, 150, 150))
                if not self.FIRST_RUN:
                    if self.NN_MODULE_TYPE =='deepCNN':
                        probs = self.cnet.predict_proba(curent_state_tensor)
                    elif self.NN_MODULE_TYPE =='deepMLP':
                        probs = self.net.predict_proba(curent_state_tensor)
                    sampled_action = np.random.choice(np.arange(self.total_actions),
                                                      p=probs.reshape(self.total_actions, ))
                # use the probabilities to pick an action

                # sampled_action = np.random.choice(np.arange(self.total_actions), p=probs.reshape(self.total_actions,))
                # print("sampled action=",sampled_action)

                #step in the environmentto get new state and reward
                decoded_vel,decoded_delta = self.decode_action_idx(sampled_action)
                # print(decoded_vel,decoded_delta)

                new_state, reward, complete_status = self.sim_node.step(decoded_vel,decoded_delta)
                print(decoded_vel,decoded_delta,reward)

                # print(reward,new_state)
                # record sessions to train later on
                states.append(current_state)
                actions.append(sampled_action)
                total_reward += reward -time_penalty
                current_state = new_state
                if complete_status:
                    break
            except KeyboardInterrupt:
                raise SystemError

        if not self.sim_node.COLLISION_FLAG:
            self.sim_node.toggle_navigation()
        return states, actions, total_reward


    def select_elites(self,states_batch, actions_batch, rewards_batch, percentile=50):
        """
        Select states and actions from games that have rewards >= percentile
        :param states_batch: list of lists of states, states_batch[session_i][t]
        :param actions_batch: list of lists of actions, actions_batch[session_i][t]
        :param rewards_batch: list of rewards, rewards_batch[session_i]
        :returns: elite_states,elite_actions, both 1D lists of states and respective actions from elite sessions"""

        reward_threshold = np.percentile(rewards_batch, percentile)
        elite_states = []
        elite_actions = []
        for i in range(len(states_batch)):
            if rewards_batch[i] >= reward_threshold:
                for j in range(len(states_batch[i])):
                    elite_states.append(states_batch[i][j])
                    elite_actions.append(actions_batch[i][j])
        return elite_states, elite_actions

    def train_CEM(self):
        for i in range(self.N_GENERATIONS):
            # generate new sessions
            print("Generation ",i)
            sessions = [ self.generate_session(self.N_TSTEPS_HORIZON) for i in range(self.N_SESSIONS) ]
            if self.FIRST_RUN:
                self.FIRST_RUN = False

            states_batch, actions_batch, rewards_batch = map(np.array, zip(*sessions))

            elite_states, elite_actions = self.select_elites(states_batch, actions_batch, rewards_batch, self.ELITE_PERCENTILE)

            self.partial_fit_data(elite_states, elite_actions)

            self.show_progress(rewards_batch, self.log, self.ELITE_PERCENTILE, reward_range=[0, np.max(rewards_batch)])

    def partial_fit_data(self,elite_states,elite_actions):
        X = torch.tensor(np.reshape(elite_states, (-1, 1, 150, 150)), dtype=torch.float32)
        Y = torch.tensor(np.array(elite_actions).astype(int))
        if self.NN_MODULE_TYPE =='deepCNN':
            '''Use DeepCNN module to fit states with respective actions'''
            self.cnet.partial_fit(X, Y, range(self.total_actions))

        elif self.NN_MODULE_TYPE =='deepMLP':
            '''Use DeepMLP module to fit states with respective actions'''
            self.net.partial_fit(X, Y, range(self.total_actions))

    def show_progress(self,rewards_batch, log, percentile, reward_range):
        """
        Displays training progress.
        """
        mean_reward = np.mean(rewards_batch)
        threshold = np.percentile(rewards_batch, percentile)
        self.log.append([mean_reward, threshold])
        # clear_output(True)
        print("mean reward = ",mean_reward, " threshold=",threshold)
        plt.figure(figsize=[8, 4])
        plt.subplot(1, 2, 1)
        plt.plot(list(zip(*self.log))[0], label='Mean rewards')
        plt.plot(list(zip(*self.log))[1], label='Reward thresholds')
        plt.legend()
        plt.grid()

        # plt.subplot(1, 2, 2)
        # plt.hist(rewards_batch, range=reward_range)
        # plt.vlines([np.percentile(rewards_batch, percentile)],
        #            [0], [100], label="percentile", color='red')
        # plt.legend()
        # plt.grid()
        plt.show()

if __name__ == "__main__":
    deep_cem = DeepCEM()
    while not rospy.is_shutdown():
        if deep_cem.sim_node.PROG_COMPLETE_FLAG:
            # deep_cem.sim_node.write_data_to_file('train_data.csv')
            # deep_cem.sim_node.construct_lidar_observation_images()
            deep_cem.sim_node.PROG_COMPLETE_FLAG =False
    rospy.spin()


