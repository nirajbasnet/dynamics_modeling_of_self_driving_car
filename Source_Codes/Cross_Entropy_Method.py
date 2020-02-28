import numpy as np

NUM_OF_SAMPLES_EACH_ITER = 10
NUM_OF_ELITE_SAMPLES = 4


def generate_trajectories(num_samples, vel_mean, vel_var, steering_mean, steering_var):
    # Samples "num_samples" trajectories using the given parameters and returns
    # example: for the first trajectory, the velocity will be vel = vel_mean + sqrroot(vel_var) *

    sampled_velocities = np.random.normal(loc=vel_mean, scale=np.sqrt(vel_var), size=num_samples)
    sampled_steering_angles = np.random.normal(loc=steering_mean, scale=np.sqrt(steering_var), size=num_samples)

    # Now generate 10 trajectories using the 10 combinaitons of velocities and steering angles above
    sampled_trajectories = []  # suppose you got a list of sampled traectories

    return sampled_velocities, sampled_steering_angles, sampled_trajectories


def find_best_traj_indices(list_of_trajectories):
    # do in-place sorting find the new mean = the average of the sum of the best five
    # find the variance of these best five values
    return None


def get_new_model_params(lst_sampled_vels, lst_sampled_str_angles, best_idxs):

    best_vel_sum = 0
    best_str_angle_sum = 0

    temp_vel_list_for_calc_variance = []
    temp_angle_list_for_calc_variance = []

    for item in range(len(best_idxs)):
        best_vel_sum += lst_sampled_vels[best_idxs[item]]
        best_str_angle_sum += lst_sampled_str_angles[best_idxs[item]]

        temp_vel_list_for_calc_variance.append(lst_sampled_vels[best_idxs[item]])
        temp_angle_list_for_calc_variance.append(lst_sampled_str_angles[best_idxs[item]])

    new_vel_mean = best_vel_sum/5.0
    new_angle_mean = best_str_angle_sum/5.0


    new_vel_var = np.var(np.array(temp_vel_list_for_calc_variance))
    new_angle_var = np.var(np.array(temp_angle_list_for_calc_variance))

    return new_vel_mean, new_vel_var, new_angle_mean, new_angle_var




def cross_entropy_method(smoothing_factor, velocity_mean, velocity_var, steering_angle_mean, steering_angle_var):
    # initialize with a mean and variance drawn from a gaussian for both velocity and steering angle

    #Change the stopping criteria later
    while True:
        # generate sample trajectories
        list_of_sampled_vels, list_of_sampled_str_angles, list_of_sample_trajs = generate_trajectories(
            NUM_OF_SAMPLES_EACH_ITER, velocity_mean, velocity_var, steering_angle_mean, steering_angle_var)

        # find the best trajectories and the indices
        best_indices = find_best_traj_indices(list_of_sample_trajs)

        # get the new mean and the new variance from evaluation
        new_velocity_mean, new_velocity_var, new_steering_angle_mean, new_steering_angle_var = get_new_model_params(list_of_sampled_vels, list_of_sampled_str_angles, best_indices)

        #smoothly update the parameters
        velocity_mean = smoothing_factor * velocity_mean + (1-smoothing_factor) * new_velocity_mean
        velocity_var = smoothing_factor * velocity_var + (1-smoothing_factor) * new_velocity_var
        steering_angle_mean = smoothing_factor * steering_angle_mean + (1-smoothing_factor) * new_steering_angle_mean
        steering_angle_var = smoothing_factor * steering_angle_var + (1-smoothing_factor) * new_velocity_var


