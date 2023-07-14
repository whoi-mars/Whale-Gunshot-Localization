import os
import argparse
import itertools
import math
from multiprocessing import Pool

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
import pandas as pd

from whale_gunshot_localization import config, PROJECT_ROOT_DIR
from whale_gunshot_localization.utils.experimental import Localizer, MultilaterationOpt, MultilaterationGrid

# parse input arguments
parser = argparse.ArgumentParser(description="Run Monte Carlo simulations for data association/localization for a train of calls")
parser.add_argument('--simulate', action='store_true',
                    help="simulate rather than use previous results if available (default: false)")
parser.add_argument('--save_figs', action='store_true',
                    help="save figures (default: false)")
parser.add_argument('--background', '-b', action='store_true',
                    help='silence the progress bar')
args = parser.parse_args()

def to_positive_theta(theta):
    if theta >= 0:
        return theta
    else:
        return 360 - abs(theta)

def get_bearing(x_ends, y_ends):

    theta = math.atan2(y_ends[1] - y_ends[0], x_ends[1] - x_ends[0]) * (180 / math.pi)
    if theta >= 0:
        return theta
    else:
        return 360 - abs(theta)


def plot_localization(source_locs, locs_est=None, title='', buffer=6000): 
    
    # load constants
    TOSSIT_locations = np.asarray([config['TOSSIT']['TOSSIT_y'], config['TOSSIT']['TOSSIT_x']]).T
    min_x = config['scaling']['min_x']
    max_x = config['scaling']['max_x']
    min_y = config['scaling']['min_y']
    max_y = config['scaling']['max_y']
    
    fig, ax = plt.subplots(1, 1)
    ax.plot(TOSSIT_locations[:,1], TOSSIT_locations[:,0], '^', markersize=12, label='sensor', markeredgewidth=2)
    ax.plot(source_locs[:,1], source_locs[:,0], 'o', label='ground truth')
    if locs_est is not None:
        ax.plot(locs_est[:,1], locs_est[:,0], 'x', label='estimate', markeredgewidth=2)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_xlim([min_x-buffer, max_x+buffer])
    ax.set_ylim([min_y-buffer, max_y+buffer])
    ax.invert_yaxis()
    ax.legend()
    if len(title):
        ax.set_title(title)
    fig.savefig('test.png')

def generate_trajectory(num_points, beam_width=20, measurement_stats=(0, 500000), timing_stats=(1, 0.5), repetition_time=0.5, num_repetitions=1, whale_speed=1.3, chunk_size=2, max_channel_offset=40, rng=None):
    
    # check inputs
    assert num_points > 0, "number of sources must be non-negative"

    if rng is None:
        rng = np.random
    
    # convert whale speed to m / second
    whale_speed *= (1000 / 60)
    # convert timing_stats to seconds
    timing_stats = tuple([i*60 for i in timing_stats])
    # convert chunk size to seconds
    chunk_size *= 60
    
    # get TOSSIT locations and extreme coordinate values
    TOSSIT_locations = np.asarray([config['TOSSIT']['TOSSIT_y'], config['TOSSIT']['TOSSIT_x']]).T
    min_x = config['scaling']['min_x']
    max_x = config['scaling']['max_x']
    min_y = config['scaling']['min_y']
    max_y = config['scaling']['max_y']
    
    # choose number of sources and generate source locations
    source_locs = np.concatenate((rng.uniform(min_y, max_y, size=(1,1)), rng.uniform(min_x, max_x, size=(1,1))), axis=1)
    time_stamps = np.arange(0, num_repetitions * repetition_time, repetition_time)
    heading = rng.uniform(0, 359, size=1)
    
    curr_loc = source_locs[[0]]
    measurements = [np.asarray([]) for _ in range(TOSSIT_locations.shape[0])]
    while True:
        
        # get measurements
        for t in range(TOSSIT_locations.shape[0]):
            # calculate range measurements from all sources to TOSSIT t, adding Gaussian noise
            r = np.linalg.norm(curr_loc - TOSSIT_locations[t,:], axis=1) + \
                rng.normal(loc=measurement_stats[0], scale=np.sqrt(measurement_stats[1]), size=num_repetitions)
            measurements[t] = np.append(measurements[t], r)

        if source_locs.shape[0] == num_points:
            break
        
        # time delta to next source
        new_time_delta = rng.normal(loc=timing_stats[0], scale=np.sqrt(timing_stats[1]))
        
        # timestamp(s) for current source location
        time_stamps = np.append(time_stamps, [time_stamps[-1] + new_time_delta + repetition_time*i for i in range(num_repetitions)])

        # calculate magnitude of transition vector
        vec_mag = whale_speed * new_time_delta
    
        # transition vector to get from current location to the next one
        transition_vector = (vec_mag * np.asarray([-np.sin(np.radians(heading)), np.cos(np.radians(heading))])).T
        
        # get new location and save
        curr_loc += transition_vector
        source_locs = np.concatenate((source_locs, curr_loc), axis=0)
        
        # get new heading
        heading += rng.uniform(-beam_width / 2, beam_width / 2)
    
    # create offset timestamps for each sensor
    time_stamps = time_stamps[np.newaxis,:]
    for t in range(TOSSIT_locations.shape[0]):
        offset = rng.uniform(low=0, high=max_channel_offset)
        time_stamps = np.concatenate((time_stamps, time_stamps[[0],:] + offset))    
    
    # split measurements
    measurements_list = []
    pointer = 0
    while pointer <= time_stamps[:,-1].max():
        
        new_measurements = []
        for t in range(TOSSIT_locations.shape[0]):
        
            # get measurements indices of time chunk
            idx = np.where((time_stamps[t] >= pointer) & (time_stamps[t] < pointer + chunk_size))[0]

            # save in list
            new_measurements.append(measurements[t][idx])

        # append location measurements to overall list        
        measurements_list.append(new_measurements)
        
        # iterate pointer
        pointer += chunk_size
                        
    return source_locs, measurements_list

def monte_carlo(source_locs, measurements_list, localizer_params):
    # instantiate localizer
    l = Localizer(**localizer_params)

    # calculate bearing
    theta_orig = to_positive_theta(math.atan2((-source_locs[-1,0]) - (-source_locs[0,0]), source_locs[-1,1] - source_locs[0,1]) * (180 / math.pi))
    theta = get_bearing([source_locs[0,1], source_locs[-1,1]], [-source_locs[0,0], -source_locs[-1,0]])

    # build hypergraph
    path_detected = False
    for i, measurement_set in enumerate(measurements_list):
        successful = l.set_measurements(measurement_set, **set_measurement_params)

        # if successful localize
        if not successful:
            continue
        else:
            # we've detected some portion of the path
            path_detected = True
            
            # associate/localize
            assoc, locs_est = l.associate_and_localize(method='partition', last_step=False)

            if i == 0:
                locs_ests = locs_est
            else:
                locs_ests = np.append(locs_ests, locs_est, axis=0)

    if path_detected:
        # make point vectors
        x = locs_ests[:, [1]]
        y = -locs_ests[:,0]

        # get endpoints of a regression on the
        # estimated locations
        reg = LinearRegression().fit(x,y)
        
        # get theta_hat
        x_ends = np.asarray([x[0], x[-1]])
        y_ends = reg.predict(x_ends)
        theta_hat_orig = to_positive_theta(math.atan2(y_ends[1] - y_ends[0], x_ends[1] - x_ends[0]) * (180 / math.pi))
        theta_hat = get_bearing(x_ends.squeeze(), y_ends)

        return path_detected, theta, theta_hat
    return path_detected, float('nan'), float('nan')

def monte_carlo_sim(n, var_list, localizer_params, set_measurement_params, data_gen_params):
    
    assert data_gen_params['beam_width'] == 0, "beam width must be 0 for monte carlo simulations"

    columns = ["k", "consistency_thresh", "method_thresh", "prune", "measurement_variance", "success_rate", "theta_error", "theta_error_std"]
    df = pd.DataFrame(columns=columns)

    for var in var_list:
        if args.background:
            print(f'working on -- var: {var} km...', flush=True)
        
        # keep track of theta
        theta_list = []

        # keep track of theta_hat
        theta_hat_list = []

        # keep track of successes
        success_list = []

        # generate data
        source_locs_list = []
        measurements_set_list = []
        for _ in range(n):
            # generate data
            source_locs, measurements_list = generate_trajectory(measurement_stats=(0, var), **data_gen_params)
            source_locs_list.append(source_locs)
            measurements_set_list.append(measurements_list)
        
        with Pool(processes=100) as pool:
            results = pool.starmap(monte_carlo, tqdm(zip(source_locs_list, measurements_set_list, itertools.repeat(localizer_params)), total=n, disable=args.background))

        # save values
        for result_set in results:
            success_list.append(result_set[0])
            theta_list.append(result_set[1])
            theta_hat_list.append(result_set[2])
        success_list = np.asarray(success_list)
        theta_list = np.asarray(theta_list)
        theta_hat_list = np.asarray(theta_hat_list)
        
        df = pd.concat([df, pd.DataFrame({
            "k" : [localizer_params["k"]],
            "consistency_thresh" : [localizer_params["consistency_thresh"]],
            "method_thresh" : [localizer_params["multilat"].method_thresh],
            "prune" : [localizer_params["prune"]],
            "measurement_variance" : [var],
            "success_rate" : [np.mean(success_list)],
            "theta_error" : [np.nanmean(np.abs(theta_list - theta_hat_list))],
            "theta_error_std" : [np.nanstd(np.abs(theta_list - theta_hat_list))],
        })])

    return df

if __name__ == "__main__":

    # path for results CSV
    path = os.path.join(PROJECT_ROOT_DIR, "experiments", "results", "path_finder", "path_finder_sim_results.csv")

    if args.simulate:
        
        # random number generators
        rng1 = np.random.default_rng(12345)
        rng2 = np.random.default_rng(54321)

        # parameters
        localizer_params = dict(k=4, multilat=MultilaterationOpt(method_thresh=0.95, rng=rng1), consistency_thresh=1000, prune=False)
        set_measurement_params = dict(adaptive=False, adaptive_max=5000, threshold_delta=500)
        data_gen_params = dict(rng=None, num_points=8, beam_width=0, timing_stats=(1, 0.5), repetition_time=0.5, num_repetitions=1, whale_speed=1.3, chunk_size=2, max_channel_offset=40)

        df = monte_carlo_sim(n=1,
                             var_list=[10, 100, 1000, 10000, 100000, 1000000], 
                             localizer_params=localizer_params,
                             set_measurement_params=set_measurement_params,
                             data_gen_params=data_gen_params)

        # append results if CSV exists
        if os.path.exists(path):
            df.to_csv(path, mode='a', index=False, header=False)
            df = pd.read_csv(path)
        else:
            df.to_csv(path, index=False)
    else:
        # check that we have simulated resuts in a CSV
        if os.path.exists(path):
            df = pd.read_csv(path)
        else:
            raise RuntimeError(f"'{path}' does not exist")

    # matplotlib settings
    matplotlib.rcParams.update({'font.size' : 16})

    # set of variances tested
    var_list = sorted(list(set(df['measurement_variance'])))

    # get values to plot
    theta_error = []
    theta_error_std = []
    for var in var_list:
        df_plot = df[df['measurement_variance'] == var]
        theta_error.extend(df_plot['theta_error'].tolist())
        theta_error_std.extend(df_plot['theta_error_std'].tolist())

    # get figs and axes
    figs = [plt.figure() for _ in range(1)]
    axs = [fig.gca() for fig in figs]

    # bearing error
    axs[0].errorbar([str(int(var / 1e6)) if (var / 1e6).is_integer() else str(var / 1e6) for var in var_list], theta_error, yerr=theta_error_std, capsize=10, fmt='-o', markersize=8, linewidth=3, elinewidth=3, capthick=3)
    axs[0].set_xlabel("Range Measurement Variance [km$^{2}$]")
    axs[0].set_ylabel("Absolute Theta Error [$^{\circ}$]")
    axs[0].set_title("Bearing Error")

    for ax in axs:
        ax.grid()

    if args.save_figs:
        fig_path = os.path.join(PROJECT_ROOT_DIR, "experiments", "results", "path_finder")
        figs[0].savefig(os.path.join(fig_path, "theta_error.png"))
