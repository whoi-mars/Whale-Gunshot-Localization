import os
import argparse
import itertools
import math
import pickle

from pathlib import Path
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import pandas as pd
import dask
from dask.distributed import Client, LocalCluster, progress
from PIL import Image


import whale_gunshot_localization.utils.math_tools as math_tools
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

def nested_list_max(l):
    maxx = -float('inf')
    for ll in l:
        if len(ll):
            ll = sorted(ll)
            if ll[-1] > maxx:
                maxx = ll[-1]
    return maxx

def is_in_bay(source_locs, bathym, map_origin, dx, dy):
    """
    Check if source locations are in CCB.
    
    Parameters
    ----------
    source_locs : array-like[array-like]
        matrix of generated source locations'
    bathym : PIL.Image
        geotiff of the bathymetry
    map_origin : array-like of shape 1 X 2
        pixels coordinates of the origin of the bathymetry map
    dx : float
        approximate change in meters when going in the X direction
    dy : float
        approximate change in meters when going in the Y direction
    
    Returns
    -------
    : bool
        whether all source_locs are in water (True) or not (False)
    """

    for loc in source_locs:
        # line cutting off locations on the open-ocean side of Provincetown
        line1 = loc[0] - 0.88434446716*loc[1] < -38672.47
        # line cutting off locations on the west side of the Cape Cod Canal
        line2 = loc[0] - 0.84448322668*loc[1] > 19385.3 
        pixel = bathym.getpixel((round(map_origin[1] + loc[1]/dx), round(map_origin[0] + loc[0]/dy)))
        if pixel == 147 or line1 or line2:
            return False
    return True

def get_bearing(x_ends, y_ends):
    """
    Given the x/y enpoints of a line, return the bearing.

    Parameters
    ----------
    x_ends : array-like
        list of start and end x coordinates of line
    y_ends : array-like
        list of start and end y coordinates of line

    Returns
    -------
    : float
        bearing of line
    """

    return math.atan2(y_ends[1] - y_ends[0], x_ends[1] - x_ends[0]) * (180 / math.pi)

def bearing_error(bearing1, bearing2):
    """
    Calculate the error between two bearings

    Parameters
    ----------
    bearing1 : array-like
        list of bearings
    bearing2 : array-like
        list of corresponding bearings

    Returns
    -------
    : array-like
        list of bearing errors
    """

    # make sure bearings are numpy arrays
    if not isinstance(bearing1, np.ndarray):
        bearing1 = np.asarray([bearing1])
    if not isinstance(bearing2, np.ndarray):
        bearing2 = np.asarray([bearing2])

    # calculate bearing error
    theta = np.abs(bearing1 - bearing2)
    theta[theta > 180] = 360 - theta[theta > 180]
    return theta

# def plot_localization(source_locs, locs_est=None, title='', buffer=6000): 
    
#     # load constants
#     TOSSIT_locations = np.asarray([config['TOSSIT']['TOSSIT_y'], config['TOSSIT']['TOSSIT_x']]).T
#     min_x = config['scaling']['min_x']
#     max_x = config['scaling']['max_x']
#     min_y = config['scaling']['min_y']
#     max_y = config['scaling']['max_y']
    
#     fig, ax = plt.subplots(1, 1)
#     ax.plot(TOSSIT_locations[:,1], TOSSIT_locations[:,0], '^', markersize=12, label='sensor', markeredgewidth=2)
#     ax.plot(source_locs[:,1], source_locs[:,0], 'o', label='ground truth')
#     if locs_est is not None:
#         ax.plot(locs_est[:,1], locs_est[:,0], 'x', label='estimate', markeredgewidth=2)
#     ax.set_xlabel("X [m]")
#     ax.set_ylabel("Y [m]")
#     ax.set_xlim([min_x-buffer, max_x+buffer])
#     ax.set_ylim([min_y-buffer, max_y+buffer])
#     ax.invert_yaxis()
#     ax.legend()
#     if len(title):
#         ax.set_title(title)
#     fig.savefig('test.png')

def generate_trajectory(num_points, beam_width=20, measurement_stats=(0, 500000), timing_stats=(1, 0.5), repetition_time=0.5, num_repetitions=1, whale_speed=1.3, chunk_size=2, max_channel_offset=40, rng=None, in_sensors=False):
    
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

    if in_sensors:
        min_x = np.min(TOSSIT_locations[:,1])
        max_x = np.max(TOSSIT_locations[:,1])
        min_y = np.min(TOSSIT_locations[:,0])
        max_y = np.max(TOSSIT_locations[:,0])
    else:
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
        time_stamps = np.concatenate((time_stamps, time_stamps[[0],:] + offset), axis=0)    
    
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
    
    # results dict
    results = {"path_detected": False,
               "theta": float('nan'),
               "theta_hat": float('nan'),}
    
    # instantiate localizer
    l = Localizer(**localizer_params)

    # calculate bearing
    # theta = get_bearing([source_locs[0,1], source_locs[-1,1]], [-source_locs[0,0], -source_locs[-1,0]])
    idx1 = np.argmin(source_locs[:,1])
    idx2 = np.argmax(source_locs[:,1])
    theta = get_bearing([source_locs[idx1,1], source_locs[idx2,1]], [-source_locs[idx1,0], -source_locs[idx2,0]])
    results["theta"] = theta

    # build hypergraph
    for i, measurement_set in enumerate(measurements_list):
        successful = l.set_measurements(measurement_set, **set_measurement_params)

        # if successful localize
        if not successful:
            continue
        else:
            # we've detected some portion of the path
            #path_detected = True
            results["path_detected"] = True

            # associate/localize
            assoc, locs_est = l.associate_and_localize(method='partition', reduce_dups=False, last_step=False)

            if i == 0:
                locs_ests = locs_est
            else:
                locs_ests = np.append(locs_ests, locs_est, axis=0)

    if results["path_detected"]:
        # make point vectors
        x = locs_ests[:, [1]]
        y = -locs_ests[:,0]

        pcr = make_pipeline(StandardScaler(), PCA(n_components=1), LinearRegression())
        pcr.fit(x, y)
        pca = pcr.named_steps["pca"]

        x_ends = np.asarray([[np.min(x)], [np.max(x)]])
        y_ends = pcr.predict(x_ends)
        theta_hat = get_bearing(x_ends.squeeze(), y_ends)
        results["theta_hat"] = theta_hat

    return results

def monte_carlo_sim(n, max_time_offset_list, std_list, localizer_params, set_measurement_params, data_gen_params):
    
    assert data_gen_params['beam_width'] == 0, "beam width must be 0 for monte carlo simulations"

    columns = ["k", "consistency_thresh", "method_thresh", "prune", "max_time_offset", "std", "success_rate", "theta", "theta_hat"]
    df = pd.DataFrame(columns=columns)

    # load map
    Image.MAX_IMAGE_PIXELS = 729744000
    bathym = Image.open(os.path.join(config['dataset']['data_directory'], "mikesbathym.tif"))
    map_origin = config['TOSSIT']['map_origin']
    dy = config['TOSSIT']['dy']
    dx = config['TOSSIT']['dx']
    min_x = config['scaling']['min_x']
    max_x = config['scaling']['max_x']
    min_y = config['scaling']['min_y']
    max_y = config['scaling']['max_y']

    for std in std_list:

        # change consistency threshold based on measurement variance
        localizer_params_final = localizer_params.copy()
        localizer_params_final['consistency_thresh'] = localizer_params_final['consistency_thresh'][std]

        for max_time_offset in max_time_offset_list:
            if args.background:
                print(f'working on -- max_time_offset: {max_time_offset}, std: {std} m...', flush=True)
            
            # generate data
            source_locs_list = []
            measurements_set_list = []
            for _ in range(n):
                while True:
                    source_locs, measurements_list = generate_trajectory(measurement_stats=(0, std ** 2), max_channel_offset=max_time_offset, **data_gen_params)
                    if all([abs(nested_list_max(l)) <= config['scaling']['max_r'] for l in measurements_list]) \
                       and is_in_bay(source_locs, bathym, map_origin, dx, dy):
                        break
                source_locs_list.append(source_locs)
                measurements_set_list.append(measurements_list)
            
            # delayed for loop using dask
            results = []
            for i in range(n):
                res = dask.delayed(monte_carlo)(source_locs_list[i],
                                                measurements_set_list[i],
                                                localizer_params_final)
                results.append(res)

            # parallelize MC
            results = client.compute(results)
            if not args.background:
                progress(results)
            results = client.gather(results)

            #with Pool(processes=100) as pool:
            #    results = pool.starmap(monte_carlo, tqdm(zip(source_locs_list, measurements_set_list, itertools.repeat(localizer_params)), total=n, disable=args.background, postfix={'max_time_offset' : max_time_offset, 'std' : std}))

            # save values
            theta_list = []
            theta_hat_list = []
            success_list = []
            for result in results:
                success_list.append(result["path_detected"])
                theta_list.append(result["theta"])
                theta_hat_list.append(result["theta_hat"])
            
            df = pd.concat([df, pd.DataFrame({
                "k" : [localizer_params_final["k"] for _ in range(n)],
                "consistency_thresh" : [localizer_params_final["consistency_thresh"] for _ in range(n)],
                "method_thresh" : [localizer_params_final["multilat"].method_thresh for _ in range(n)],
                "prune" : [localizer_params_final["prune"] for _ in range(n)],
                "max_time_offset" : [max_time_offset for _ in range(n)],
                "std" : [std for _ in range(n)],
                "success_rate" : success_list,
                "theta" : theta_list,
                "theta_hat" : theta_hat_list,
            })])

    return df

if __name__ == "__main__":

    # path for results CSV
    path = os.path.join(PROJECT_ROOT_DIR, "experiments", "results", "path_finder", "path_finder_sim_results.csv")
    fig_path = os.path.join(PROJECT_ROOT_DIR, "experiments", "results", "path_finder")

    # set up results directory
    Path(fig_path).mkdir(exist_ok=True)

    ##########################################
    #          simulate/load results         #
    ##########################################

    if args.simulate:
        
        # dask setup
        cluster = LocalCluster(n_workers=100, processes=True)
        client = Client(cluster)

        # random number generators
        rng1 = np.random.default_rng(12345)
        rng2 = np.random.default_rng(54321)

        # parameters
        localizer_params = dict(k=4, multilat=MultilaterationOpt(method_thresh=float('inf'), rng=rng1), consistency_thresh={0: 2, 250: 1500, 500: 2500, 750: 3500, 1000: 3500}, prune=False)
        set_measurement_params = dict(adaptive=False, adaptive_max=5000, threshold_delta=500)
        data_gen_params = dict(rng=rng2, num_points=8, beam_width=0, timing_stats=(1, 0.5), repetition_time=0.5, num_repetitions=1, whale_speed=1.3, chunk_size=2, in_sensors=False)

        df = monte_carlo_sim(n=300,
                             max_time_offset_list=[0, 10],
                             std_list=[0, 250, 500, 750, 1000],
                             localizer_params=localizer_params,
                             set_measurement_params=set_measurement_params,
                             data_gen_params=data_gen_params)
        df.to_csv(path, index=False)
    else:
        # check that we have simulated resuts in a CSV
        if os.path.exists(path):
            df = pd.read_csv(path)
        else:
            raise RuntimeError(f"'{path}' does not exist")

    ##########################################
    #               make plots               #
    ##########################################

    # matplotlib settings
    matplotlib.rcParams.update({'font.size' : 16})

    # set of variances tested
    std_list = sorted(list(set(df['std'])))
    max_time_offset_list = sorted(list(set(df['max_time_offset'])))

    # make variance integer if possible
    def intify(x):
        if isinstance(x, float) and x.is_integer():
            return str(int(x))
        else:
            return str(np.around(x, 2))

    figs = [plt.figure() for _ in range(2)]
    axs = [fig.gca() for fig in figs]

    df["theta_error"] = bearing_error(df["theta"], df["theta_hat"]).squeeze().tolist()
    sns.boxplot(x=df['std'], 
                y=df['theta_error'], 
                hue=df['max_time_offset'], 
                ax=axs[0], 
                showfliers=False,
                showmeans=True, 
                linewidth=1, 
                meanprops={"marker":"s","markerfacecolor":"white", "markeredgecolor":"blue"},)
    axs[0].set_xticks(np.arange(len(std_list)), [intify(std) for std in std_list])
    axs[0].set_xlabel("Range Measurement Standard Deviation [m]")
    axs[0].set_ylabel("Absolute Theta Error [$^{\circ}$]")
    axs[0].set_title("Bearing Error")
    axs[0].legend(title='max channel offset [s]')
    axs[0].set_axisbelow(True)
    # sns.stripplot(data=df, ax=axs[0], x="std", y="theta_error", hue="max_time_offset", palette=sns.color_palette("tab10"), dodge=True)
    # axs[0].legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)

    theta_err_dict = dict()
    vals = np.zeros((len(std_list), len(max_time_offset_list)))
    annot = np.zeros((len(std_list), len(max_time_offset_list)))
    for i, std in enumerate(std_list):
        for j, toff in enumerate(max_time_offset_list):
        
            d = df[(df["std"] == std) & (df["max_time_offset"] == toff)]

            # get error for category
            err = d["theta_error"]
            
            # calculate IQR
            q1 = np.percentile(err, 25)
            q3 = np.percentile(err, 75)
            IQR = q3 - q1

            # get
            vals[i, j] = np.percentile(err[err > 1.5*IQR], 90)
            annot[i, j] = len(err[err > 1.5*IQR]) / len(err) 

    sns.heatmap(vals,
                annot=annot,
                xticklabels=max_time_offset_list,
                yticklabels=std_list,
                cbar_kws={'label': '90th Percentile Outliers'},
                ax=axs[1])
    axs[1].set_xlabel("Max Time Offset [s]")
    axs[1].set_ylabel("Measurement Standard Deviation [m]")
    axs[1].invert_yaxis()


    for i, ax in enumerate(axs):
        if i == 1:
            continue
        ax.grid()

    if args.save_figs:
        figs[0].savefig(os.path.join(fig_path, "theta_error.png"))
        figs[1].savefig(os.path.join(fig_path, "outlier_analysis.png"))

    # make table of means and stds
    dff = df.groupby(by=['std', 'max_time_offset']).agg({'theta_error' : ['mean', 'std']})
    print(dff)