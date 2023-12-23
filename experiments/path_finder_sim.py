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
import whale_gunshot_localization.sim_tools.sim_datagen as sim_datagen
import whale_gunshot_localization.sim_tools.sim_data_checks as sim_data_checks

# parse input arguments
parser = argparse.ArgumentParser(description="Run Monte Carlo simulations for data association/localization for a train of calls")
parser.add_argument('--simulate', action='store_true',
                    help="simulate rather than use previous results if available (default: false)")
parser.add_argument('--save_figs', action='store_true',
                    help="save figures (default: false)")
parser.add_argument('--suppress_warnings', action='store_true',
                    help="tell Python to suppress warnings")
parser.add_argument('--background', '-b', action='store_true',
                    help='silence the progress bar')
args = parser.parse_args()

# suppress warnings
if args.suppress_warnings:
    import warnings
    warnings.filterwarnings("ignore")

##################################################
#               helper functions                 #
##################################################

def nested_list_max(l):
    """
    Get the max value from a list of lists where the sublists
    may be of different sizes.

    Parameters
    ----------
    l : List[array-like]
        list of ragged lists
    
    Returns
    -------
    : float
        maximum values
    """

    maxx = -float('inf')
    for ll in l:
        if len(ll):
            ll = sorted(ll)
            if ll[-1] > maxx:
                maxx = ll[-1]
    return maxx

def ragged_concat(l):
    if not isinstance(l[0], np.ndarray):
        return l
    else:
        res = []
        for sl in l:
            for e in sl:
                res.append(e)
        return np.asarray(res)


def filter_oob_locs(locs, buffer=15000):
    """
    Filter out locations that are outside of the specified training region
    with a specified buffer.

    Parameters
    ----------
    locs : array-like, of shape N X 2
        locations to filter
    buffer : float
        how much to buffer out the sides of the training region

    Return
    ------
    locs : array-like, of shape M X 2
        filter locations
    : int
        how many locations were filtered out
    """
    
    idx = np.where((config['scaling']['min_y'] - buffer <= locs[:,0]) & \
                   (locs[:,0] <= (config['scaling']['max_y'] + buffer)) & \
                   (config['scaling']['min_x'] - buffer <= locs[:,1]) & \
                   (locs[:,1] <= (config['scaling']['max_x'] + buffer)))[0]
    return locs[idx,:], len(locs) - len(idx)

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

##################################################
#                  monte carlo                   #
##################################################

def monte_carlo(source_locs, measurements_list, source_assocs_list, TOSSIT_associations_list, localizer_params):
    """
    Run an instance of the monte carlo simulation.

    Parameters
    ----------
    source_locs : array-like, of shape N X 2
        source locations in the path
    measurements_list : List[List[array-like]]
        list of measurements collected at each source location in the path
        on all sensors
    localizer_params : dict
        dictionary of parameters for localizer object
        - k : int --> k value for group-k consistency check
        - multilat : MultilaterationBase --> instantiated multilateration object to use
        - consistency_thresh : float --> dict which maps std --> group-k threshhold
        - prune : bool --> ensure that all measurements in k-1 subgroups intersect to count as a consistent set

    Returns
    -------
    results : dict
        dictionary with the following entries
        - path_detected --> whether or not a path was detected
        - theta --> true source bearing
        - theta_hat --> estimated source bearing
        - num_OOB --> number of source location estimates out of bounds of the region 
    """

    # results dict
    results = {"path_detected": False,
               "theta": float('nan'),
               "theta_hat": float('nan'),
               "best_theta_hat": float('nan'),
               "num_OOB": float('nan'),}
    
    # instantiate localizer
    l = Localizer(**localizer_params)

    # calculate bearing
    # theta = get_bearing([source_locs[0,1], source_locs[-1,1]], [-source_locs[0,0], -source_locs[-1,0]])
    idx1 = np.argmin(source_locs[:,1])
    idx2 = np.argmax(source_locs[:,1])
    theta = get_bearing([source_locs[idx1,1], source_locs[idx2,1]], [-source_locs[idx1,0], -source_locs[idx2,0]])
    results["theta"] = theta

    # instantiate localizer for baseline
    localizer = localizer_params['multilat']
    TOSSIT_locations = np.asarray([config['TOSSIT']['TOSSIT_y'], config['TOSSIT']['TOSSIT_x']]).T
    localizer.set_map({
        'TOSSIT_locations' : TOSSIT_locations,
        'min_x' : config['scaling']['min_x'],
        'max_x' : config['scaling']['max_x'],
        'min_y' : config['scaling']['min_y'],
        'max_y' : config['scaling']['max_y'],
    })

    # build hypergraph
    best_locs = []
    for i, measurement_set in enumerate(measurements_list):
        successful = l.set_measurements(measurement_set, **set_measurement_params)
        # if successful localize
        if not successful:
            continue
        else:
            # we've detected some portion of the path
            results["path_detected"] = True

            # associate/localize
            assoc, locs_est = l.associate_and_localize(method='partition', reduce_dups=False, last_step=False)
            locs_est, n = filter_oob_locs(locs_est)
            results["num_OOB"] = n
            assoc_flat = ragged_concat(source_assocs_list[i])
            measurements_flat = ragged_concat(measurement_set[i])
            TOSSIT_associations_flat = ragged_concat(TOSSIT_associations_list[i])
            curr_source_locs_idx = np.asarray(list(set(assoc_flat)), dtype=int)
            curr_source_locs = source_locs[curr_source_locs_idx]

            # calculate best locations if we know the assocaitions
            num_ests = min(locs_est.shape[0], len(curr_source_locs_idx))
            source_loc_combs = np.asarray(list(map(list, itertools.permutations(curr_source_locs))))
            source_loc_idx_combs = np.asarray(list(map(list, itertools.permutations(curr_source_locs_idx))))
            errors_matrix = np.sqrt(((source_loc_combs[:,:num_ests,:] - locs_est[np.newaxis,:num_ests,:]) ** 2).sum(axis=2)).sum(axis=1)
            detected_sources = source_loc_idx_combs[np.argmin(errors_matrix)][:num_ests]

            assoc_flat = ragged_concat(source_assocs_list[i])
            measurements_flat = ragged_concat(measurement_set)
            TOSSIT_associations_flat = ragged_concat(TOSSIT_associations_list[i])
            #print("MF: ", measurements_flat)
            #print("TF: ", TOSSIT_associations_flat)
            for sidx in detected_sources:
                idx = np.where(assoc_flat == sidx)[0]
                if len(idx) < 3:
                    continue
                _, loc = localizer.localize(measurements_flat[idx], TOSSIT_associations_flat[idx].astype(int))
                best_locs.append(loc)

            if i == 0:
                locs_ests = locs_est
            else:
                locs_ests = np.append(locs_ests, locs_est, axis=0)
    best_locs = np.asarray(best_locs)

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

        if len(best_locs) > 0:
            # make point vectors
            x = best_locs[:, [1]]
            y = -best_locs[:,0]

            pcr = make_pipeline(StandardScaler(), PCA(n_components=1), LinearRegression())
            pcr.fit(x, y)
            pca = pcr.named_steps["pca"]

            x_ends = np.asarray([[np.min(x)], [np.max(x)]])
            y_ends = pcr.predict(x_ends)
            theta_hat = get_bearing(x_ends.squeeze(), y_ends)
            results["best_theta_hat"] = theta_hat

    return results

def monte_carlo_sim(n, max_time_offset_list, std_list, localizer_params, set_measurement_params, data_gen_params):
    """
    Function to run the monte carlo simulations

    Parameters
    ----------
    n : int
        number of monte carlo runs to do per std/time offsset combo
    max_time_offset_list : List[float]
        list of maximum time offsets to randomly shift each channel by (relative to the first)
    std_list : List[float]
        list of range measurement noise stds
    localizer_params : dict
        dictionary of parameters for localizer object
        - k : int --> k value for group-k consistency check
        - multilat : MultilaterationBase --> instantiated multilateration object to use
        - consistency_thresh : float --> dict which maps std --> group-k threshhold
        - prune : bool --> ensure that all measurements in k-1 subgroups intersect to count as a consistent set
    set_measurement_params : dict
        dictionary of parameters for the set_measurement method of the localizer object
        - adaptive : bool --> whether or not we want to increase the consistency threshold if nothing was deemed group-k consistent
        - adaptive_delta : float --> how much to adaptively increase consistency threshold by
        - adaptive_max : float --> maximum consistency threshold before giving up in adaptive mode
    data_gen_parames : dict
        dictionary of parameters for simulated data generation
        - rng : numpy.random._generator.Generator --> RNG object
        - num_points : int --> number of points in the path
        - beam_width : float --> arc within which we generate the next source point
        - timeing_stats : Tuple[float, float] --> mean/std of call generation (in minutes)
        - repetition_time : float --> time between repetitions when > 1
        - repetitions : int --> number of repetitions
        - whale_speed : float --> speed of simulated whale in km/hr
        - chunk_size : float --> chunk of data to analyze at once (in minutes)
        - in_sensors : bool --> whether or not to generate soure locations only in the sensor network

    Returns
    -------
    df : pandas.DataFrame
        dataframe of results
    """

    assert data_gen_params['beam_width'] == 0, "beam width must be 0 for monte carlo simulations"

    columns = ["k", "consistency_thresh", "method_thresh", "prune", "max_time_offset", "std", "success_rate", "theta", "theta_hat", "num_OOB"]
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
        localizer_params_final['multilat'] = localizer_params_final['multilat'][std]


        for max_time_offset in max_time_offset_list:
            if args.background:
                print(f'working on -- max_time_offset: {max_time_offset}, std: {std} m...', flush=True)
            
            # generate data
            source_locs_list = []
            source_assocs_set_list = []
            measurements_set_list = []
            TOSSIT_assocs_set_list = []
            for _ in range(n):
                while True:
                    source_locs, measurements_list, source_assocaitions_list, TOSSIT_associations_list = sim_datagen.generate_trajectory(measurement_stats=(0, std ** 2), max_channel_offset=max_time_offset, **data_gen_params)
                    if all([abs(nested_list_max(l)) <= config['scaling']['max_r'] for l in measurements_list]) \
                       and sim_data_checks.is_in_bay(source_locs, bathym, map_origin, dx, dy):
                        break
                source_locs_list.append(source_locs)
                source_assocs_set_list.append(source_assocaitions_list)
                measurements_set_list.append(measurements_list)
                TOSSIT_assocs_set_list.append(TOSSIT_associations_list)
            
            # delayed for loop using dask
            results = []
            for i in range(n):
                res = dask.delayed(monte_carlo)(source_locs_list[i],
                                                measurements_set_list[i],
                                                source_assocs_set_list[i],
                                                TOSSIT_assocs_set_list[i],
                                                localizer_params_final)
                results.append(res)

            # parallelize MC
            results = client.compute(results)
            if not args.background:
                progress(results)
            results = client.gather(results)

            # save values
            theta_list = []
            theta_hat_list = []
            best_theta_hat_list = []
            success_list = []
            num_OOB_list = []
            for result in results:
                success_list.append(result["path_detected"])
                theta_list.append(result["theta"])
                theta_hat_list.append(result["theta_hat"])
                best_theta_hat_list.append(result["best_theta_hat"])
                num_OOB_list.append(result["num_OOB"])
            
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
                "best_theta_hat" : best_theta_hat_list,
                "num_OOB": num_OOB_list,
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
        localizer_params = dict(k=4, multilat={i : MultilaterationOpt(method_thresh=0.95, rng=rng1) for i in [0, 15, 30, 45, 60, 75, 750]}, consistency_thresh={0: 2, 15: 20, 30: 75, 45: 55, 60: 70, 75: 85, 750: 3000}, prune=False)
        set_measurement_params = dict(adaptive=False, adaptive_max=5000, threshold_delta=500)
        data_gen_params = dict(rng=rng2, num_points=8, beam_width=0, timing_stats=(1, 0.5), repetition_time=0.5, num_repetitions=1, whale_speed=1.3, chunk_size=2, in_sensors=False)

        df = monte_carlo_sim(n=100,
                             max_time_offset_list=[0, 10],
                             std_list=[0, 15, 30, 45, 60, 75, 750],
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

    #---------------------------------------------------#
    #-------- error distributions and percentiles ------#
    #---------------------------------------------------#

    df["theta_error"] = bearing_error(df["theta"], df["theta_hat"]).squeeze().tolist()
    df["best_theta_error"] = bearing_error(df["theta"], df["best_theta_hat"]).squeeze().tolist()
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

    figg, axx = plt.subplots(len(std_list), len(max_time_offset_list), figsize=(26,26))
    if not isinstance(axx, np.ndarray):
        axx = np.asarray([[axx]])
    elif len(axx.shape) < 2:
        axx = np.asarray([axx])

    unsupervised_per = np.zeros((len(std_list), len(max_time_offset_list)))
    # annot = np.zeros((len(std_list), len(max_time_offset_list)))
    for i, std in enumerate(std_list):
        for j, toff in enumerate(max_time_offset_list):
        
            d = df[(df["std"] == std) & (df["max_time_offset"] == toff)]

            # get error for category
            err = d["theta_error"]
            err_best = d["best_theta_error"]
            
            # store 95th percentile error
            unsupervised_per[i,j] = np.percentile(err, 90)

            _,bins,_ = axx[i,j].hist(err, alpha=0.5, bins=300, label='unsupervised')
            axx[i,j].hist(err_best, alpha=0.5, bins=bins, label='ideal')
            axx[i,j].set_title(f"$\sigma$={std} m, " + "$t_{offset}$=" + f"{toff} s", fontsize=22)
            axx[i,j].tick_params(axis='x', labelsize=16)
            axx[i,j].tick_params(axis='y', labelsize=16)
            axx[i,j].yaxis.get_offset_text().set_fontsize(14)
            axx[i,j].xaxis.get_offset_text().set_fontsize(14)

    handles, labels = axx[0,0].get_legend_handles_labels()
    figg.legend(handles, labels, loc='upper center', prop={'size': 28})
    figg.subplots_adjust(wspace=0.35, hspace=0.35)
    figg.text(0.5, 0.04, "Bearing Error [degrees]", ha='center', va='center', fontsize=28)
    figg.text(0.05, 0.5, "Example Count", ha='center', va='center', rotation=90, fontsize=28)

    sns.heatmap(unsupervised_per,
                # annot=annot,
                xticklabels=max_time_offset_list,
                yticklabels=std_list,
                cbar_kws={'label': '95th Percentile Outliers'},
                ax=axs[1])
    axs[1].set_xlabel("Max Time Offset [s]")
    axs[1].set_ylabel("Measurement Standard Deviation [m]")
    axs[1].invert_yaxis()

    #---------------------------------------------------#
    #------------ finalize plot/data and save ----------#
    #---------------------------------------------------#

    for i, ax in enumerate(axs):
        if i == 1:
            continue
        ax.grid()

    if args.save_figs:
        figs[0].savefig(os.path.join(fig_path, "theta_error.png"))
        figs[1].savefig(os.path.join(fig_path, "unsupervised_percentile.png"))
        figg.savefig(os.path.join(fig_path, "hists.png"))

    # make table of means and stds
    def perc90(iterable):
        a = np.asarray(iterable)
        return np.percentile(a, 90)
    dff = df.groupby(by=['std', 'max_time_offset']).agg({'theta_error' : ['mean', 'std', perc90],
                                                         "best_theta_error" : ['mean', 'std', perc90],
                                                         'num_OOB': ['mean', 'std'],})
    print(dff)