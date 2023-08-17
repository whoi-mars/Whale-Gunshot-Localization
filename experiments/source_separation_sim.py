import os
import itertools
import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm
import dask
from dask.diagnostics import ProgressBar
from dask.distributed import Client, LocalCluster
from PIL import Image

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from whale_gunshot_localization.utils.experimental import Localizer, MultilaterationOpt, MultilaterationGrid
import whale_gunshot_localization.utils.math_tools as math_tools
from whale_gunshot_localization import config, PROJECT_ROOT_DIR

# parse input arguments
parser = argparse.ArgumentParser(description="Run Monte Carlo simulations for data association/localization algorithm")
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
    for loc in source_locs:
        if bathym.getpixel((round(map_origin[1] + loc[1]/dx), round(map_origin[0] + loc[0]/dy))) == 147:
            return False
    return True

def generate_measurements(num_sources, rng, var=10, num_delete=0, in_sensors=False):
    """
    Generate some synthetic measurements to test with data association/localization algoritms.
    
    Parameters
    ----------
    num_sources : int
        maximum number of sources
    var : float
        variance of Gaussian noise added to range measurements
    num_delete : int
        maximum number of measurements to hide/delete at each source
        
    Returns
    -------
    range_measurements : List[array-like]
        each sublist contains range measurements and is associated with a particular TOSSIT
    source_associations : List[array-like]
        same shape as range_measurements. each sublist contains numbers which associate the
        range_measurement in the corresponding spot in the data structure with a source.
    TOSSIT_association : List[array-like]
        same shape as range_measurements. each sublist contains numbers which associate the
        range_measurement in the corresponding spot in the data structure with a TOSSIT.    
    source_locs : array-like[array-like]
        matrix of generated source locations
    """ 
    
    # check inputs
    assert num_sources > 0, "number of sources must be non-negative"
    assert num_delete >= 0, "max signals to delete at each sensor must be non-negative"
    
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
    source_locs = np.concatenate((rng.uniform(min_y, max_y, size=(num_sources,1)), rng.uniform(min_x, max_x, size=(num_sources,1))), axis=1)
    
    # generate range measurements and log associations
    range_measurements, TOSSIT_associations, del_list, source_associations = [], [], [], []
    for t in range(TOSSIT_locations.shape[0]):
        
        # calculate range measurements from all sources to TOSSIT t, adding Gaussian noise
        r = np.abs(np.linalg.norm(source_locs - TOSSIT_locations[t,:], axis=1) + \
            rng.normal(loc=0, scale=np.sqrt(var), size=num_sources))
        range_measurements.append(r)
        
        # generate arrays for the TOSSIT associations of the measuremnts.
        TOSSIT_associations.append(np.ones((num_sources,), dtype=int) * t)
        
        # generate arrays of source associations for the measurements
        source_associations.append(np.arange(num_sources))
        
    # delete from each source
    for s in range(num_sources):
        delete_num = rng.choice([0, num_delete])
        if delete_num:
            to_delete = rng.choice(np.arange(TOSSIT_locations.shape[0]), replace=False, size=delete_num)
            for d in to_delete:
                idx = np.argwhere(source_associations[d] == s)
                range_measurements[d] = np.delete(range_measurements[d], idx)
                source_associations[d] = np.delete(source_associations[d], idx)
                TOSSIT_associations[d] = np.delete(TOSSIT_associations[d], idx)
                
    # make sure smallest association value is 0
    bias = min([a[0] for a in source_associations if len(a)])
    if bias > 0:
        for i, a in enumerate(source_associations):
            source_associations[i] = a - bias
    
    return range_measurements, source_associations, TOSSIT_associations, source_locs


def monte_carlo(measurements, s_assocs, t_assocs, s_locs, localizer_params):
    
    # dask setup
    #cluster = LocalCluster(n_workers=100, processes=True)
    #client = Client(cluster)

    # boolean results
    over_predict_sources = False
    FN = False
    FP = False

    # get number of possible associations
    assoc_flat = np.concatenate(s_assocs)
    possible_associations = []
    for p in range(max(assoc_flat) + 1):
        group = np.where(assoc_flat == p)[0]
        if len(group) >= localizer_params["k"]:
            possible_associations.append(p)
    possible = len(possible_associations) > 0

    # instantiate localizer
    L = Localizer(**localizer_params)

    # perform data assoc/loc
    successful = L.set_measurements(measurements, **set_measurement_params)
    if not successful:
        # check for FN
        if possible:
            FN = True
        return float('nan'), FN, FP, float('nan'), float('nan'), float('nan')
    else:
        # check for FP
        if not possible:
            FP = True
            return float('nan'), FN, FP, float('nan'), float('nan'), float('nan')
        assocs_est, locs_est = L.associate_and_localize(method='partition', reduce_dups=True, last_step=True)

    # if we predict too many sources return
    if locs_est.shape[0] > s_locs.shape[0]:
        over_predict_sources = True
        return over_predict_sources, FN, FP, float('nan'), float('nan'), float('nan')

    # calculate location errors
    num_ests = locs_est.shape[0]
    source_loc_combs = np.asarray(list(map(list, itertools.permutations(s_locs))))
    source_loc_idx_combs = np.asarray(list(map(list, itertools.permutations(np.arange(s_locs.shape[0])))))
    errors_matrix = np.sqrt(((source_loc_combs[:,:num_ests,:] - locs_est[np.newaxis,:,:]) ** 2).sum(axis=2)).sum(axis=1)
    res = np.sqrt(((source_loc_combs[np.argmin(errors_matrix),:num_ests,:] - locs_est[np.newaxis,:,:]) ** 2).sum(axis=2)).flatten()

    # calculate best possible localization errors with correct associations
    detected_sources = sorted(source_loc_idx_combs[np.argmin(errors_matrix)])
    measurements_flat = np.concatenate(measurements)
    TOSSIT_flat = np.concatenate(t_assocs)
    localizer = MultilaterationOpt()
    TOSSIT_locations = np.asarray([config['TOSSIT']['TOSSIT_y'], config['TOSSIT']['TOSSIT_x']]).T
    localizer.set_map({
        'TOSSIT_locations' : TOSSIT_locations,
        'min_x' : config['scaling']['min_x'],
        'max_x' : config['scaling']['max_x'],
        'min_y' : config['scaling']['min_y'],
        'max_y' : config['scaling']['max_y'],
    })

    best_locs = []
    for sidx in detected_sources:
        idx = np.where(assoc_flat == sidx)[0]
        _, loc = localizer.localize(measurements_flat[idx], TOSSIT_flat[idx])
        best_locs.append(loc)
    best_locs = np.asarray(best_locs)
    best_res = np.sqrt(((best_locs - s_locs) ** 2).sum(axis=1))


    # calculate percentage of possible sources localized
    percent_possible_detections = len(assocs_est) / len(possible_associations)
    
    return over_predict_sources, FN, FP, percent_possible_detections, res, best_res


def run_monte_carlo(n, std_list, num_sources_list, sparse_distance, localizer_params, set_measurement_params, data_gen_params):
    
    columns = ["num_sources", "std", "over_predict_sources", "FN", "FP", "percent_possible_detections", "localization_error", "best_localization_error"]
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
        for num_sources in num_sources_list:

            if args.background:
                print(f'working on -- std: {std} m, num_sources: {num_sources}...', flush=True)

            measurements_list = []
            source_associations_list = []
            TOSSIT_associations_list = []
            source_locs_list = []
            for _ in range(n):
                while True:
                    measurements, source_associations, TOSSIT_associations, source_locs = generate_measurements(num_sources=num_sources, var=std ** 2, **data_gen_params)
                    if nested_list_max(measurements) <= config['scaling']['max_r'] \
                       and math_tools.is_sparse_locs(source_locs, thresh=sparse_distance) \
                       and is_in_bay(source_locs, bathym, map_origin, dx, dy):
                        break
                measurements_list.append(measurements)
                source_associations_list.append(source_associations)
                TOSSIT_associations_list.append(TOSSIT_associations)
                source_locs_list.append(source_locs)

            # delayed for loop using dask
            results = []
            for i in range(n):
                res = dask.delayed(monte_carlo)(measurements_list[i],
                                                source_associations_list[i],
                                                TOSSIT_associations_list[i],
                                                source_locs_list[i],
                                                localizer_params)
                results.append(res)

            # parallelize MC
            if not args.background:
                ProgressBar().register()
            results = dask.compute(*results, scheduler='processes')
            
            source_over_predict_list = []
            localization_error_list = []
            FN_list = []
            FP_list = []
            best_localization_error_list = []
            percent_possible_detections_list = []
            for result in results:
                
                # save results
                source_over_predict_list.append(result[0]),
                FN_list.append(result[1])
                FP_list.append(result[2])
                percent_possible_detections_list.append(result[3])
                
                # save localization errors in a string format
                if isinstance(result[4], np.ndarray):
                    loc_err_str = ""
                    best_loc_err_str = ""
                    for err, best_err in zip(result[4], result[5]):
                        loc_err_str += f"{err};"
                        best_loc_err_str += f"{best_err};"
                    localization_error_list.append(loc_err_str)
                    best_localization_error_list.append(best_loc_err_str)
                else:
                    localization_error_list.append(result[4])
                    best_localization_error_list.append(result[5])

            # append results to dataframe
            df = pd.concat([df, pd.DataFrame({"num_sources": [num_sources for _ in range(n)],
                                              "std": [std for _ in range(n)],
                                              "over_predict_sources": source_over_predict_list,
                                              "FN": FN_list,
                                              "FP": FP_list,
                                              "percent_possible_detections": percent_possible_detections_list,
                                              "localization_error": localization_error_list,
                                              "best_localization_error":best_localization_error_list,})], ignore_index=True)
    
    return df

if __name__ == "__main__":
    
    # path for results CSV
    path = os.path.join(PROJECT_ROOT_DIR, "experiments", "results", "data_assoc_and_loc", "data_association_and_localization_sim_results.csv")

    ##########################################
    #          simulate/load results         #
    ##########################################

    # path for results CSV
    path = os.path.join(PROJECT_ROOT_DIR, "experiments", "results", "data_assoc_and_loc", "data_association_and_localization_sim_results.csv")

    if args.simulate:

        # set random seed
        # make two of these for monte carlo and localizer and make an internal one for monte
        # carlo which does either the source locations or variance
        rng1 = np.random.default_rng(1524)
        rng2 = np.random.default_rng(1524)

        # parameters for localizer and data_generator
        localizer_params = dict(k=4, multilat=MultilaterationOpt(method_thresh=float('inf'), rng=rng1), consistency_thresh=2000, dup_thresh=3000, prune=False)
        set_measurement_params = dict(adaptive=False, adaptive_max=5000, threshold_delta=500)
        data_gen_params = dict(num_delete=0, rng=rng2, in_sensors=True)

        # run MC
        df = run_monte_carlo(n=150,
                             std_list=[0, 250, 500, 750, 1000],
                             num_sources_list=range(1,6),
                             sparse_distance=10,
                             localizer_params=localizer_params,
                             set_measurement_params=set_measurement_params,
                             data_gen_params=data_gen_params)

        df.to_csv(path, index=False)

    df = pd.read_csv(path)
    
    ##########################################
    #               make plots               #
    ##########################################

    # matlab settings
    matplotlib.rcParams.update({'font.size': 16})

    # varying parameters
    # var_list = sorted(list(set(df['var'])))
    std_list = sorted(list(set(df['std'])))
    num_sources_max = df['num_sources'].max()
    num_sources_min = df['num_sources'].min()

    # get figs and axes
    figs = [plt.figure() for _ in range(7)]
    axs = [fig.gca() for fig in figs]

    # make variance integer if possible
    def intify(x):
        if isinstance(x, float) and x.is_integer():
            return str(int(x))
        else:
            return str(np.around(x, 2))

    #-----------------------------------#
    #------- plot location error -------#
    #-----------------------------------#
    
    # collect individual localization errors
    num_sources_list = []
    location_error_list = []
    best_location_error_list = []
    std_list = []
    len_detects = []
    for _, row in df.iterrows():
        if isinstance(row['localization_error'], str):
            for err, best_err in zip(row['localization_error'].split(';')[:-1], row['best_localization_error'].split(';')[:-1]):
                len_detects.append(len(row['localization_error'].split(';')[:-1]))
                num_sources_list.append(row['num_sources'])
                location_error_list.append(float(err))
                best_location_error_list.append(float(best_err))
                std_list.append(row['std'])
    
    df_loc = pd.DataFrame({'num_sources': num_sources_list,
                           'loc_error': location_error_list,
                           'best_loc_error': best_location_error_list,
                           'std': std_list,})

    # location stats
    loc_cols = df_loc.groupby(by=['std', 'num_sources']).agg({'loc_error': ['mean', 'std'], 'best_loc_error': ['mean', 'std']})

    sns.boxplot(x=df_loc['num_sources'], 
                y=df_loc['loc_error'], 
                hue=[intify(x) for x in df_loc['std']], 
                showfliers=False,
                showmeans=True,
                linewidth=1,
                meanprops={"marker":"s","markerfacecolor":"white", "markeredgecolor":"blue"},
                ax=axs[0])
    axs[0].legend(title='Standard Deviation [m]')
    axs[0].set_title("Localization Error")
    axs[0].set_xlabel("Number of Sources")
    axs[0].set_ylabel("RMSE [m]")
    axs[0].set_axisbelow(True)

    for ax in axs:
        ax.grid()

    dff = df.groupby(by=['std', 'num_sources']).agg({'over_predict_sources': ['mean'],
                                                     'FN': ['mean'],
                                                     'FP': ['mean'],
                                                     'percent_possible_detections': ['mean','std']})
    
    print(pd.concat([dff, loc_cols], axis=1))

    if args.save_figs:
        fig_path = os.path.join(PROJECT_ROOT_DIR, "experiments","results", "data_assoc_and_loc")
        figs[0].savefig(os.path.join(fig_path, "test.png"))