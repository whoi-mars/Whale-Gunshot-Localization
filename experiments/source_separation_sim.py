import os
import itertools
import argparse
import copy

import numpy as np
import pandas as pd
import dask
from dask.distributed import Client, LocalCluster, progress
from PIL import Image
import pyproj as proj

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from whale_gunshot_localization.utils.experimental import Localizer, MultilaterationOpt
import whale_gunshot_localization.sim_tools.sim_datagen as sim_datagen
import whale_gunshot_localization.utils.math_tools as math_tools
from whale_gunshot_localization import config, PROJECT_ROOT_DIR

# parse input arguments
parser = argparse.ArgumentParser(description="Run Monte Carlo simulations for data association/localization algorithm")
parser.add_argument('--simulate', action='store_true',
                    help="simulate rather than use previous results if available (default: false)")
parser.add_argument('--save_figs', action='store_true',
                    help="save figures (default: false)")
parser.add_argument('--suppress_warnings', action='store_true',
                    help="tell Python to suppress warnings")
parser.add_argument('--background', '-b', action='store_true',
                    help='silence the progress bar')
args = parser.parse_args()

seed = 1524

# suppress warnings
if args.suppress_warnings:
    import warnings
    warnings.filterwarnings("ignore")

def monte_carlo(measurements, s_assocs, t_assocs, s_locs, localizer_params, data_gen_params):
    """
    Run an instance of the monte carlo simulation.

    Parameters
    ----------
    measurements_list : List[array-like]
        list of lists of measurements collected on all sensors
    s_assocs : List[array-like]
        same shape as range_measurements. each sublist contains numbers which associate the
        range_measurement in the corresponding spot in the data structure with a source.
    t_assocs : List[array-like]
        same shape as range_measurements. each sublist contains numbers which associate the
        range_measurement in the corresponding spot in the data structure with a TOSSIT.
    s_locs : array-like, of shape N X 2
        source locations in the path
    localizer_params : dict
        dictionary of parameters for localizer object
        - k : int --> k value for group-k consistency check
        - multilat : MultilaterationBase --> instantiated multilateration object to use
        - consistency_thresh : float --> dict which maps std --> group-k threshhold
        - prune : bool --> ensure that all measurements in k-1 subgroups intersect to count as a consistent set
        - min_assoc_size : int --> minumum number of measurements required for a valid association
    data_gen_params : dict
        dictionary of parameters for sparse source generation
        - num_delete : int --> maximum number of farthest range measuremnets to delete for a given source (chosen from uniform distribution)
        - rng : numpy.random._generator.Generator --> RNG object
        - in_sensors : bool --> whether or not to generate soure locations only in the sensor network

    Returns
    -------
    results : dict
        results dictionary
        - over_predict_sources : bool --> whether or not more sources were predicted than exist
        - FN : bool --> there were sources to detect, but none were detected
        - FP : bool --> there were no soures to detect (or insufficient measurments), but something was detected
        - percent_possible_detections : float --> percentage of detectable sources that were detected
        - localization error : List[float] --> list of localization errors
        - best_localization_error : List[float] --> list of localization errors for detected sources assuming known data association
    """

    # initialize results dict
    results = {
        "over_predict_sources": float('nan'),
        "FN": False,
        "FP": False,
        "percent_possible_detections": float('nan'),
        "localization_error": float('nan'),
        "best_localization_error": float('nan'),
    }

    # get number of possible associations
    assoc_flat = np.concatenate(s_assocs)
    possible_associations = []
    for p in range(max(assoc_flat) + 1):
        group = np.where(assoc_flat == p)[0]
        if len(group) >= localizer_params["min_assoc_size"]:
            possible_associations.append(p)
    possible = len(possible_associations) > 0

    # instantiate localizer
    L = Localizer(**localizer_params)

    # perform data assoc/loc
    successful = L.set_measurements(measurements, **set_measurement_params)
    if not successful:
        # check for FN
        if possible:
            results["FN"] = True
        return results
    else:
        # check for FP
        if not possible:
            results["FP"] = True
            return results
        assocs_est, locs_est = L.associate_and_localize(reduce_dups=True, last_step=True)
        if len(locs_est) == 0:
            results["FN"] = True
            return results

    # if no FP or FN, set over_predict_sources to False
    results["over_predict_sources"] = False

    # if we predict too many sources return
    if locs_est.shape[0] > s_locs.shape[0]:
        results["over_predict_sources"] = True
        return results

    # calculate location errors
    num_ests = locs_est.shape[0]
    source_loc_combs = np.asarray(list(map(list, itertools.permutations(s_locs))))
    source_loc_idx_combs = np.asarray(list(map(list, itertools.permutations(np.arange(s_locs.shape[0])))))
    errors_matrix = np.sqrt(((source_loc_combs[:,:num_ests,:] - locs_est[np.newaxis,:,:]) ** 2).sum(axis=2)).sum(axis=1)
    res = np.sqrt(((source_loc_combs[np.argmin(errors_matrix),:num_ests,:] - locs_est[np.newaxis,:,:]) ** 2).sum(axis=2)).flatten()
    results["localization_error"] = res

    # calculate best possible localization errors with correct associations
    detected_sources = source_loc_idx_combs[np.argmin(errors_matrix)][:num_ests]
    measurements_flat = np.concatenate(measurements)
    TOSSIT_flat = np.concatenate(t_assocs)
    localizer = localizer_params['multilat']
    localizer.set_map({
        'TOSSIT_locations' : localizer_params['TOSSIT_locations'],
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
    best_res = np.sqrt(((best_locs - s_locs[detected_sources]) ** 2).sum(axis=1))
    results['best_localization_error'] = best_res

    # calculate percentage of possible sources localized
    percent_possible_detections = len(assocs_est) / len(possible_associations)
    results['percent_possible_detections'] = percent_possible_detections
    
    return results

def run_monte_carlo(n, std_list, num_sources_list, sparse_distance, localizer_params, set_measurement_params, data_gen_params):

    columns = ["in_sensors", "method_thresh", "num_delete", "num_sources", "std", "over_predict_sources", "FN", "FP", "percent_possible_detections", "localization_error", "best_localization_error"]
    df = pd.DataFrame(columns=columns)

    for std in std_list:
        
        # change consistency threshold based on measurement variance
        localizer_params_final = localizer_params.copy()
        localizer_params_final['consistency_thresh'] = localizer_params_final['consistency_thresh'][std]
        localizer_params_final['multilat'] = localizer_params_final['multilat'][std]
        localizer_params_final['min_assoc_size'] = localizer_params_final['min_assoc_size'][std]
        localizer_params_final['k'] = localizer_params_final['k'][std]
        
        for num_sources in num_sources_list:

            print(f'working on -- std: {std} m, num_sources: {num_sources}...', flush=True)

            measurements_list = []
            source_associations_list = []
            TOSSIT_associations_list = []
            source_locs_list = []

            rng = np.random.default_rng(seed)
            for i in range(n):
                while True:
                    measurements, source_associations, TOSSIT_associations, source_locs = sim_datagen.generate_measurements(num_sources=num_sources, var=std ** 2, rng=rng, **data_gen_params)
                    # if np.concatenate(measurements).max() <= config['scaling']['max_r'] \
                    #    and math_tools.is_sparse_locs(source_locs, thresh=sparse_distance):
                    if math_tools.is_sparse_locs(source_locs, thresh=sparse_distance):
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
                                                localizer_params_final,
                                                data_gen_params)
                results.append(res)

            # parallelize MC
            results = client.compute(results)
            if not args.background:
                progress(results)
            results = client.gather(results)

            source_over_predict_list = []
            localization_error_list = []
            FN_list = []
            FP_list = []
            best_localization_error_list = []
            percent_possible_detections_list = []
            for result in results:
                
                # save results
                source_over_predict_list.append(result['over_predict_sources']),
                FN_list.append(result["FN"])
                FP_list.append(result["FP"])
                percent_possible_detections_list.append(result["percent_possible_detections"])
                
                # save localization errors in a string format
                if isinstance(result["localization_error"], np.ndarray):
                    loc_err_str = ""
                    best_loc_err_str = ""
                    for err, best_err in zip(result["localization_error"], result["best_localization_error"]):
                        loc_err_str += f"{err};"
                        best_loc_err_str += f"{best_err};"
                    localization_error_list.append(loc_err_str)
                    best_localization_error_list.append(best_loc_err_str)
                else:
                    localization_error_list.append(result["localization_error"])
                    best_localization_error_list.append(result["best_localization_error"])

            # append results to dataframe
            df = pd.concat([df, pd.DataFrame({"in_sensors": [data_gen_params["in_sensors"] for _ in range(n)],
                                              "method_thresh": [localizer_params_final['multilat'].method_thresh for _ in range(n)],
                                              "num_delete": [data_gen_params["num_delete"] for _ in range(n)],
                                              "num_sources": [num_sources for _ in range(n)],
                                              "std": [std for _ in range(n)],
                                              "over_predict_sources": source_over_predict_list,
                                              "FN": FN_list,
                                              "FP": FP_list,
                                              "percent_possible_detections": percent_possible_detections_list,
                                              "localization_error": localization_error_list,
                                              "best_localization_error": best_localization_error_list})], ignore_index=True)
    
    return df

if __name__ == "__main__":

    # path for results CSV
    path = os.path.join(PROJECT_ROOT_DIR, "experiments", "results", "data_assoc_and_loc", "data_association_and_localization_sim_results.csv")

    ##########################################
    #          simulate/load results         #
    ##########################################

    if args.simulate:

        # dask setup
        cluster = LocalCluster(n_workers=128, processes=True)
        client = Client(cluster)

        # derive TOSSIT locations relative to the first from the lat/lons
        TOSSIT_latlons = np.asarray([config['TOSSIT']['TOSSIT_lat'], config['TOSSIT']['TOSSIT_lon']]).T
        pargs = proj.Proj(proj="aeqd", lat_0=TOSSIT_latlons[0, 0], lon_0=TOSSIT_latlons[0, 1], datum="WGS84", units="m")
        xs, ys = pargs(TOSSIT_latlons[:,1], TOSSIT_latlons[:,0])
        TOSSIT_locations = np.asarray([-ys, xs]).T

        # parameters for localizer and data_generator
        localizer_params = dict(k={0: 4, 15: 4, 30: 4, 750: 4}, 
                                multilat={i : MultilaterationOpt(method_thresh=float('inf'), seed=seed) for i in [0, 15, 30, 750]}, 
                                consistency_thresh={0: 2, 15: 30, 30: 75, 750: 500}, 
                                dup_thresh=1000, 
                                prune=False,
                                TOSSIT_locations=TOSSIT_locations,
                                min_assoc_size={0: 9, 15: 9, 30: 9, 750: 9})
        set_measurement_params = dict(adaptive=False, 
                                      adaptive_max=5000, 
                                      threshold_delta=500)
        data_gen_params = dict(num_delete=0, 
                               in_sensors=True,
                               TOSSIT_locations=TOSSIT_locations)

        # run MC
        df = run_monte_carlo(n=100,
                             std_list=[0, 15, 30, 750],
                             num_sources_list=range(1,4),
                             sparse_distance=2000,
                             localizer_params=localizer_params,
                             set_measurement_params=set_measurement_params,
                             data_gen_params=data_gen_params)
        df.to_csv(path, index=False)
    
    ##########################################
    #               make plots               #
    ##########################################

    # read results
    df = pd.read_csv(path)

    # matlab settings
    matplotlib.rcParams.update({'font.size': 16})

    # varying parameters
    std_list_all = sorted(list(set(df['std'])))
    num_sources_max = df['num_sources'].max()
    num_sources_min = df['num_sources'].min()

    # get figs and axes
    num_figs = 4
    figs = [plt.figure() for _ in range(num_figs)]
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
    location_error_dict = {(s, n): [] for s in std_list_all for n in range(num_sources_min, num_sources_max + 1)}
    best_location_error_dict = {(s, n): [] for s in std_list_all for n in range(num_sources_min, num_sources_max + 1)} 
    for _, row in df.iterrows():
        if isinstance(row['localization_error'], str):
            for err, best_err in zip(row['localization_error'].split(';')[:-1], row['best_localization_error'].split(';')[:-1]):
                num_sources_list.append(row['num_sources'])
                location_error_list.append(float(err))
                best_location_error_list.append(float(best_err))
                std_list.append(row['std'])

                location_error_dict[(row['std'], row['num_sources'])].append(float(err))
                best_location_error_dict[(row['std'], row['num_sources'])].append(float(best_err))

    
    df_loc = pd.DataFrame({'num_sources': num_sources_list,
                           'loc_error': location_error_list,
                           'best_loc_error': best_location_error_list,
                           'std': std_list,})

    df_loc_high = df_loc[df_loc["std"] >= 100]
    df_loc_low = df_loc[df_loc["std"] < 100]

    # location stats
    def perc90(iterable):
        a = np.asarray(iterable)
        a = a[~np.isnan(a)]
        return np.percentile(a, 90)
    def perc10(iterable):
        a = np.asarray(iterable)
        a = a[~np.isnan(a)]
        return np.percentile(a, 10)
    loc_cols = df_loc.groupby(by=['std', 'num_sources']).agg({'loc_error': ['mean', 'std', perc90], 'best_loc_error': ['mean', 'std', perc90]})

    sns.boxplot(x=df_loc_low['num_sources'], 
                y=df_loc_low['loc_error'], 
                hue=[intify(x) for x in df_loc_low['std']], 
                showfliers=1,
                showmeans=True,
                linewidth=1,
                meanprops={"marker":"s","markerfacecolor":"white", "markeredgecolor":"blue"},
                ax=axs[0])
    axs[0].legend(title='Standard Deviation [m]')
    axs[0].set_title("Unsupervised Localization Error")
    axs[0].set_xlabel("Number of Sources")
    axs[0].set_ylabel("Error [m]")
    axs[0].set_ylim([-5, 110])
    axs[0].set_axisbelow(True)

    sns.boxplot(x=df_loc_low['num_sources'], 
                y=df_loc_low['best_loc_error'], 
                hue=[intify(x) for x in df_loc_low['std']], 
                showfliers=1,
                showmeans=True,
                linewidth=1,
                meanprops={"marker":"s","markerfacecolor":"white", "markeredgecolor":"blue"},
                ax=axs[1])
    axs[1].legend(title='Standard Deviation [m]')
    axs[1].set_title("Ideal Localization Error")
    axs[1].set_xlabel("Number of Sources")
    axs[1].set_ylabel("Error [m]")
    axs[1].set_ylim([-5, 110])
    axs[1].set_axisbelow(True)

    sns.boxplot(x=df_loc_high['num_sources'], 
                y=df_loc_high['loc_error'], 
                hue=[intify(x) for x in df_loc_high['std']], 
                showfliers=1,
                showmeans=True,
                linewidth=1,
                meanprops={"marker":"s","markerfacecolor":"white", "markeredgecolor":"blue"},
                ax=axs[2])
    axs[2].legend(title='Standard Deviation [m]')
    axs[2].set_title("Unsupervised Localization Error")
    axs[2].set_xlabel("Number of Sources")
    axs[2].set_ylabel("Error [m]")
    axs[2].set_ylim([-5, 5000])
    axs[2].set_axisbelow(True)

    sns.boxplot(x=df_loc_high['num_sources'], 
                y=df_loc_high['best_loc_error'], 
                hue=[intify(x) for x in df_loc_high['std']], 
                showfliers=1,
                showmeans=True,
                linewidth=1,
                meanprops={"marker":"s","markerfacecolor":"white", "markeredgecolor":"blue"},
                ax=axs[3])
    axs[3].legend(title='Standard Deviation [m]')
    axs[3].set_title("Ideal Localization Error")
    axs[3].set_xlabel("Number of Sources")
    axs[3].set_ylabel("Error [m]")
    axs[3].set_ylim([-5, 5000])
    axs[3].set_axisbelow(True)

    #---------------------------------------------------#
    #------------ finalize plot/data and save ----------#
    #---------------------------------------------------#

    for ax in axs:
        ax.grid()

    dff = df.groupby(by=['std', 'num_sources']).agg({'over_predict_sources': ['mean'],
                                                     'FN': ['mean'],
                                                     'FP': ['mean'],
                                                     'percent_possible_detections': ['mean','std',perc10]})

    if args.save_figs:
        fig_path = os.path.join(PROJECT_ROOT_DIR, "experiments","results", "data_assoc_and_loc")
        figs[0].savefig(os.path.join(fig_path, "unsupervised_location_error_low.png"))
        figs[1].savefig(os.path.join(fig_path, "best_location_error_low.png"))
        figs[2].savefig(os.path.join(fig_path, "unsupervised_location_error_high.png"))
        figs[3].savefig(os.path.join(fig_path, "best_location_error_high.png"))

    # results table
    print(pd.concat([dff, loc_cols], axis=1))