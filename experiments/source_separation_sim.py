import os
import itertools
import argparse
import copy

import numpy as np
import pandas as pd
from tqdm import tqdm
import dask
from dask.distributed import Client, LocalCluster, progress
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

# def plot_localization(locs_est, buffer=0, title=None, bathym=None, dates=None, save=None): 
#     """
#     Plot estimated source locations.

#     Parameters
#     ----------
#     locs_est : List[np.array], with each subarray of shape N X 2
#         list of lists of estimated locations where the first column stores the Y
#         coordinate and the second stores the X coordinates
#     buffer : float
#         how much to plot outside of the limits established in the config file
#     title : str
#         plot title
#     bathym : PIL.Image
#         geotiff of the bathymetry
#     dates : List[datetime.datetime]
#         list of dates associated with each sublist of location estiamtes
#         in locs_est
#     save : str
#         path at which to save the plot if desired
#     """

#     # format inputs
#     if not isinstance(locs_est, list):
#         locs_est = [locs_est]
#     if not isinstance(dates, list):
#         dates = [dates]
#     # assert len(locs_est) == len(dates), "locs_est and dates lists must have a one-to-one correspondence"

#     # random list of color for plotting
#     rng = np.random.default_rng(1111)
#     colors = [rng.uniform(0, 1, size=3) for _ in range(len(locs_est))]

#     # load constants
#     TOSSIT_locations = np.asarray([config['TOSSIT']['TOSSIT_y'], config['TOSSIT']['TOSSIT_x']]).T
#     min_x = config['scaling']['min_x']
#     max_x = config['scaling']['max_x']
#     min_y = config['scaling']['min_y']
#     max_y = config['scaling']['max_y']
#     map_origin = config['TOSSIT']['map_origin']
#     dy = config['TOSSIT']['dy']
#     dx = config['TOSSIT']['dx']

#     # crop bathymetry appropriately 
#     bathym = bathym.crop((round(map_origin[1] + (min_x/dx) - (buffer/dx)), 
#                           round(map_origin[0] + (min_y/dy) - (buffer/dy)), 
#                           round(map_origin[1] + (max_x/dx) + (buffer/dx)), 
#                           round(map_origin[0] + (max_y/dy) + (buffer/dy))))

#     fig, ax = plt.subplots(1, 1)
#     ax.plot(TOSSIT_locations[:,1], TOSSIT_locations[:,0], '^', color="#21EE71", markersize=10, label='sensor', markeredgewidth=2)
#     for i, l in enumerate(locs_est):
#         ax.plot(l[:,1], l[:,0], 'x', color=colors[i], label=f'estimate ({dates[i]})' if dates[0] else 'estimate', markersize=6, markeredgewidth=2)
#     if title is not None:
#         ax.set_title(title)
#     ax.set_xlabel("X [m]")
#     ax.set_ylabel("Y [m]")
#     ax.set_xlim([min_x-buffer, max_x+buffer])
#     ax.set_ylim([min_y-buffer, max_y+buffer])
#     ax.invert_yaxis()
#     implot = ax.imshow(bathym, extent=(min_x - buffer, max_x + buffer, max_y + buffer, min_y - buffer))
#     # ax.legend()
    
#     # save or show
#     if save is not None:
#         fig.savefig(save)
#     else:
#         fig.show()
    
#     # close figure
#     plt.close(fig)

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
    if num_delete:
        num_delete = rng.choice([0, num_delete])
        for s in range(num_sources): 
            distances = np.sqrt(((TOSSIT_locations - source_locs[s,:]) ** 2).sum(axis=1))
            to_delete = np.argsort(distances)[::-1][:TOSSIT_locations.shape[0] - num_delete]
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

    # initialize results dict
    results = {
        "over_predict_sources": float('nan'),
        "FN": False,
        "FP": False,
        "percent_possible_detections": float('nan'),
        "localization_error": float('nan'),
        "best_localization_error": float('nan'),
    }

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
            results["FN"] = True
        #return float('nan'), FN, FP, float('nan'), float('nan'), float('nan')
        return results
    else:
        # check for FP
        if not possible:
            FP = True
            results["FP"] = True
            #return float('nan'), FN, FP, float('nan'), float('nan'), float('nan')
            return results
        assocs_est, locs_est = L.associate_and_localize(method='partition', reduce_dups=True, last_step=True)

    # if no FP or FN, set over_predict_sources to False
    results["over_predict_sources"] = False

    # if we predict too many sources return
    if locs_est.shape[0] > s_locs.shape[0]:
        over_predict_sources = True
        results["over_predict_sources"]
        #return over_predict_sources, FN, FP, float('nan'), float('nan'), float('nan')
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
    best_res = np.sqrt(((best_locs - s_locs[detected_sources]) ** 2).sum(axis=1))
    results['best_localization_error'] = best_res

    # calculate percentage of possible sources localized
    percent_possible_detections = len(assocs_est) / len(possible_associations)
    results['percent_possible_detections'] = percent_possible_detections
    
    #return over_predict_sources, FN, FP, percent_possible_detections, res, best_res
    return results

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
        
        # change consistency threshold based on measurement variance
        localizer_params_final = localizer_params.copy()
        localizer_params_final['consistency_thresh'] = localizer_params_final['consistency_thresh'][std]
        
        for num_sources in num_sources_list:

            if args.background:
                print(f'working on -- std: {std} m, num_sources: {num_sources}...', flush=True)

            measurements_list = []
            source_associations_list = []
            TOSSIT_associations_list = []
            source_locs_list = []
            for i in range(n):
                while True:
                    measurements, source_associations, TOSSIT_associations, source_locs = generate_measurements(num_sources=num_sources, var=std ** 2, **data_gen_params)
                    if np.concatenate(measurements).max() <= config['scaling']['max_r'] \
                       and math_tools.is_sparse_locs(source_locs, thresh=sparse_distance) \
                       and is_in_bay(source_locs, bathym, map_origin, dx, dy):
                        break
                measurements_list.append(measurements)
                source_associations_list.append(source_associations)
                TOSSIT_associations_list.append(TOSSIT_associations)
                source_locs_list.append(source_locs)

            # Image.MAX_IMAGE_PIXELS = 729744000
            # bathym = Image.open(os.path.join(config['dataset']['data_directory'], "mikesbathym.tif"))
            # plot_localization(source_locs_list, buffer=0, title=None, bathym=bathym, dates=None, save='image.png') 
            # return

            # delayed for loop using dask
            results = []
            for i in range(n):
                res = dask.delayed(monte_carlo)(measurements_list[i],
                                                source_associations_list[i],
                                                TOSSIT_associations_list[i],
                                                source_locs_list[i],
                                                localizer_params_final)
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
                # source_over_predict_list.append(result[0]),
                # FN_list.append(result[1])
                # FP_list.append(result[2])
                # percent_possible_detections_list.append(result[3])
                source_over_predict_list.append(result['over_predict_sources']),
                FN_list.append(result["FN"])
                FP_list.append(result["FP"])
                percent_possible_detections_list.append(result["percent_possible_detections"])
                
                # save localization errors in a string format
                # if isinstance(result[4], np.ndarray):
                if isinstance(result["localization_error"], np.ndarray):
                    loc_err_str = ""
                    best_loc_err_str = ""
                    # for err, best_err in zip(result[4], result[5]):
                    for err, best_err in zip(result["localization_error"], result["best_localization_error"]):
                        loc_err_str += f"{err};"
                        best_loc_err_str += f"{best_err};"
                    localization_error_list.append(loc_err_str)
                    best_localization_error_list.append(best_loc_err_str)
                else:
                    # localization_error_list.append(result[4])
                    # best_localization_error_list.append(result[5])
                    localization_error_list.append(result["localization_error"])
                    best_localization_error_list.append(result["best_localization_error"])

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

    if args.simulate:

        # dask setup
        cluster = LocalCluster(n_workers=100, processes=True)
        client = Client(cluster)

        # set random seed
        # make two of these for monte carlo and localizer and make an internal one for monte
        # carlo which does either the source locations or variance
        rng1 = np.random.default_rng(1524)
        rng2 = np.random.default_rng(1524)

        # parameters for localizer and data_generator
        localizer_params = dict(k=4, multilat=MultilaterationOpt(method_thresh=float('inf'), rng=rng1), consistency_thresh={0: 2, 250: 500, 500: 1500, 750: 2000, 1000: 2500}, dup_thresh=3000, prune=False)
        set_measurement_params = dict(adaptive=False, adaptive_max=5000, threshold_delta=500)
        data_gen_params = dict(num_delete=6, rng=rng2, in_sensors=True)

        # run MC
        df = run_monte_carlo(n=300,
                             std_list=[0, 250, 500, 750, 1000],
                             num_sources_list=range(1,6),
                             sparse_distance=10,
                             localizer_params=localizer_params,
                             set_measurement_params=set_measurement_params,
                             data_gen_params=data_gen_params)
        df.to_csv(path, index=False)

    # read results
    df = pd.read_csv(path)
    
    ##########################################
    #               make plots               #
    ##########################################

    # matlab settings
    matplotlib.rcParams.update({'font.size': 16})

    # varying parameters
    # var_list = sorted(list(set(df['var'])))
    std_list_all = sorted(list(set(df['std'])))
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
    axs[0].set_title("Unsupervised Localization Error")
    #axs[0].set_title("Unsupervised Localization Error (N/2 Measuements)")
    axs[0].set_xlabel("Number of Sources")
    axs[0].set_ylabel("Error [m]")
    #axs[0].set_ylim([0, 8000])
    axs[0].set_axisbelow(True)

    sns.boxplot(x=df_loc['num_sources'], 
                y=df_loc['best_loc_error'], 
                hue=[intify(x) for x in df_loc['std']], 
                showfliers=False,
                showmeans=True,
                linewidth=1,
                meanprops={"marker":"s","markerfacecolor":"white", "markeredgecolor":"blue"},
                ax=axs[1])
    axs[1].legend(title='Standard Deviation [m]')
    axs[1].set_title("Ideal Localization Error")
    #axs[1].set_title("Ideal Localization Error (N/2 Measurements)")
    axs[1].set_xlabel("Number of Sources")
    axs[1].set_ylabel("Error [m]")
    #axs[1].set_ylim([0, 8000])
    axs[1].set_axisbelow(True)

    #-----------------------------------#
    #--------- outlier analysis --------#
    #-----------------------------------#

    vals = np.zeros((len(std_list_all), num_sources_max - num_sources_min + 1))
    vals_best = np.zeros(vals.shape)
    anno = np.zeros(vals.shape)
    annob = np.zeros(vals.shape)
    for i, std in enumerate(std_list_all):
        for j, n in enumerate(range(num_sources_min, num_sources_max + 1)):
            q1 = np.percentile(location_error_dict[(std, n)], 25)
            q3 = np.percentile(location_error_dict[(std, n)], 75)
            q1b = np.percentile(best_location_error_dict[(std, n)], 25)
            q3b = np.percentile(best_location_error_dict[(std, n)], 75)
            IQR = q3 - q1
            IQRb = q3b - q1b

            location_error_dict[(std, n)] = np.asarray(location_error_dict[(std, n)])
            best_location_error_dict[(std, n)] = np.asarray(best_location_error_dict[(std, n)])
            val = np.percentile(location_error_dict[(std, n)][location_error_dict[(std, n)] > 1.5*IQR], 90)
            val_best = np.percentile(best_location_error_dict[(std, n)][best_location_error_dict[(std, n)] > 1.5*IQRb], 90)
            vals[i,j] = val
            vals_best[i,j] = val_best

            anno[i,j] = len(location_error_dict[(std, n)][location_error_dict[(std, n)] > 1.5*IQR]) / len(location_error_dict[(std, n)])
            annob[i,j] = len(best_location_error_dict[(std, n)][best_location_error_dict[(std, n)] > 1.5*IQRb]) / len(best_location_error_dict[(std, n)])

    sns.heatmap(vals, 
                annot=anno,
                xticklabels=range(num_sources_min, num_sources_max + 1),
                yticklabels=std_list_all,
                cbar_kws={'label': '90th Percentile Outliers'}, 
                ax=axs[2])
    axs[2].set_xlabel("Number of Sources")
    axs[2].set_ylabel("Measurement Standard Deviation [m]")
    axs[2].set_title("Outlier Analysis (Unsupervised)")
    #axs[2].set_title("Outlier Analysis (Unsupervised, N/2 Measurements)")
    axs[2].invert_yaxis()


    sns.heatmap(vals_best, 
                annot=annob,
                xticklabels=range(num_sources_min, num_sources_max + 1),
                yticklabels=std_list_all,
                cbar_kws={'label': '90th Perentile Outliers'}, 
                ax=axs[3])
    axs[3].set_xlabel("Number of Sources")
    axs[3].set_ylabel("Measurement Standard Deviation [m]")
    axs[3].set_title("Outlier Analysis (Ideal)")
    #axs[3].set_title("Outlier Analysis (Ideal, N/2 Measurements)")
    axs[3].invert_yaxis()

    #-----------------------------------#
    #-------- error districutions ------#
    #-----------------------------------#

    figg, axx = plt.subplots(vals.shape[0], vals.shape[1], figsize=(26,26))
    if not isinstance(axx, np.ndarray):
        axx = np.asarray([[axx]])
    elif len(axx.shape) < 2:
        axx = np.asarray([axx])

    for i, std in enumerate(std_list_all):
        for j, n in enumerate(range(num_sources_min, num_sources_max + 1)):
            #_,bins,_ = axx[i,j].hist(location_error_dict[(std,n)] / 1000, alpha=0.5, bins=150, label='unsupervised (N/2 measurements)')
            #axx[i,j].hist(best_location_error_dict[(std,n)] / 1000, alpha=0.5, bins=bins, label='ideal (N/2 measurements)')
            _,bins,_ = axx[i,j].hist(location_error_dict[(std,n)] / 1000, alpha=0.5, bins=150, label='unsupervised')
            axx[i,j].hist(best_location_error_dict[(std,n)] / 1000, alpha=0.5, bins=bins, label='ideal')
            axx[i,j].set_title(f"n={n}, $\sigma$={std} m", fontsize=22)
            axx[i,j].tick_params(axis='x', labelsize=16)
            axx[i,j].tick_params(axis='y', labelsize=16)
            axx[i,j].yaxis.get_offset_text().set_fontsize(14)
            axx[i,j].xaxis.get_offset_text().set_fontsize(14)
    handles, labels = axx[0,0].get_legend_handles_labels()
    figg.legend(handles, labels, loc='upper center', prop={'size': 28})
    figg.subplots_adjust(wspace=0.35, hspace=0.35)
    figg.text(0.5, 0.04, "Localization Error [km]", ha='center', va='center', fontsize=28)
    figg.text(0.05, 0.5, "Example Count", ha='center', va='center', rotation=90, fontsize=28)

    for i, ax in enumerate(axs):
        if i in [2, 3]:
            continue
        ax.grid()

    dff = df.groupby(by=['std', 'num_sources']).agg({'over_predict_sources': ['mean'],
                                                     'FN': ['mean'],
                                                     'FP': ['mean'],
                                                     'percent_possible_detections': ['mean','std']})

    if args.save_figs:
        fig_path = os.path.join(PROJECT_ROOT_DIR, "experiments","results", "data_assoc_and_loc")
        figs[0].savefig(os.path.join(fig_path, "unsupervised_location_error.png"))
        figs[1].savefig(os.path.join(fig_path, "best_location_error.png"))
        figs[2].savefig(os.path.join(fig_path, "unsupervised_outliers.png"))
        figs[3].savefig(os.path.join(fig_path, "best_outliers.png"))
        figg.savefig(os.path.join(fig_path, "hists.png"))

    # results table
    print(pd.concat([dff, loc_cols], axis=1))
    
    # # get number of outliers
    # location_error_list = np.asarray(location_error_list)
    # best_location_error_list = np.asarray(best_location_error_list)

    # q3 = np.percentile(location_error_list, 75)
    # q1 = np.percentile(location_error_list, 25)
    # q3b = np.percentile(best_location_error_list, 75)
    # q1b = np.percentile(best_location_error_list, 25)
    
    # IQR = q3 - q1
    # IQRb = q3b - q1b

    # print()
    # print("-----------------------------------------------")
    # print("% Outliers: ", np.around(100 * len(location_error_list[location_error_list >= 1.5*IQR])/len(location_error_list), 2))
    # print("% Outliers Best: ", np.around(100 * len(best_location_error_list[best_location_error_list >= 1.5*IQRb])/len(best_location_error_list), 2))
    # print("-----------------------------------------------")
    # print("90th Percentile Outliers: ", np.around(np.percentile(location_error_list[location_error_list >= 1.5*IQR],90)), " m")
    # print("90th Percentile Outliers Best: ", np.around(np.percentile(best_location_error_list[best_location_error_list >= 1.5*IQRb],90)), " m")
    # print("-----------------------------------------------")
