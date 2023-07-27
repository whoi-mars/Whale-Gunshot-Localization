import os
import argparse
from multiprocessing import Pool
import itertools

from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

from whale_gunshot_localization import config, PROJECT_ROOT_DIR
from whale_gunshot_localization.utils.experimental import Localizer, MultilaterationOpt, MultilaterationGrid

# parse input arguments
parser = argparse.ArgumentParser(description="Run Monte Carlo simulations for data association/localization algorithm")
parser.add_argument('--simulate', action='store_true',
                    help="simulate rather than use previous results if available (default: false)")
parser.add_argument('--save_figs', action='store_true',
                    help="save figures (default: false)")
parser.add_argument('--background', '-b', action='store_true',
                    help='silence the progress bar')
args = parser.parse_args()

def generate_measurements(num_sources, rng, var=10, num_delete=0):
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
        r = np.linalg.norm(source_locs - TOSSIT_locations[t,:], axis=1) + \
            rng.normal(loc=0, scale=np.sqrt(var), size=num_sources)
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

##################################################################################################################

def monte_carlo(measurements, source_associations, TOSSIT_associations, source_locs, localizer_params):
    """
    Function to run the localizer on a set of sparse generated sources.

    Parameters
    ----------
    measurements : List[array-like]
        each sublist contains range measurements and is associated with a particular TOSSIT
    source_associations : List[array-like]
        same shape as range_measurements. each sublist contains numbers which associate the
        range_measurement in the corresponding spot in the data structure with a source.
    TOSSIT_associations : List[array-like]
        same shape as range_measurements. each sublist contains numbers which associate the
        range_measurement in the corresponding spot in the data structure with a TOSSIT.    
    source_locs : array-like[array-like]
        matrix of generated source locations
    localizer_params : dict
        dictionary containing keys, 'k', 'multilat', 'consistency_thresh', and 'prune' to
        parameterize the Localizer object

    Returns
    -------
    missed association : bool
        there were associations, but none were identified (True)
    FP : bool
        associations were identified despite there being none
    total_fail_count : bool
        more associations were identified than were possible
    loc_err_list : List[float]
        list of localization errors, or [float('nan')] if any of the first three bools are True
    : float
        missasociated fraction of measurements, or [float('nan')] if any of the first three bools are True
    : float
        missed fraction of measurements, or [float('nan')] if any of the first three bools are True
    : float
        number of missed measurements, or [float('nan')] if any of the first three bools are True
    : float
        number of missasociated measurements, or [float('nan')] if any of the first three bools are True
    : float
        fraction of possible associations identified
    """

    l = Localizer(**localizer_params)
    
    missed_association_count = False
    FP = False
    total_fail_count = False

    assoc_flat = np.concatenate(source_associations)

    # get total number of measurements
    num_measurements = 0
    for m in measurements:
        num_measurements += len(m)

    # detect if an association is possible and
    # collect node numbers which correspond to
    # possible associations
    possible_associations = []
    gt_nodes = []
    for p in range(max(assoc_flat) + 1):
        group = np.where(assoc_flat == p)[0]
        if len(group) >= localizer_params["k"]:
            possible_associations.append(p)
            gt_nodes.extend(group)
        else:
            num_measurements -= len(group)
    possible = (len(possible_associations) > 0)

    # build measurement graph and detect FNs
    successful = l.set_measurements(measurements, **set_measurement_params)
    if not successful:
        if possible:
            missed_association_count = True
        return missed_association_count, FP, total_fail_count, float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan')

    # catch FPs
    if not possible:
        FP = True
        return missed_association_count, FP, total_fail_count, float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan')

    # associate and localize
    assoc, locs = l.associate_and_localize(method='partition', last_step=True)

    # determine if number of associations is wrong
    if (len(assoc) > len(possible_associations)) and len(possible_associations) > 0:
        total_fail_count = True
        return missed_association_count, FP, total_fail_count, float('nan'), float('nan'), float('nan'), float('nan'), float('nan'), float('nan')

    # calculte measurements missed in associations
    missed_measurements = len(set(gt_nodes) - set.union(*assoc))

    num_wrong_associations = 0
    for p in possible_associations:
        
        # ideal association group
        group = set(np.where(assoc_flat == p)[0])
        
        # find best batch by finding smallest set difference with candidates
        set_differences = [a - group for a in assoc]
        best_match = min(set_differences, key=len)
        num_wrong_associations += len(best_match)

    # greedily take smallest error for each
    loc_err_list = []
    loc_err_list_str = ""
    for loc in locs:
        errs = np.sqrt(((source_locs - loc) ** 2).sum(axis=1))
        idx_delete = np.argmin(errs)
        loc_err_list.append(errs[idx_delete])
        loc_err_list_str += str(errs[idx_delete]) + ';'
        source_locs = np.delete(source_locs, idx_delete, axis=0)

    return missed_association_count, FP, total_fail_count, loc_err_list_str[:-1], num_wrong_associations / num_measurements, missed_measurements / num_measurements, missed_measurements, num_wrong_associations, len(assoc) / len(possible_associations)

def monte_carlo_sim(n, var_list, num_sources_list, localizer_params, set_measurement_params, data_gen_params):
    """
    Run sparse source data association/localization Monte Carlo simulations.

    Parameters
    ----------

    """

    columns = ["num_sources", "var", "delete", "k", "consistency_thresh", "method_thresh", "prune", "average_estimated_source_fraction", "average_misassociated", "average_misassociated_fraction", "average_location_error", "average_missed", "average_missed_fraction", "total_fail_fraction", "no_association_fraction", "false_association_fraction"]
    df = pd.DataFrame(columns=columns)

    for var in var_list:
        for num_sources in num_sources_list:
            
            if args.background:
                print(f'working on -- var: {var} m, num_sources: {num_sources}...', end='', flush=True)

            # no successful association were made, but they were possible
            missed_association_list = []

            # number of identified associations does not match number of sources
            total_fail_list = []
            
            # count associations when it shouldn't be possible
            FP_list = []

            # measurements not put in any association
            total_missed_frac_list = []
            total_missed_list = []

            # fraction of measurements wrongly associated for each of the "n" runs
            wrong_assoc_list = []
            total_wrong_assoc_list = []

            # fraction of viable sources estimated
            estimated_frac_list = []

            # squared error list
            loc_err_list = []

            # randomly generate measurements
            measurements_list = []
            source_associations_list = []
            TOSSIT_associations_list = []
            source_locs_list = []
            for _ in range(n):
                measurements, source_associations, TOSSIT_associations, source_locs = generate_measurements(num_sources=num_sources, var=var, **data_gen_params)
                measurements_list.append(measurements)
                source_associations_list.append(source_associations)
                TOSSIT_associations_list.append(TOSSIT_associations)
                source_locs_list.append(source_locs)

            with Pool(processes=100) as pool:
                results = pool.starmap(monte_carlo, tqdm(zip(measurements_list, source_associations_list, TOSSIT_associations_list, source_locs_list, itertools.repeat(localizer_params)), total=n, postfix={'var' : var, 'num_sources' : num_sources}))
            
            # accumulate results
            for result in results:
                missed_association_list.append(result[0])
                FP_list.append(result[1])
                total_fail_list.append(result[2])
                loc_err_list.append(result[3])
                wrong_assoc_list.append(result[4])
                total_missed_frac_list.append(result[5])
                total_missed_list.append(result[6])
                total_wrong_assoc_list.append(result[7])
                estimated_frac_list.append(result[8])

            df = pd.concat([df, pd.DataFrame({"num_sources": [num_sources for _ in range(n)],
                                              "var": [var for _ in range(n)],
                                              "delete": [data_gen_params["num_delete"] for _ in range(n)],
                                              "k": [localizer_params["k"] for _ in range(n)],
                                              "consistency_thresh": [localizer_params["consistency_thresh"] for _ in range(n)],
                                              "method_thresh": [localizer_params["multilat"].method_thresh for _ in range(n)],
                                              "prune": [localizer_params["prune"] for _ in range(n)],
                                              "adaptive": [set_measurement_params["adaptive"] for _ in range(n)],
                                              "adaptive_max": [set_measurement_params["adaptive_max"] for _ in range(n)],
                                              "threshold_delta": [set_measurement_params["threshold_delta"] for _ in range(n)],
                                              "average_estimated_source_fraction": estimated_frac_list,
                                              "average_misassociated": total_wrong_assoc_list,
                                              "average_misassociated_fraction": wrong_assoc_list,
                                              "average_location_error": loc_err_list,
                                              "average_missed": total_missed_list,
                                              "average_missed_fraction": total_missed_frac_list,
                                              "total_fail_fraction": total_fail_list,
                                              "no_association_fraction": missed_association_list,
                                              "false_association_fraction": FP_list})], ignore_index=True)

        if args.background:
            print("done!")

    if args.background:
        print("finished!")

    return df

if __name__ == "__main__":

    # path for results CSV
    path = os.path.join(PROJECT_ROOT_DIR, "experiments", "results", "data_assoc_and_loc", "data_association_and_localization_sim_results.csv")

    ##########################################
    #          simulate/load results         #
    ##########################################

    if args.simulate:
        # set random seed
        # make two of these for monte carlo and localizer and make an internal one for monte
        # carlo which does either the source locations or variance
        rng1 = np.random.default_rng(1524)
        rng2 = np.random.default_rng(1524)

        # parameters for localizer and data_generator
        localizer_params = dict(k=4, multilat=MultilaterationOpt(method_thresh=0.95, rng=rng1), consistency_thresh=1000, prune=False)
        set_measurement_params = dict(adaptive=False, adaptive_max=5000, threshold_delta=500)
        data_gen_params = dict(num_delete=0, rng=rng2)

        # run data association/localization experiment
        df = monte_carlo_sim(n=150,
                             var_list=[1e1, 1e2, 1e3, 1e4, 1e5, 1e6], 
                             num_sources_list=range(1, 6), 
                             localizer_params=localizer_params, 
                             set_measurement_params=set_measurement_params,
                             data_gen_params=data_gen_params,)
        
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
    
    ##########################################
    #               make plots               #
    ##########################################

    # matlab settings
    matplotlib.rcParams.update({'font.size': 16})

    # varying parameters
    var_list = sorted(list(set(df['var'])))
    num_sources_max = df['num_sources'].max()
    num_sources_min = df['num_sources'].min()

    # get figs and axes
    figs = [plt.figure() for _ in range(7)]
    axs = [fig.gca() for fig in figs]

    # make variance integer if possible
    def intify(x):
        if isinstance(x, float) and x.is_integer():
            return int(x)
        else:
            return x
        
            
    df['var'] = df['var'].apply(intify)

    #-----------------------------------#
    #------- plot location error -------#
    #-----------------------------------#

    # extract individual location errors
    num_sources_list = []
    location_error_list = []
    variance_list = []
    for _, row in df.iterrows():
        if isinstance(row['average_location_error'], str):
            location_error_list.extend([float(m) for m in row['average_location_error'].split(';')])
            for _ in range(len(row['average_location_error'].split(';'))):
                num_sources_list.append(row['num_sources'])
                variance_list.append(row['var'])
        else:
            location_error_list.append(float('nan'))
            num_sources_list.append(row['num_sources'])
            variance_list.append(row['var'])
    df_loc = pd.DataFrame({'num_sources' : num_sources_list,
                           'loc_error' : location_error_list,
                           'var' : variance_list,}).dropna(axis=0)
    
    # make plot
    sns.boxplot(x=df_loc['num_sources'], 
                y=df_loc['loc_error'], 
                hue=df_loc['var'], 
                showfliers=False,
                showmeans=True,
                linewidth=1,
                meanprops={"marker":"s","markerfacecolor":"white", "markeredgecolor":"blue"},
                ax=axs[0])
    axs[0].legend(title='Variance [m$^{2}$]')
    axs[0].set_title("Localization Error")
    axs[0].set_xlabel("Number of Sources")
    axs[0].set_ylabel("Average Absolute Error [m]")
    axs[0].set_axisbelow(True)
    axs[0].set_yscale("log")

    #------------------------------------#
    #------- plot misassociations -------#
    #------------------------------------#

    sns.boxplot(x=df['num_sources'],
                y=df['average_misassociated_fraction'],
                hue=df['var'],
                showfliers=False,
                showmeans=True,
                linewidth=1,
                meanprops={"marker":"s","markerfacecolor":"white", "markeredgecolor":"blue"},
                ax=axs[1])
    axs[1].legend(title='Variance [m$^{2}$]')
    axs[1].set_title("Data Association Error")
    axs[1].set_xlabel("Number of Sources")
    axs[1].set_ylabel("Average Fraction of Misassociated Measurements")
    axs[1].set_axisbelow(True)

    #------------------------------------#
    #----- plot missed measurements -----#
    #------------------------------------#

    sns.boxplot(x=df['num_sources'],
                y=df['average_missed_fraction'],
                hue=df['var'],
                showfliers=False,
                showmeans=True,
                linewidth=1,
                meanprops={"marker":"s","markerfacecolor":"white", "markeredgecolor":"blue"},
                ax=axs[2])
    axs[2].legend(title='Variance [m$^{2}$]')
    axs[2].set_title("Missed Measurements")
    axs[2].set_xlabel("Number of Sources")
    axs[2].set_ylabel("Average Fraction of Missed Measurements")

    sns.lineplot(x=df['num_sources'], 
                 y=np.mean(df['total_fail_fraction']),
                 hue=df['var'],
                 palette=sns.color_palette("tab10"),
                 linewidth=5,
                 marker='o',
                 ax=axs[3])
    axs[3].legend(title='Variance [m$^{2}$]')
    axs[3].set_title("Total Fails")
    axs[3].set_xlabel("Number of Sources")
    axs[3].set_ylabel("Total Fail Run Fraction")
    axs[3].xaxis.get_major_locator().set_params(integer=True)

    sns.lineplot(x=df['num_sources'],
                 y=np.mean(df['no_association_fraction']),
                 hue=df['var'],
                 palette=sns.color_palette("tab10"),
                 linewidth=5,
                 marker='o',
                 ax=axs[4])
    axs[4].legend(title='Variance [m$^{2}$]')
    axs[4].set_title("Missed Associations (FN)")
    axs[4].set_xlabel("Number of Sources")
    axs[4].set_ylabel("Missed Association Run Fraction")
    axs[4].xaxis.get_major_locator().set_params(integer=True)

    # axs[5].plot(sources, df_plot['false_association_fraction'], '-o', label=f"$\sigma^{2}$ = {int(var / 1e6) if (var / 1e6).is_integer() else var / 1e6} km")
    sns.lineplot(x=df['num_sources'],
                 y=np.mean(df['false_association_fraction']),
                 hue=df['var'],
                 palette=sns.color_palette(("tab10")),
                 linewidth=5,
                 marker='o',
                 ax=axs[5])
    axs[5].legend(title='Variance [m$^{2}$]')
    axs[5].set_title("False Associations (FP)")
    axs[5].set_xlabel("Number of Sources")
    axs[5].set_ylabel("False Association Run Fraction")
    axs[5].xaxis.get_major_locator().set_params(integer=True)

    sns.boxplot(x=df['num_sources'],
                y=df['average_estimated_source_fraction'],
                hue=df['var'],
                showfliers=False,
                showmeans=True,
                linewidth=1,
                meanprops={"marker":"s","markerfacecolor":"white", "markeredgecolor":"blue"},
                ax=axs[6])
    axs[6].legend(title='Variance [m$^{2}$]')
    axs[6].set_title("Percent of Sources Detected That Are Possible")
    axs[6].set_xlabel("Number of Sources")
    axs[6].set_ylabel("Estimated Source Run Fraction")

    for i, ax in enumerate(axs):
        ax.grid()

    plt.show()

    if args.save_figs:
        fig_path = os.path.join(PROJECT_ROOT_DIR, "experiments","results", "data_assoc_and_loc")
        figs[0].savefig(os.path.join(fig_path, 'loc_error.png'))
        figs[1].savefig(os.path.join(fig_path, 'data_assoc_error.png'))
        figs[2].savefig(os.path.join(fig_path, 'missed_measurements.png'))
        figs[3].savefig(os.path.join(fig_path, 'total_fails.png'))
        figs[4].savefig(os.path.join(fig_path, 'missed_associations.png'))
        figs[5].savefig(os.path.join(fig_path, 'false_associations.png'))
        figs[6].savefig(os.path.join(fig_path, 'percent_det.png'))



        
    

