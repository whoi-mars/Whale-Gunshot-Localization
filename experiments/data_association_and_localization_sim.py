import os
import argparse

from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

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

def monte_carlo_sim(n, var_list, num_sources_list, localizer_params, set_measurement_params, data_gen_params):

    columns = ["num_sources", "var", "delete", "k", "consistency_thresh", "method_thresh", "prune", "average_estimated_source_fraction", "average_misassociated", "average_misassociated_fraction", "average_location_error", "average_missed", "average_missed_fraction", "total_fail_fraction", "no_association_fraction", "false_association_fraction"]
    df = pd.DataFrame(columns=columns)

    l = Localizer(**localizer_params)

    with tqdm(total=len(var_list) * len(num_sources_list) * n, disable=args.background) as pbar:
        for var in var_list:
            if args.background:
                print(f'working on -- var: {var} km...', end='', flush=True)
            for num_sources in num_sources_list:
                
                # no successful association were made, but they were possible
                missed_association_count = 0

                # number of identified associations does not match number of sources
                total_fail_count = 0
                
                # count associations when it shouldn't be possible
                FP = 0

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

                for _ in range(n):

                    # randomly generate measurements
                    measurements, source_associations, TOSSIT_associations, source_locs = generate_measurements(num_sources=num_sources, var=var, **data_gen_params)
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
                            missed_association_count += 1
                        pbar.update(1)
                        continue
                    
                    # catch FPs
                    if not possible:
                        FP += 1
                        pbar.update(1)
                        continue
                    
                    # associate and localize
                    assoc, locs = l.associate_and_localize(method='partition')

                    # determine if number of associations is wrong
                    if (len(assoc) > len(possible_associations)) and len(possible_associations) > 0:
                        total_fail_count += 1
                        pbar.update(1)
                        continue

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

                    # keep track of lists to calculate averages
                    wrong_assoc_list.append(num_wrong_associations / num_measurements)
                    total_missed_frac_list.append(missed_measurements / num_measurements)
                    total_missed_list.append(missed_measurements)
                    total_wrong_assoc_list.append(num_wrong_associations)
                    estimated_frac_list.append(len(assoc) / len(possible_associations))

                    # greedily take smallest error for each
                    for loc in locs:
                        errs = np.sqrt(((source_locs - loc) ** 2).sum(axis=1))
                        idx_delete = np.argmin(errs)
                        loc_err_list.append(errs[idx_delete])
                        source_locs = np.delete(source_locs, idx_delete, axis=0)

                    # reset localizer
                    l.reset()

                    # update progress bar
                    pbar.update(1)

                df = pd.concat([df, pd.DataFrame({"num_sources": [num_sources],
                                                "var": [var],
                                                "delete": [data_gen_params["num_delete"]],
                                                "k": [localizer_params["k"]],
                                                "consistency_thresh": [localizer_params["consistency_thresh"]],
                                                "method_thresh": [localizer_params["method_thresh"]],
                                                "prune": [localizer_params["prune"]],
                                                "grid": [localizer_params["grid"]],
                                                "adaptive": [set_measurement_params["adaptive"]],
                                                "adaptive_max": [set_measurement_params["adaptive_max"]],
                                                "threshold_delta": [set_measurement_params["threshold_delta"]],
                                                "average_estimated_source_fraction": [np.around(np.mean(estimated_frac_list), 4)],
                                                "average_misassociated":[np.around(np.mean(total_wrong_assoc_list), 4)],
                                                "average_misassociated_fraction": [np.around(np.mean(wrong_assoc_list), 4)],
                                                "average_location_error": [np.around(np.mean(loc_err_list), 4)],
                                                "average_missed": [np.around(np.mean(total_missed_list), 4)],
                                                "average_missed_fraction": [np.around(np.mean(total_missed_frac_list), 4)],
                                                "total_fail_fraction": [np.around(np.mean(total_fail_count / n), 4)],
                                                "no_association_fraction": [np.around(np.mean(missed_association_count / n), 4)],
                                                "false_association_fraction": [np.around(FP / n, 4)]})], ignore_index=True)

            if args.background:
                print("done!")

        if args.background:
            print("finished!")

    return df

if __name__ == "__main__":

    # path for results CSV
    path = os.path.join(PROJECT_ROOT_DIR, "experiments", "results", "data_association_and_localization_sim_results.csv")

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
                             var_list=[0, 250, 500, 750, 1000, 1250], 
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
    
    # get figs and axes
    figs = [plt.figure() for _ in range(7)]
    axs = [fig.gca() for fig in figs]

    for var in var_list:

        # get appropriate df rows
        df_plot = df[df['var'] == var].sort_values(by=['num_sources'])
        sources = df_plot['num_sources'].tolist()

        # localization error
        axs[0].xaxis.get_major_locator().set_params(integer=True)
        axs[0].plot(sources, df_plot['average_location_error'], '-o', label=f"$\sigma^{2}$ = {var / 1000} km")
        axs[0].legend()
        axs[0].set_title("Localization Error")
        axs[0].set_xlabel("Number of Sources")
        axs[0].set_ylabel("Average Absolute Error [m]")

        # misassociated measurements
        axs[1].xaxis.get_major_locator().set_params(integer=True)
        axs[1].plot(sources, df_plot["average_misassociated"], '-o', label=f"$\sigma^{2}$ = {var / 1000} km")
        axs[1].legend()
        axs[1].set_title("Data Association Error")
        axs[1].set_xlabel("Number of Sources")
        axs[1].set_ylabel("Average Number of Misassociated Measurements")

        # missed measurements
        axs[2].xaxis.get_major_locator().set_params(integer=True)
        axs[2].plot(sources, df_plot['average_missed'], '-o', label=f"$\sigma^{2}$ = {var / 1000} km")
        axs[2].legend()
        axs[2].set_title("Missed Measurements")
        axs[2].set_xlabel("Number of Sources")
        axs[2].set_ylabel("Average Number of Missed Measurements")

        # total fails (not really verafiable)
        axs[3].xaxis.get_major_locator().set_params(integer=True)
        axs[3].plot(sources, df_plot['total_fail_fraction'], '-o', label=f"$\sigma^{2}$ = {var / 1000} km")
        axs[3].legend()
        axs[3].set_title("Total Fails")
        axs[3].set_xlabel("Number of Sources")
        axs[3].set_ylabel("Total Fail Run Fraction")

        # all associations missed
        axs[4].xaxis.get_major_locator().set_params(integer=True)
        axs[4].plot(sources, df_plot['no_association_fraction'], '-o', label=f"$\sigma^{2}$ = {var / 1000} km")
        axs[4].legend()
        axs[4].set_title("Missed Associations (FN)")
        axs[4].set_xlabel("Number of Sources")
        axs[4].set_ylabel("Missed Association Run Fraction")

        # found associations when there were none
        axs[5].xaxis.get_major_locator().set_params(integer=True)
        axs[5].plot(sources, df_plot['false_association_fraction'], '-o', label=f"$\sigma^{2}$ = {var / 1000} km")
        axs[5].legend()
        axs[5].set_title("False Associations (FP)")
        axs[5].set_xlabel("Number of Sources")
        axs[5].set_ylabel("False Association Run Fraction")

        axs[6].xaxis.get_major_locator().set_params(integer=True)
        axs[6].plot(sources, df_plot['average_estimated_source_fraction'], '-o', label=f"$\sigma^{2}$ = {var / 1000} km")
        axs[6].legend()
        axs[6].set_title("Percent of Sources Detected That Are Possible")
        axs[6].set_xlabel("Number of Sources")
        axs[6].set_ylabel("Estimated Source Run Fraction")

    for ax in axs:
        ax.set_ylim(bottom=0)
        ax.grid()

    plt.show()

    if args.save_figs:
        fig_path = os.path.join(PROJECT_ROOT_DIR, "experiments", "results")
        figs[0].savefig(os.path.join(fig_path, 'loc_error.png'))
        figs[1].savefig(os.path.join(fig_path, 'data_assoc_error.png'))
        figs[2].savefig(os.path.join(fig_path, 'missed_measurements.png'))
        figs[3].savefig(os.path.join(fig_path, 'total_fails.png'))
        figs[4].savefig(os.path.join(fig_path, 'missed_associations.png'))
        figs[5].savefig(os.path.join(fig_path, 'false_associations.png'))
        figs[6].savefig(os.path.join(fig_path, 'percent_det.png'))



        
    

