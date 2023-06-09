import os
import argparse

from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from whale_gunshot_localization import config, PROJECT_ROOT_DIR
from whale_gunshot_localization.utils.experimental import Localizer

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
        maximum number of measurements to hide/delete at each TOSSIT
        
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
        
        # randomly generate number of signals to delete at each TOSSIT. 
        # ensure that we have at least one signal per TOSSIT for simplicity.
        to_delete = rng.choice(np.arange(min(num_delete+1, len(range_measurements[t]))))
        del_list.append(to_delete)
        
        # generate arrays of source associations for the measurements
        source_associations.append(np.arange(num_sources))        
    
    # perform deletions
    for i, (m, a, t, d) in enumerate(zip(range_measurements, source_associations, TOSSIT_associations, del_list)):
        if d:
            del_idx = rng.choice(np.arange(len(m)), replace=False, size=d)
            range_measurements[i] = np.delete(m, del_idx)
            source_associations[i] = np.delete(a, del_idx)
            TOSSIT_associations[i] = np.delete(t, del_idx)
                
    # make sure smallest association value is 0
    bias = min([a[0] for a in source_associations if len(a)])
    if bias > 0:
        for i, a in enumerate(source_associations):
            source_associations[i] = a - bias
    
    return range_measurements, source_associations, TOSSIT_associations, source_locs

def monte_carlo_sim(n, num_sources_list, localizer_params, data_gen_params):

    columns = ["num_sources", "var", "delete", "k", "consistency_thresh", "method_thresh", "prune", "average_estimated_source_fraction", "average_misassociated", "average_misassociated_fraction", "average_location_error", "average_missed", "average_missed_fraction", "total_fail_fraction", "no_association_fraction", "false_association_fraction"]
    df = pd.DataFrame(columns=columns)

    l = Localizer(**localizer_params)

    with tqdm(total=len(num_sources_list) * n) as pbar:
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
                measurements, source_associations, TOSSIT_associations, source_locs = generate_measurements(num_sources=num_sources, **data_gen_params)
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
                successful = l.set_measurements(measurements)
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
                                              "var": [data_gen_params["var"]],
                                              "delete": [data_gen_params["num_delete"]],
                                              "k": [localizer_params["k"]],
                                              "consistency_thresh": [localizer_params["consistency_thresh"]],
                                              "method_thresh": [localizer_params["method_thresh"]],
                                              "prune": [localizer_params["prune"]],
                                              "average_estimated_source_fraction": [np.around(np.mean(estimated_frac_list), 4)],
                                              "average_misassociated":[np.around(np.mean(total_wrong_assoc_list), 4)],
                                              "average_misassociated_fraction": [np.around(np.mean(wrong_assoc_list), 4)],
                                              "average_location_error": [np.around(np.mean(loc_err_list), 4)],
                                              "average_missed": [np.around(np.mean(total_missed_list), 4)],
                                              "average_missed_fraction": [np.around(np.mean(total_missed_frac_list), 4)],
                                              "total_fail_fraction": [np.around(np.mean(total_fail_count / n), 4)],
                                              "no_association_fraction": [np.around(np.mean(missed_association_count / n), 4)],
                                              "false_association_fraction": [np.around(FP / n, 4)]})], ignore_index=True)

    return df

if __name__ == "__main__":

    # parse input arguments
    parser = argparse.ArgumentParser(description="Run Monte Carlo simulations for data association/localization algorithm")
    parser.add_argument('--simulate', action='store_true',
                        help="simulate rather than use previous results if available (default: false)")
    parser.add_argument('--save_figs', action='store_true',
                        help="save figures (default: false)")
    args = parser.parse_args()

    # path for results CSV
    path = os.path.join(PROJECT_ROOT_DIR, "experiments", "data_association_and_localization_sim_results.csv")

    ##########################################
    #          simulate/load results         #
    ##########################################

    if args.simulate:
        # set random seed
        rng = np.random.default_rng(1524)

        # parameters for localizer and data_generator
        localizer_params = dict(k=4, consistency_thresh=2000, method_thresh=0.95, prune=False, rng=rng)
        data_gen_params = dict(var=1000, num_delete=1, rng=rng)

        # run data association/localization experiment
        df = monte_carlo_sim(n=150, 
                             num_sources_list=range(1, 2), 
                             localizer_params=localizer_params, 
                             data_gen_params=data_gen_params,)
        df.to_csv(path, index=False)
    else:
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
    var_list = set(df['var'])
    
    for var in var_list:

        # get appropriate df rows
        df_plot = df[df['var'] == var].sort_values(by=['num_sources'])
        sources = df_plot['num_sources'].tolist()

        # localization error
        fig1 = plt.figure()
        ax1 = fig1.gca()
        ax1.xaxis.get_major_locator().set_params(integer=True)
        plt.plot(sources, df_plot['average_location_error'], '-o', label=f"Var = {var / 1000} km")
        ax1.set_ylim(bottom=0)
        ax1.set_xlim(left=1)
        plt.grid()
        plt.legend()
        plt.title("Localization Error")
        plt.xlabel("Number of Sources")
        plt.ylabel("Average Absolute Error [m]")

        # misassociated measurements
        fig2 = plt.figure()
        ax2 = fig2.gca()
        ax2.xaxis.get_major_locator().set_params(integer=True)
        plt.plot(sources, df_plot["average_misassociated"], '-o', label=f"Var = {var / 1000} km")
        ax2.set_ylim(bottom=0)
        ax2.set_xlim(left=1)
        plt.grid()
        plt.legend()
        plt.title("Data Association Error")
        plt.xlabel("Number of Sources")
        plt.ylabel("Average Number of Misassociated Measurements")

        # missed measurements
        fig3 = plt.figure()
        ax3 = fig3.gca()
        ax3.xaxis.get_major_locator().set_params(integer=True)
        plt.plot(sources, df_plot['average_missed'], '-o', label=f"Var = {var / 1000} km")
        ax3.set_ylim(bottom=0)
        ax3.set_xlim(left=1)
        plt.grid()
        plt.legend()
        plt.title("Missed Measurements")
        plt.xlabel("Number of Sources")
        plt.ylabel("Average Number of Missed Measurements")

        # total fails (not really verafiable)
        fig4 = plt.figure()
        ax4 = fig4.gca()
        ax4.xaxis.get_major_locator().set_params(integer=True)
        plt.plot(sources, df_plot['total_fail_fraction'], '-o', label=f"Var = {var / 1000} km")
        ax4.set_ylim(bottom=0)
        ax4.set_xlim(left=1)
        plt.grid()
        plt.legend()
        plt.title("Total Fails")
        plt.xlabel("Number of Sources")
        plt.ylabel("Total Fail Run Fraction")

        # all associations missed
        fig5 = plt.figure()
        ax5 = fig5.gca()
        ax5.xaxis.get_major_locator().set_params(integer=True)
        plt.plot(sources, df_plot['no_association_fraction'], '-o', label=f"Var = {var / 1000} km")
        ax5.set_ylim(bottom=0)
        ax5.set_xlim(left=1)
        plt.grid()
        plt.legend()
        plt.title("Missed Associations (FN)")
        plt.xlabel("Number of Sources")
        plt.ylabel("Missed Association Run Fraction")

        # found associations when there were none
        fig6 = plt.figure()
        ax6 = fig6.gca()
        ax6.xaxis.get_major_locator().set_params(integer=True)
        plt.plot(sources, df_plot['false_association_fraction'], '-o', label=f"Var = {var / 1000} km")
        ax6.set_ylim(bottom=0)
        ax6.set_xlim(left=1)
        plt.grid()
        plt.legend()
        plt.title("False Associations (FP)")
        plt.xlabel("Number of Sources")
        plt.ylabel("False Association Run Fraction")

        fig7 = plt.figure()
        ax7 = fig7.gca()
        ax7.xaxis.get_major_locator().set_params(integer=True)
        plt.plot(sources, df_plot['average_estimated_source_fraction'], '-o', label=f"Var = {var / 1000} km")
        ax7.set_ylim(bottom=0)
        ax7.set_xlim(left=1)
        plt.grid()
        plt.legend()
        plt.title("Percent of Sources Detected That Are Possible")
        plt.xlabel("Number of Sources")
        plt.ylabel("Estimated Source Run Fraction")

    plt.show()

    if args.save_figs:
        fig_path = os.path.join(PROJECT_ROOT_DIR, "experiments")
        fig1.savefig(os.path.join(fig_path, 'loc_error.png'))
        fig2.savefig(os.path.join(fig_path, 'data_assoc_error.png'))
        fig3.savefig(os.path.join(fig_path, 'missed_measurements.png'))
        fig4.savefig(os.path.join(fig_path, 'total_fails.png'))
        fig5.savefig(os.path.join(fig_path, 'missed_associations.png'))
        fig6.savefig(os.path.join(fig_path, 'false_associations.png'))
        fig7.savefig(os.path.join(fig_path, 'percent_det.png'))



        
    

