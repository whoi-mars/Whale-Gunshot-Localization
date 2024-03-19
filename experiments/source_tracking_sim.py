import numpy as np
import pyproj as proj
import matplotlib.pyplot as plt
import itertools
import pandas as pd
import os
import argparse

import whale_gunshot_localization.sim_tools.sim_datagen as sim_datagen
from whale_gunshot_localization.utils.experimental import ParLocalizer, MultilaterationOpt
from whale_gunshot_localization import config, PROJECT_ROOT_DIR

# parse input arguments
parser = argparse.ArgumentParser(description="Run Monte Carlo simulations for data association/localization algorithm")
parser.add_argument('--simulate', action='store_true',
                    help="simulate rather than use previous results if available (default: false)")
parser.add_argument('--save_figs', action='store_true',
                    help="save figures (default: false)")
# parser.add_argument('--suppress_warnings', action='store_true',
#                     help="tell Python to suppress warnings")
parser.add_argument('--background', '-b', action='store_true',
                    help='silence the progress bar')
args = parser.parse_args()

def perc10(iterable):
    a = np.asarray(iterable)
    a = a[~np.isnan(a)]
    return np.percentile(a, 10)

def run_simulation(source_params, var, TOSSIT_locations, rng, localizer_params):

    for i, v in enumerate(var):

        # simulate signals
        range_measurements_list, source_associations_list, TOSSIT_associations_list, source_locs_list, source_ids_list = sim_datagen.generate_simple_paths(source_params, v ** 2, TOSSIT_locations, rng)
        max_sources = len(max(source_ids_list, key=len))

        # dataframe to store results
        if i == 0:
            columns = ["method_thresh", "num_sources", "std", "over_predict_sources", "FN", "FP", "percent_possible_detections"] + [f"loc_error_{i}" for i in range(max_sources)]
            df = pd.DataFrame(columns=columns)

        for step, (range_measurements, source_assocaitions, TOSSIT_associations, source_locs, source_ids) in enumerate(zip(range_measurements_list, source_associations_list, TOSSIT_associations_list, source_locs_list, source_ids_list)):
            
            if args.background:
                print(f"{step+1}/{np.max(source_params)}")

            # initialize results dict
            results = {
                "over_predict_sources": float('nan'),
                "FN": False,
                "FP": False,
                "percent_possible_detections": float('nan'),
            }

            # get number of possible associations
            assoc_flat = np.concatenate(source_assocaitions)
            possible_associations = []
            for p in range(max(assoc_flat) + 1):
                group = np.where(assoc_flat == p)[0]
                if len(group) >= localizer_params["min_assoc_size"]:
                    possible_associations.append(p)
            possible = (len(possible_associations) > 0)

            # data association/localizatoin
            L = ParLocalizer(**localizer_params)
            successful = L.set_measurements(range_measurements)

            if not successful:
                # check for FN
                if possible:
                    results["FN"] = True
            else:
                # check for FP
                if not possible:
                    results["FP"] = True
                    return results
                assocs_est, locs_est = L.associate_and_localize(last_step=True)
                if len(locs_est) == 0:
                    results["FN"] = True
                    return results

            # if no FP or FN, set over_predict_sources to False
            results["over_predict_sources"] = False

            # if we predict too many sources return
            if locs_est.shape[0] > source_locs.shape[0]:
                results["over_predict_sources"] = True

            # align source locations and estimated source locations
            num_ests = locs_est.shape[0]
            est_loc_combs = np.asarray(list(map(list, itertools.permutations(locs_est))))
            est_loc_idx_combs = np.asarray(list(map(list, itertools.permutations(np.arange(locs_est.shape[0])))))
            errors_matrix = np.sqrt(((est_loc_combs[:,:num_ests,:] - source_locs[np.newaxis,:,:]) ** 2).sum(axis=2)).sum(axis=1)
            res = np.sqrt(((est_loc_combs[np.argmin(errors_matrix),:num_ests,:] - source_locs) ** 2).sum(axis=1)).flatten()
            
            # add error values to appropriate lists
            detection_idxs = est_loc_idx_combs[np.argmin(errors_matrix)]
            names = [f"loc_error_{i}" for i in range(max_sources)]
            values = [float('nan') for i in range(max_sources)]
            loc_error_dict = dict(zip(names, values))        
            for err, idx in zip(res, detection_idxs):
                loc_error_dict[f"loc_error_{source_ids[idx]}"] = err

            save_dict = dict(zip(columns, [[localizer_params['multilat'].method_thresh], [len(source_locs)], [np.sqrt(var)], [results["over_predict_sources"]], [results["FN"]], [results["FP"]], [len(assocs_est) / len(possible_associations)]]))
            save_dict.update(loc_error_dict)
            df = pd.concat([df, pd.DataFrame(save_dict)], ignore_index=True)

    return df

if __name__ == "__main__":
    # derive TOSSIT locations relative to the first from the lat/lons
    TOSSIT_latlons = np.asarray([config['TOSSIT']['TOSSIT_lat'], config['TOSSIT']['TOSSIT_lon']]).T
    pargs = proj.Proj(proj="aeqd", lat_0=TOSSIT_latlons[0, 0], lon_0=TOSSIT_latlons[0, 1], datum="WGS84", units="m")
    xs, ys = pargs(TOSSIT_latlons[:,1], TOSSIT_latlons[:,0])
    TOSSIT_locations = np.asarray([-ys, xs]).T

    ############
    # Settings #
    ############ 
    csv_path = os.path.join(PROJECT_ROOT_DIR, "experiments", "results", "data_assoc_and_loc", "source_tracking_sim_results.csv")
    plot_path = os.path.join(PROJECT_ROOT_DIR, "experiments","results", "data_assoc_and_loc")

    localizer_params = dict(k=4,
                            multilat=MultilaterationOpt(method_thresh=float('inf')),
                            consistency_thresh=100,
                            TOSSIT_locations=TOSSIT_locations,
                            min_assoc_size=9)


    if args.simulate:                  
        df = run_simulation([(1,100), (15, 100), (25, 80), (40, 60)], 
                            std=[30], 
                            TOSSIT_locations=TOSSIT_locations, 
                            rng=np.random.default_rng(1324),
                            localizer_params=localizer_params)
        df.to_csv(csv_path, index=False)

    # load data
    df = pd.read_csv(csv_path)

    # get stats
    df_std = df.groupby(by=["std"]).agg(({'over_predict_sources': ['mean'],
                                          'FN': ['mean'],
                                          'FP': ['mean'],
                                          'loc_error_0': ['mean'],
                                          'loc_error_1': ['mean'],
                                          'loc_error_2': ['mean'],
                                          'loc_error_3': ['mean'],
                                          'percent_possible_detections': ['mean','std',perc10]}))
    print(df_std)

    # get results for each noise level
    df_std_list = [df[df["std"] == i] for i in [30]]

    if args.save_figs:

        # error of targets over time
        fig0 = plt.figure()
        for i in range(df['num_sources'].max()):
            plt.plot(df[f"loc_error_{i}"], label=f"target {i+1}")
        plt.legend()
        plt.grid()
        plt.xlabel("Time Step")
        plt.ylabel("Localization Error [m]")
        fig0.savefig(os.path.join(plot_path, "target_loc_error.png"))

        # number of sources over time
        fig1 = plt.figure()
        plt.plot(df_std_list[0]["num_sources"])
        plt.xlabel("Time Step")
        plt.ylabel("Numbe of Sources")
        fig1.savefig(os.path.join(plot_path, "present_sources.png"))


