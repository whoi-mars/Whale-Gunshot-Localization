import numpy as np
import pyproj as proj
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.patches as mpatches
import seaborn as sns
import itertools
import pandas as pd
pd.options.display.max_columns = 50
import os
import argparse
import scipy
import scipy.stats as stats

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
def cdf1(iterable):
    a = np.asarray(iterable)
    a = a[~np.isnan(a)]
    return len(a[a == 1.0]) / len(a)

def run_simulation(source_params, std_list, TOSSIT_locations, seed, localizer_params, loc0=None, bearing0=None, beamwidth=None, n_del=0):

    for i, std in enumerate(std_list):

        localizer_params_final = localizer_params.copy()
        localizer_params_final['consistency_thresh'] = localizer_params_final['consistency_thresh'][std]
        localizer_params_final['multilat'] = localizer_params_final['multilat'][std]
        localizer_params_final['k'] = localizer_params_final['k'][std]

        for ii, nd in enumerate(n_del):
            rng = np.random.default_rng(seed)

            # simulate signals
            range_measurements_list, source_associations_list, TOSSIT_associations_list, source_locs_list, source_ids_list = sim_datagen.generate_simple_paths(source_params, std ** 2, TOSSIT_locations, rng, loc0=loc0.copy(), bearing0=bearing0.copy(), beamwidth=beamwidth, n_del=nd)
            max_sources = len(max(source_ids_list, key=len))

            # dataframe to store results
            if i == 0 and ii == 0:
                columns = ["method_thresh", "num_sources", "std", "over_predict_sources", "FN", "FP", "percent_possible_detections", "n_delete"] + [f"loc_error_{i}" for i in range(max_sources)] + [f"best_loc_error_{i}" for i in range(max_sources)]
                df = pd.DataFrame(columns=columns)

            for step, (range_measurements, source_assocaitions, TOSSIT_associations, source_locs, source_ids) in enumerate(zip(range_measurements_list, source_associations_list, TOSSIT_associations_list, source_locs_list, source_ids_list)):                
                if args.background:
                    if step == 0 and nd == 0:
                        print(f"working on std = {std}...")
                    if step == 0 and nd > 0:
                        print(f"working on (std, n_del) = ({std} m, {nd})...")
                    print(f"{step+1}/{np.max(source_params)}")

                if len(source_ids) != 4:
                    continue

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
                    if len(group) >= localizer_params_final["min_assoc_size"]:
                        possible_associations.append(p)
                possible = (len(possible_associations) > 0)

                # data association/localizatoin
                L = ParLocalizer(**localizer_params_final)
                successful = L.set_measurements(range_measurements)

                if not successful:
                    # check for FN
                    if possible:
                        results["FN"] = True
                else:
                    # check for FP
                    if not possible:
                        results["FP"] = True
                        assocs_est = []
                        locs_est = []
                    else:
                        assocs_est, locs_est = L.associate_and_localize(last_step=True)
                        if len(locs_est) == 0:
                            results["FN"] = True


                if not results["FN"] and not results["FP"]:
                    # if we predict too many sources return
                    if locs_est.shape[0] > source_locs.shape[0]:
                        results["over_predict_sources"] = True
                    else:
                        # if no FP or FN, set over_predict_sources to False
                        results["over_predict_sources"] = False
                        

                names_est = [f"loc_error_{i}" for i in range(max_sources)]
                names_best = [f"best_loc_error_{i}" for i in range(max_sources)]
                values = [float('nan') for i in range(max_sources)]
                loc_error_dict = dict(zip(names_est, values))
                best_loc_error_dict = dict(zip(names_best, values))       
                if results["over_predict_sources"] == False:
                    # align source locations and estimated source locations
                    num_ests = locs_est.shape[0]
                    nans = np.zeros((source_locs.shape[0] - num_ests, 2))
                    nans[:] = np.nan
                    locs_est = np.concatenate((locs_est, nans), axis=0)

                    est_loc_combs = np.asarray(list(map(list, itertools.permutations(locs_est))))
                    est_loc_idx_combs = np.asarray(list(map(list, itertools.permutations(np.arange(locs_est.shape[0])))))
                    errors_matrix = np.nansum(np.sqrt(((est_loc_combs - source_locs[np.newaxis,:,:]) ** 2).sum(axis=2)), axis=1)
                    res = np.sqrt(((est_loc_combs[np.argmin(errors_matrix),:,:] - source_locs) ** 2).sum(axis=1)).flatten()
                    
                    # add error values to appropriate lists
                    for err, sid in zip(res, source_ids):
                        loc_error_dict[f"loc_error_{sid}"] = err

                    # get best localizations
                    measurements_flat = np.concatenate(range_measurements)
                    TOSSIT_flat = np.concatenate(TOSSIT_associations)
                    assoc_flat = np.concatenate(source_assocaitions)
                    localizer = localizer_params_final['multilat']
                    localizer.set_map({
                        'TOSSIT_locations' : localizer_params_final['TOSSIT_locations'],
                        'min_x' : config['scaling']['min_x'],
                        'max_x' : config['scaling']['max_x'],
                        'min_y' : config['scaling']['min_y'],
                        'max_y' : config['scaling']['max_y'],
                    })

                    detection_idxs = est_loc_idx_combs[np.argmin(errors_matrix)]
                    best_locs = []
                    for c, didx in enumerate(detection_idxs):
                        if np.isnan(locs_est[didx,:].sum()):
                            best_locs.append(np.asarray([float('nan'), float('nan')]))
                        else:
                            idx = np.where(assoc_flat == c)[0]
                            _, loc = localizer.localize(measurements_flat[idx], TOSSIT_flat[idx])
                            best_locs.append(loc)
                    best_locs = np.asarray(best_locs)
                    best_res = np.sqrt(((best_locs - source_locs) ** 2).sum(axis=1))
                    for err, sid in zip(best_res, source_ids):
                        best_loc_error_dict[f"best_loc_error_{sid}"] = err

                if results["over_predict_sources"] == False:
                    save_dict = dict(zip(columns, [[localizer_params_final['multilat'].method_thresh], [len(source_locs)], [std], [results["over_predict_sources"]], [results["FN"]], [results["FP"]], [len(assocs_est) / len(possible_associations)], [nd]]))
                else:
                    save_dict = dict(zip(columns, [[localizer_params_final['multilat'].method_thresh], [len(source_locs)], [std], [results["over_predict_sources"]], [results["FN"]], [results["FP"]], [float('nan')], [nd]]))
                save_dict.update(loc_error_dict)
                save_dict.update(best_loc_error_dict)
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
    # paths
    csv_path = os.path.join(PROJECT_ROOT_DIR, "experiments", "results", "missing_data_tracking_sim", "source_tracking_sim_results.csv")
    stats_csv_path = os.path.join(PROJECT_ROOT_DIR, "experiments", "results", "missing_data_tracking_sim", "source_tracking_sim_stats.csv")
    plot_path = os.path.join(PROJECT_ROOT_DIR, "experiments","results", "missing_data_tracking_sim")
   
    # parameters
    seed = 1324
    num_sources = 4
    source_params = [(1,50) for _ in range(num_sources)] # [(1,100), (15, 100), (25, 80), (40, 60)]
    # source_params = [(1, 200), (50, 200), (70, 170), (85, 135)]
    std_list = [660]
    beamwidth = None
    num_delete = [0]
    # source_locs = np.asarray([[-5600.,1000.],
    #                           [-5100.,200.],
    #                           [-3800.,-2600.],
    #                           [-2000.,-3700]])
    # source_locs = np.asarray([[-7200.,-1600.],
    #                         [-5000.,-1900.],
    #                         [-1000.,-1000],
    #                         [-4000.,-5600.]])
    # bearings = np.asarray([-40., -45., 200., 280.])
    source_locs = np.asarray([[-7200.,-1600.],
                            [-5700.,-2500.],
                            [-700.,-1000],
                            [-4200.,-5200.]])
    bearings = np.asarray([-37., -48., 200., 255.])
    localizer_params = dict(k={0: 4, 15: 4, 30: 4, 660: 5},
                            multilat={i : MultilaterationOpt(method_thresh=float('inf'), seed=seed) for i in std_list},
                            consistency_thresh={0: 2, 15: 100, 30: 100, 660: 500},
                            TOSSIT_locations=TOSSIT_locations,
                            min_assoc_size=9)

    if args.simulate:                  
        df = run_simulation(source_params, 
                            std_list=std_list,
                            n_del=num_delete, 
                            TOSSIT_locations=TOSSIT_locations, 
                            seed=seed,
                            localizer_params=localizer_params,
                            loc0=source_locs,
                            bearing0=bearings,
                            beamwidth=beamwidth)
        df.to_csv(csv_path, index=False)

    # load data
    df = pd.read_csv(csv_path)

    # get stats
    base_agg_dict = {'over_predict_sources': ['mean'],
                     'FN': ['mean'],
                     'FP': ['mean'],
                     'percent_possible_detections': ['mean','std',perc10,cdf1]}
    loc_agg_dict = {i : ['mean', 'std'] for i in list(df.columns[df.columns.str.contains('loc_error_')])}
    best_loc_agg_dict = {i : ['mean', 'std'] for i in list(df.columns[df.columns.str.contains('best_loc_error_')])}
    base_agg_dict.update(loc_agg_dict)
    base_agg_dict.update(best_loc_agg_dict)
    df_std = df.groupby(by=["std", "num_sources", "n_delete"]).agg((base_agg_dict))
    print(df_std)
    df_std.to_csv(stats_csv_path)

    # get results for each noise level
    std_list = np.unique(df["std"])
    num_del_list = np.unique(df["n_delete"])
    df_std_ndel_list = [df[(df["std"] == i) & df["n_delete"] == j] for i in std_list for j in num_del_list]

    if args.save_figs:

        # number of sources over time
        fig1 = plt.figure()
        plt.plot(df_std_ndel_list[0]["num_sources"])
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.xlabel("Time Step", fontsize=18)
        plt.ylabel("Numbe of Sources", fontsize=18)
        plt.grid()
        fig1.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
        fig1.savefig(os.path.join(plot_path, "present_sources.png"))

        # plot source paths
        _, _, _, source_locs_list, source_ids_list = sim_datagen.generate_simple_paths(source_params, 0, TOSSIT_locations, np.random.default_rng(seed), loc0=source_locs.copy(), bearing0=bearings.copy(), beamwidth=beamwidth)
        max_sources = len(max(source_ids_list, key=len))
        xloc = [[] for _ in range(max_sources)]
        yloc = [[] for _ in range(max_sources)]
        for i, j in zip(source_locs_list, source_ids_list):
            for ii, jj in zip(i, j):
                xloc[jj].append(ii[1])
                yloc[jj].append(ii[0])
        fig2 = plt.figure()
        fig2.gca().set_axisbelow(True)
        for tn, (xx, yy) in enumerate(zip(xloc, yloc)):
            plt.plot(np.asarray(xx) / 1000, -np.asarray(yy) / 1000, linewidth=4, label=f"target {tn+1}", zorder=2)
        plt.plot(TOSSIT_locations[:,1] / 1000, -TOSSIT_locations[:,0] / 1000, 'kX', markersize=15, zorder=1, label="TOSSIT")
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.xlabel("X [km]", fontsize=18)
        plt.ylabel("Y [km]", fontsize=18)
        plt.grid()
        leg = plt.legend(prop={'size': 14})
        leg.get_frame().set_linewidth(3.0)
        plt.axis('square')
        plt.ylim(-4, 8)
        fig2.savefig(os.path.join(plot_path, "source_paths.png"))

        # plot
        df_auto = pd.melt(df, var_name='Target', value_name='Loc_Error', id_vars=['std', 'num_sources', 'n_delete'], value_vars=[f'loc_error_{i}' for i in range(df['num_sources'].max())])
        df_auto['Type'] = "Automatic"
        df_man = pd.melt(df, var_name='Target', value_name='Loc_Error', id_vars=['std', 'num_sources', 'n_delete'], value_vars=[f'best_loc_error_{i}' for i in range(df['num_sources'].max())])
        df_man['Type'] = "Manual"
        df_long = df_auto.merge(df_man, how="outer")

        df_long_high = df_long[df_long["std"] >= 100]
        df_long_low = df_long[df_long["std"] < 100]
        std_list_high = [j for j in std_list if j >= 100]
        std_list_low = [j for j in std_list if j < 100]

        # box
        def d(inp):
            if inp[:5] == "best_":
                return inp[5:]
            else:
                return inp
        colors = sns.color_palette("tab10")

        max_del = max(num_delete)
        fig3, spax = plt.subplots(1, len(num_delete), figsize=(15,10), sharey=True)
        fig3.subplots_adjust(wspace=0)
        for i in range(len(num_delete)):
            dft = df_long_low[df_long_low['n_delete'] == num_delete[i]]
            targets = dft['Target']
            dft['Target'] = targets.apply(d)
            b = sns.boxplot(x=dft['Type'], 
                            y=dft['Loc_Error'], 
                            hue=dft[['Target', 'std']].apply(tuple, axis=1), 
                            showfliers=True,
                            showmeans=True,
                            hue_order=[(f'loc_error_{k}', m) for m in std_list_low for k in range(max_sources)],
                            linewidth=1,
                            meanprops={"marker":"s","markerfacecolor":"white", "markeredgecolor":"blue"},
                            ax=spax[i])
            
            plist = list(spax[i].patches)
            plist = [p for p in plist if "Rectangle" not in str(p)]
            c = 0
            for idx, p in enumerate(plist):
                p.set_facecolor(colors[c])
                if (idx+1) % (max_sources) == 0:
                    c += 1
                if (idx+1) % (len(std_list_low)*(max_sources)) == 0:
                    c = 0

            # LEGEND
            if i < max_del - 1:
                spax[i].legend([],[], frameon=False)
            
            # LABELS
            if i+1 > 1:
                spax[i].set_ylabel("")
            else:
                spax[i].set_ylabel("Localization Error [m]", fontsize=22, labelpad=20)
            spax[i].set_xlabel(f"{np.around(100 *num_delete[i] / TOSSIT_locations.shape[0], 2)}%", fontsize=20)
            spax[i].grid(axis='y')
            b.tick_params(labelsize=18)
        
        spax[-1].legend(handles=[mpatches.Patch(color=colors[i], label="$\sigma_{r}$ = " + f"{std} m") for (i, std) in enumerate(std_list_low)], prop={'size': 14})
        fig3.text(0.5, 0.01, 'Missing Data', ha='center', fontsize=18)
        fig3.savefig(os.path.join(plot_path, "del_analysis_box_low.png"))

        # box
        max_del = max(num_delete)
        fig4, spax = plt.subplots(1, len(num_delete), figsize=(15,10), sharey=True)
        fig4.subplots_adjust(wspace=0)
        for i in range(len(num_delete)):
            dft = df_long_high[df_long_high['n_delete'] == num_delete[i]]
            targets = dft['Target']
            dft['Target'] = targets.apply(d)
            b = sns.boxplot(x=dft['Type'], 
                            y=dft['Loc_Error'], 
                            hue=dft[['Target', 'std']].apply(tuple, axis=1), 
                            showfliers=True,
                            showmeans=True,
                            hue_order=[(f'loc_error_{k}', m) for m in std_list_high for k in range(max_sources)],
                            linewidth=1,
                            meanprops={"marker":"s","markerfacecolor":"white", "markeredgecolor":"blue"},
                            ax=spax[i])
            
            plist = list(spax[i].patches)
            plist = [p for p in plist if "Rectangle" not in str(p)]
            c = 0
            for idx, p in enumerate(plist):
                p.set_facecolor(colors[c])
                if (idx+1) % (max_sources) == 0:
                    c += 1
                if (idx+1) % (len(std_list_high)*(max_sources)) == 0:
                    c = 0

            # LEGEND
            if i < max_del - 1:
                spax[i].legend([],[], frameon=False)
            
            # LABELS
            if i+1 > 1:
                spax[i].set_ylabel("")
            else:
                spax[i].set_ylabel("Localization Error [m]", fontsize=22, labelpad=20)
            spax[i].set_xlabel(f"{np.around(100 *num_delete[i] / TOSSIT_locations.shape[0], 2)}%", fontsize=20)
            spax[i].grid(axis='y')
            b.tick_params(labelsize=18)
        
        spax[-1].legend(handles=[mpatches.Patch(color=colors[i], label="$\sigma_{r}$ = " + f"{std} m") for (i, std) in enumerate(std_list_high)], prop={'size': 14})
        fig4.text(0.5, 0.01, 'Missing Data', ha='center', fontsize=18)
        fig4.savefig(os.path.join(plot_path, "del_analysis_box_high.png"))

        # number of sources detected
        df_high = df[df["std"] == 660]
        fig5, axs = plt.subplots(1,len(num_delete), sharey=True)
        for c, nd in enumerate(num_delete):
            dft = df_high[df_high['n_delete'] == nd]
            counts, bins = np.histogram(dft['percent_possible_detections']*4, bins=np.arange(0,5)+0.5)
            axs[c].hist(bins[:-1], bins, weights=counts, ec='k')
            axs[c].xaxis.set_major_locator(MaxNLocator(integer=True))
            axs[c].grid(axis='y')
            axs[c].set_axisbelow(True)
            if c == 0:
                axs[c].set_ylabel("Number of Algorithm Runs")
            axs[c].set_xlabel("Number of Sources Detected")
        fig5.savefig(os.path.join(plot_path, "num_sources_detected.png"))

        # number of sources detected
        df_high = df[df["std"] == 660]
        fig6, axs = plt.subplots(1,len(num_delete), sharey=True)
        bins = np.arange(0,5)+0.5
        for c, nd in enumerate(num_delete):
            dft = df_high[df_high['n_delete'] == nd]
            counts = []
            for i in range(num_sources):
                counts.append(100*np.sum(~np.isnan(dft[f'loc_error_{i}'])) / len(dft[f'loc_error_{i}']))
            axs[c].hist(bins[:-1], bins, weights=counts, ec='k')
            axs[c].xaxis.set_major_locator(MaxNLocator(integer=True))
            axs[c].grid(axis='y')
            axs[c].set_axisbelow(True)
            if c == 0:
                axs[c].set_ylabel("Detection Rate [%]")
            axs[c].set_xlabel("Target Number")
        fig6.savefig(os.path.join(plot_path, "per_source_detection.png"))
        
