import os

import numpy as np
import pyproj as proj
import argparse
import timeit
from tqdm import tqdm
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

from whale_gunshot_localization.utils.experimental import MultilaterationOpt, ParLocalizer
from whale_gunshot_localization.sim_tools.sim_datagen import generate_measurements
from whale_gunshot_localization import config, PROJECT_ROOT_DIR

# parse input arguments
parser = argparse.ArgumentParser(description="Run Monte Carlo simulations for data association/localization algorithm")
parser.add_argument('--simulate', action='store_true',
                    help="simulate rather than use previous results if available (default: false)")
# parser.add_argument('--save_figs', action='store_true',
#                     help="save figures (default: false)")
# parser.add_argument('--suppress_warnings', action='store_true',
#                     help="tell Python to suppress warnings")
parser.add_argument('--background', '-b', action='store_true',
                    help='silence the progress bar')
args = parser.parse_args()

def get_time(n_measurements, T):
    num_sources = np.ceil(n_measurements / T.shape[0]).astype(int)
    num_delete = (num_sources) * T.shape[0] - n_measurements

    measurements, source_associations, TOSSIT_associations, source_locs = generate_measurements(num_sources=num_sources,
                                                                                                num_delete=num_delete,
                                                                                                rng=np.random.default_rng(1234),
                                                                                                in_sensors=True,
                                                                                                TOSSIT_locations=T,
                                                                                                var=30)
    
    L = ParLocalizer(k=4, multilat=MultilaterationOpt(method_thresh=float('inf')), consistency_thresh=75, TOSSIT_locations=T, min_assoc_size=4)

    result_sm = timeit.timeit("L.set_measurements(measurements)", number=1, globals=locals())
    # result_al = timeit.timeit("L.associate_and_localize(last_step=True)", number=1, globals=locals())

    return result_sm, None #result_al

if __name__ == "__main__":
    
    # path for results CSV
    path = os.path.join(PROJECT_ROOT_DIR, "experiments", "results", "time_trials")
    Path(path).mkdir(exist_ok=True, parents=True)

    if args.simulate:

        # derive TOSSIT locations relative to the first from the lat/lons
        TOSSIT_latlons = np.asarray([config['TOSSIT']['TOSSIT_lat'], config['TOSSIT']['TOSSIT_lon']]).T
        pargs = proj.Proj(proj="aeqd", lat_0=TOSSIT_latlons[0, 0], lon_0=TOSSIT_latlons[0, 1], datum="WGS84", units="m")
        xs, ys = pargs(TOSSIT_latlons[:,1], TOSSIT_latlons[:,0])
        TOSSIT_locations = np.asarray([-ys, xs]).T

        sm_times, al_times = [], []
        n_measurements_list = (17, 85)
        for i in tqdm(range(*n_measurements_list), disable=args.background):
            if i % 10 == 0:
                print(i)
            result_sm, result_al = get_time(i, TOSSIT_locations)
            sm_times.append(result_sm)
            al_times.append(result_al)

        df = pd.DataFrame({"n_measurements": [i for i in range(*n_measurements_list)],
                           "time_sm": sm_times,
                           "time_al": al_times})
        df.to_csv(os.path.join(path, "time_trials.csv"), index=False)


    ##########################################
    #               make plots               #
    ##########################################

    # read results
    df = pd.read_csv(os.path.join(path, "time_trials.csv"))

    # matlab settings
    matplotlib.rcParams.update({'font.size': 14})

    # get figs and axes
    num_figs = 1
    figs = [plt.figure() for _ in range(num_figs)]
    axs = [fig.gca() for fig in figs]

    axs[0].plot(df["n_measurements"], df["time_sm"])
    # axs[0].set_yscale("log")
    axs[0].set_xlabel("Number of Measurements")
    axs[0].set_ylabel("Time [s]")
    
    for ax in axs:
        ax.grid()

    fig_path = os.path.join(PROJECT_ROOT_DIR, "experiments","results", "time_trials")
    figs[0].savefig(os.path.join(fig_path, "time_trials.png"))
