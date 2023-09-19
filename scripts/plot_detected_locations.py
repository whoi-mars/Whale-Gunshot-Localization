import os
import glob
import datetime
import pandas as pd
import numpy as np
from pathlib import Path
import argparse

from whale_gunshot_localization import config, PROJECT_ROOT_DIR
import whale_gunshot_localization.utils.plotting as plotting
import whale_gunshot_localization.utils.experimental as experimental

parser = argparse.ArgumentParser(description="Plot Localization Results")
parser.add_argument('--start', type=lambda ts : datetime.datetime.strptime(ts, '%Y-%m-%d %H:%M:%S'), default=None,
                    help='start timestamp of scan')
parser.add_argument('--end', type=lambda ts : datetime.datetime.strptime(ts, '%Y-%m-%d %H:%M:%S'), default=None,
                    help='start timestamp of scan')
# parser.add_argument('--compare', action='store_true',
#                     help='Compare acoustic detections to visual detections when detections for the same day exist.')
args = parser.parse_args()

def set_start_end_times(args):

    # get files associated with first sensor in ordered_sensors list
    wav_files = []
    for s in config['TOSSIT']['ids']:
        wav_files.extend(glob.glob(os.path.join(config['dataset']['ccb_data_directory'], s, "*.wav")))
    wav_files = np.asarray(wav_files)

    # sort wav files in chonological order by start time
    wav_files, start_times, end_times = experimental.sort_wav_chronological(wav_files)
    start_time_dict = dict(zip(wav_files, start_times))
    end_time_dict = dict(zip(wav_files, end_times))

    # set the start and end times
    if args.start is None:
        args.start = np.asarray(list(start_time_dict.values())).min().astype(datetime.datetime).replace(hour=0, minute=0, second=0)
        args.end = np.asarray(list(end_time_dict.values())).max().astype(datetime.datetime).replace(hour=0, minute=0, second=0)
    elif args.end is None:
        args.end = (args.start + datetime.timedelta(days=1))

    return args, start_time_dict, end_time_dict, wav_files

if __name__ == "__main__":
    
    # set up figure path
    fig_dir = os.path.join(PROJECT_ROOT_DIR, "scripts", "results", config['models']['model_dir'], "detection_maps")
    Path(fig_dir).mkdir(exist_ok=True, parents=True)

    # set the start and end times
    args, start_time_dict, end_time_dict, wav_files = set_start_end_times(args)

    # load scan results
    csv_path = os.path.join(PROJECT_ROOT_DIR, "scripts", "results", config['models']['model_dir'], "multi_scan_results.csv")
    if not os.path.exists(csv_path):
        raise IOError("no results file")
    df = pd.read_csv(csv_path)

    # load visual RW identifications
    df_vis = pd.read_excel(config['dataset']['ccb_visual_sightings'])

    # get bin edges
    bins_dt = pd.date_range(start=pd.to_datetime(start_time_dict[wav_files[0]]).date(), end=(pd.to_datetime(end_time_dict[wav_files[-1]]) + pd.Timedelta(1, "d")).date(), freq="D")
    df["bin"] = pd.to_datetime(pd.cut(pd.DatetimeIndex(df["global_timestamp"]), bins=bins_dt, labels=bins_dt[:-1]))
    df_vis["bin"] = df_vis["DATE"]

    # save row from days in common between visual and acoustic detections
    df_vis = df_vis[df_vis['bin'].isin(df['bin'])]
    df = df[df['bin'].isin(df_vis['bin'])]

    # all_locs_est = []
    # all_locs_est_vis = []
    # all_dates = []
    bin_grouped = df.groupby(by="bin")
    bin_grouped_vis = df_vis.groupby(by="bin")
    for acoustic_grouped, visual_grouped in zip(bin_grouped, bin_grouped_vis):

        # unpack dfs
        _, dfg_bin = acoustic_grouped
        _, dfg_bin_vis = visual_grouped

        id_grouped = dfg_bin.groupby(by="id")

        # get acoustic location estimates for a particular bin/day
        x, y = [],[]
        date = None 
        for _, dfg in id_grouped:
            dfg = dfg.reset_index()
            x.append(dfg.loc[0,"x"])
            y.append(dfg.loc[0,"y"])
            if date is None:
                date = dfg.loc[0,"bin"]
        locs_est = np.stack([y, x], axis=1)

        lon, lat = [], []
        for _, row in dfg_bin_vis.iterrows():
            lat += [row['LATITUDE']] * row['NUMBER']
            lon += [row['LONGITUDE']] * row['NUMBER']
        locs_comp = np.stack([lat, lon], axis=1)

        # plot locations by day
        plotting.plot_localization(locs_est, locs_comp=locs_comp, title="CCB-2023 Location Estimates", save=os.path.join(fig_dir, f"locations_{date.date()}.png"), dates=date.date(), bins=0.05, d_lat=0.2, d_lon=0.2, compare_latlon=True)
        # all_dates.append(date.date())
        # all_locs_est.append(locs_est)