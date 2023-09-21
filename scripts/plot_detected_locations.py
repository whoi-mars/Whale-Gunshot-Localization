"""
Script to plot acoustically derived whale locations and compare to visual estimates.
"""

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
                    help='start timestamp of plot data')
parser.add_argument('--end', type=lambda ts : datetime.datetime.strptime(ts, '%Y-%m-%d %H:%M:%S'), default=None,
                    help='start timestamp of plot data')
parser.add_argument('--compare', action='store_true',
                    help='Compare acoustic detections to visual detections when detections for the same day exist')
parser.add_argument('--bins', type=lambda x: None if x == 'None' else float(x), default=5000,
                    help='height/width of bins for heatmap of detections (meters)')
parser.add_argument('--d_lat', type=float, default=0.2,
                    help='delta in the latitude ticks on the y-axis (decimal degrees)')
parser.add_argument('--d_lon', type=float, default=0.2,
                    help='delta in the longitude ticks on the x-axis (decimal degrees)')
parser.add_argument('--buffer', type=float, default=7500,
                    help='buffer the height/width of the map frame (meters)')
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

    for date_bin in bins_dt:

        # get date-associated entries for each df
        dfg_bin = df[df["bin"] == date_bin]
        dfg_bin_vis = df_vis[df_vis["bin"] == date_bin]

        # if there are no acoustic detections move on
        if dfg_bin.empty:
            continue

        # group acoustic detections by id
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

        # if available and we want to compare, get visual detections
        if not dfg_bin_vis.empty and args.compare:
            lon, lat = [], []
            for _, row in dfg_bin_vis.iterrows():
                lat += [row['LATITUDE']] * row['NUMBER']
                lon += [row['LONGITUDE']] * row['NUMBER']
            locs_comp = np.stack([lat, lon], axis=1)
        else:
            locs_comp = None

        # plot locations by day
        fig = plotting.plot_localization(locs_est=locs_est, 
                                   locs_comp=locs_comp, 
                                   title_est="CCB-2023 Acoustic Detections", 
                                   title_comp="CCB-2023 Visual Detections",
                                   save=os.path.join(fig_dir, f"locations_{date.date()}.png"), 
                                   dates=date.date(), 
                                   buffer=args.buffer, 
                                   bins=args.bins, 
                                   d_lat=args.d_lat, 
                                   d_lon=args.d_lon,
                                   est_latlon=False, 
                                   compare_latlon=True)

        # all_dates.append(date.date())
        # all_locs_est.append(locs_est)