"""
Script to plot and compare acoustically derived whale locations and visual estimates
in a spatio temporal way relative to the time of the flight that got the visual detections.
"""

import os
import glob
import datetime
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import matplotlib.pyplot as plt

from whale_gunshot_localization import config, PROJECT_ROOT_DIR
import whale_gunshot_localization.utils.plotting as plotting
import whale_gunshot_localization.utils.experimental as experimental

parser = argparse.ArgumentParser(description="Plot Localization Results")
parser.add_argument('--start', type=lambda ts : datetime.datetime.strptime(ts, '%Y-%m-%d %H:%M:%S'), default=None,
                    help='start timestamp of plot data')
parser.add_argument('--end', type=lambda ts : datetime.datetime.strptime(ts, '%Y-%m-%d %H:%M:%S'), default=None,
                    help='start timestamp of plot data')
parser.add_argument('--d_lat', type=float, default=0.1,
                    help='delta in the latitude ticks on the y-axis (decimal degrees)')
parser.add_argument('--d_lon', type=float, default=0.1,
                    help='delta in the longitude ticks on the x-axis (decimal degrees)')
parser.add_argument('--buffer', type=float, default=15000,
                    help='buffer the height/width of the map frame (meters)')
args = parser.parse_args()

def set_start_end_times(args):
    """
    Properly set the start and end times for plotting from the argparse input dict

    Parameters
    ----------
    args : dict
        an argparse dictionary which contains the keys 'start', 'end'

    Returns
    -------
    args : dict
        original args input with updated 'start' and 'end' KVPs
    start_time_dict : dict
        dictionary which maps wave file name to start time
    end_time_dict : dict
        dictionary which maps wave file name to end time
    wav_files : List[str]
        list of wav file names in chronological order
    """

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

    # load plane paths
    df_flightpath = pd.read_excel(config['dataset']['ccb_visual_flightpath'])

    # get bin edges
    bins_dt = pd.date_range(start=pd.to_datetime(start_time_dict[wav_files[0]]).date(), end=(pd.to_datetime(end_time_dict[wav_files[-1]]) + pd.Timedelta(1, "d")).date(), freq="D")
    df["bin"] = pd.to_datetime(pd.cut(pd.DatetimeIndex(df["global_timestamp"]), bins=bins_dt, labels=bins_dt[:-1]))
    df_vis["bin"] = df_vis["DATE"]
    df_flightpath["bin"] = df_flightpath["DATE"]

    # save row from days in common between visual and acoustic detections
    df_vis = df_vis[df_vis['bin'].isin(df['bin'])]
    df = df[df['bin'].isin(df_vis['bin'])]
    df_flightpath = df_flightpath[df_flightpath['bin'].isin(df['bin'])]

    bin_grouped = df.groupby(by="bin")
    bin_grouped_vis = df_vis.groupby(by="bin")
    bin_grouped_flightpath = df_flightpath.groupby(by="bin")
    
    assert len(bin_grouped), "There are no common dates between the acoustic and visual data to compare."

    fig, axs = plt.subplots(1, len(bin_grouped), figsize=(18, 9))
    if not isinstance(axs, np.ndarray):
        axs = np.asarray([[axs]])
    elif len(axs.shape) == 1:
        axs = axs[np.newaxis,:]

    results = []
    for i, (acoustic_grouped, visual_grouped, flightpath_grouped) in enumerate(zip(bin_grouped, bin_grouped_vis, bin_grouped_flightpath)):

        # unpack dfs
        _, dfg_bin = acoustic_grouped
        _, dfg_bin_vis = visual_grouped
        _, dfg_bin_flightpath = flightpath_grouped

        id_grouped = dfg_bin.groupby(by="id")

        # get acoustic location estimates for a particular bin/day
        x, y = [],[]
        ac_time = []
        date = None 
        for _, dfg in id_grouped:
            dfg = dfg.reset_index()
            x.append(dfg.loc[0,"x"])
            y.append(dfg.loc[0,"y"])
            ac_time.append(datetime.datetime.strptime(dfg.loc[0, "global_timestamp"], '%Y-%m-%d %H:%M:%S'))
            if date is None:
                date = dfg.loc[0,"bin"]
        locs_est = np.stack([y, x], axis=1)

        lon, lat = [], []
        comp_time = []
        for _, row in dfg_bin_vis.iterrows():
            lat += [row['LATITUDE']] * row['NUMBER']
            lon += [row['LONGITUDE']] * row['NUMBER']
            str_time = str(row["TIME (GMT)"])
            comp_time += [row["DATE"].replace(hour=int(str_time[:2]), minute=int(str_time[2:4]), second=int(str_time[4:])).to_pydatetime()] * row['NUMBER']
        locs_comp = np.stack([lat, lon], axis=1)

        path_lon, path_lat = [], []
        path_time = []
        for _, row in dfg_bin_flightpath.iterrows():
            path_lat.append(row['LATITUDE'])
            path_lon.append(row['LONGITUDE'])
            str_time = str(row["TIME (GMT)"])
            path_time.append(row["DATE"].replace(hour=int(str_time[:2]), minute=int(str_time[2:4]), second=int(str_time[4:])).to_pydatetime())
        locs_path = np.stack([path_lat, path_lon], axis=1)

        success, _ = plotting.plot_localization_time_comparison(locs_est, 
                                          locs_comp,
                                          ac_time,
                                          comp_time,
                                          locs_path,
                                          path_time, 
                                          ax=axs[0,i], 
                                          buffer=args.buffer, 
                                          title=f"{date.date()}", 
                                          d_lat=args.d_lat, 
                                          d_lon=args.d_lon, 
                                          est_latlon=False,
                                          comp_latlon=True, 
                                          sensors=True, 
                                          est_name='acoustic', 
                                          comp_name='visual')
        results.append(success)

    for i, res in enumerate(results):
        if not res:
            fig.delaxes(axs[0,i])
    fig.savefig(os.path.join(fig_dir, "location_time_comparison.png"), dpi=200)