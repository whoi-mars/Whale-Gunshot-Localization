from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt

import numpy as np
import pyproj as proj

from whale_gunshot_localization import config

def plot_localization(locs_est, locs_comp=None, buffer=6000, title=None, dates=None, bins=None, d_lat=0.1, d_lon=0.1, compare_latlon=False, save=None): 
    """
    Plot estimated source locations.

    Parameters
    ----------
    locs_est : List[np.array] or np.array, with each subarray of shape N X 2
        list of lists of estimated locations where the first column stores the Y
        coordinate and the second stores the X coordinates
    locs_comp : List[np.array] or np.array, with each subarray of shape N X 2
        list of lists of estimated locations where the first column stores the Y
        coordinate and the second stores the X coordinates
    buffer : float
        how much to plot outside of the limits established in the config file on either side
        of the width and height
    title : str
        plot title
    dates : List[datetime.datetime] or datetime.datatime
        list of dates associated with each sublist of location estiamtes
        in locs_est
    d_lat : float
        distance between lattitudes on the y-axis ticks
    d_lon : float
        distance between longitudes on the x-axis ticks
    compare_latlon : bool
        whether locs_comp are provided in lat/lon (True) or X/Y (False)
    save : str
        path at which to save the plot if desired

    """

    # format inputs
    if not isinstance(locs_est, list):
        locs_est = [locs_est]
    if locs_comp is not None and not isinstance(locs_comp, list):
        locs_comp = [locs_comp]
    if not isinstance(dates, list):
        if dates is None:
            dates = [None for _ in locs_est]
        else:
            dates = [dates]
    
    if locs_comp:
        assert len(locs_comp) == len(locs_est) == len(dates), "locs_est, locs_comp, and dates lists must have a one-to-one correspondence"
        n_subplots = 2
    else:
        assert len(locs_est) == len(dates), "locs_est and dates lists must have a one-to-one correspondence"
        n_subplots = 1

    # figure
    fig = plt.figure()

    # random list of color for plotting
    rng = np.random.default_rng(1111)
    colors = [rng.uniform(0, 1, size=3) for _ in range(len(locs_est))]

    # load constants
    TOSSIT_locations = np.asarray([config['TOSSIT']['TOSSIT_y'], config['TOSSIT']['TOSSIT_x']]).T
    min_x = config['scaling']['min_x']
    max_x = config['scaling']['max_x']
    min_y = config['scaling']['min_y']
    max_y = config['scaling']['max_y']

    # define projection objectalgorithm_sandbox
    pargs = proj.Proj(proj="aeqd", lat_0=41.9108, lon_0=-70.4292, datum="WGS84", units="m")

    # convert sensor locs to lat/lon
    lon_TOSSIT, lat_TOSSIT = pargs(TOSSIT_locations[:,1], -TOSSIT_locations[:,0], inverse=True)

    ax = fig.add_subplot(111 if n_subplots == 1 else 211)
    ax.set_title("Acoustic Detections")

    # define basemap
    width = abs(config['scaling']['max_x'] - config['scaling']['min_x']) + 2*buffer
    height = abs(config['scaling']['max_y'] - config['scaling']['min_y']) + 2*buffer
    m = Basemap(width=width, height=height, projection='aeqd',
                lat_0=41.9108, lon_0=-70.4292, resolution="f")
    lon, lat = pargs([-width / 2, width / 2], [-height / 2, height / 2], inverse=True)

    # fill background.
    m.drawmapboundary(fill_color='white')
    # draw coasts and fill continents.
    m.drawcoastlines(linewidth=0.8)
    m.fillcontinents(color='white',lake_color='white')


    m.plot(lon_TOSSIT, lat_TOSSIT, 'g^', markeredgecolor='black', latlon=True, label='sensors')
    
    for i, l in enumerate(locs_est):
        lon_est, lat_est = pargs(l[:,1], -l[:,0], inverse=True)
        if bins:
            H, x_edges, y_edges = np.histogram2d(lat_est, lon_est, bins=[np.arange(min(lat), max(lat) + bins, bins), np.arange(min(lon), max(lon) + bins, bins)])
            xx, yy = np.meshgrid(y_edges, x_edges)
            colormesh = m.pcolormesh(xx, yy, H / np.max(H), latlon=True, cmap='summer')
        m.plot(lon_est, lat_est, 'x', color=(colors[i][0], colors[i][1], colors[i][2]), latlon=True, label=f'estimate ({dates[i]})' if dates[0] else 'estimate')

    cb = m.colorbar(colormesh, location='right', label="Normalized Detections")

    m.drawparallels(np.arange(np.floor(lat[0]), np.ceil(lat[1]), d_lat), labels=[1, 0, 0, 0], color="None")
    m.drawmeridians(np.arange(np.floor(lon[0]), np.ceil(lon[1]), d_lon), labels=[0, 0, 0, 1], color="None")
    plt.legend(loc='upper left')

    if locs_comp:
        ax = fig.add_subplot(212)
        ax.set_title("Visual Detections")

        # define basemap
        width = abs(config['scaling']['max_x'] - config['scaling']['min_x']) + 2*buffer
        height = abs(config['scaling']['max_y'] - config['scaling']['min_y']) + 2*buffer
        m = Basemap(width=width, height=height, projection='aeqd',
                    lat_0=41.9108, lon_0=-70.4292, resolution="f")
        lon, lat = pargs([-width / 2, width / 2], [-height / 2, height / 2], inverse=True)

        # fill background.
        m.drawmapboundary(fill_color='white')
        # draw coasts and fill continents.
        m.drawcoastlines(linewidth=0.8)
        m.fillcontinents(color='white',lake_color='white')


        m.plot(lon_TOSSIT, lat_TOSSIT, 'g^', markeredgecolor='black', latlon=True, label='sensors')

        for i, l in enumerate(locs_comp):
            if compare_latlon:
                lat_vis, lon_vis = l[:,0], l[:,1]
            else:
                lon_vis, lat_vis = pargs(l[:,1], -l[:,0], inverse=True)

            if bins:
                H, x_edges, y_edges = np.histogram2d(lat_vis, lon_vis, bins=[np.arange(min(lat), max(lat) + bins, bins), np.arange(min(lon), max(lon) + bins, bins)])
                xx, yy = np.meshgrid(y_edges, x_edges)
                colormesh = m.pcolormesh(xx, yy, H / np.max(H), latlon=True, cmap='summer')
        m.plot(lon_vis, lat_vis, 'x', color=(colors[i][0], colors[i][1], colors[i][2]), latlon=True, label=f'estimate ({dates[i]})' if dates[0] else 'estimate')

        cb = m.colorbar(colormesh, location='right', label="Normalized Detections")

        m.drawparallels(np.arange(np.floor(lat[0]), np.ceil(lat[1]), d_lat), labels=[1, 0, 0, 0], color="None")
        m.drawmeridians(np.arange(np.floor(lon[0]), np.ceil(lon[1]), d_lon), labels=[0, 0, 0, 1], color="None")
        plt.legend(loc='upper left')
    
    if save is not None:
        plt.savefig(save)
    
    plt.close()