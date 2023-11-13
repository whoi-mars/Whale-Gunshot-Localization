from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.colors import ListedColormap
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import os

import numpy as np
import pyproj as proj
from PIL import Image

from whale_gunshot_localization.utils.experimental import calculate_GDOP_locs
from whale_gunshot_localization import config

def plot_localization_time_comparison(locs_est, locs_comp, est_times, comp_times, comp_path, comp_path_times, ax=None, buffer=6000, title=None, d_lat=0.1, d_lon=0.1, est_latlon=False, comp_latlon=False, sensors=False, est_name='est', comp_name='comp'):
    """
    Plot both visually and acoustically obtained location estimates of whales
    that exist within the duration of the flight that obtained the visual estimates.
    The flight path is also plotted. This can only be done if there is at least one
    acoustic detection during the flight

    Parameters
    ----------
    locs_est : np.array, of shape N X 2
        list of lists of estimated locations where the first column stores the Y
        coordinate and the second stores the X coordinates
    locs_comp : np.array, of shape M X 2
        list of lists of estimated locations where the first column stores the Y
        coordinate and the second stores the X coordinates
    est_times : List[datetime.datetime]
        list of datetime objects which correspond to the acoustic detections in
        locs_est
    comp_times : List[datetime.datatime]
        list of datetime objects which correspond to the visual detections in
        locs_comp
    comp_path : np.array, of shape K X 2
        the path of the plane that took the visual measuremnets went.
        the first column has latitudes and the second has longitudes.
    comp_path_times : List[datetime.datatime]
        list of datatimes associated with the flight path
    ax : Axis
        axis to put the plot on
    buffer : float
        how much to plot outside of the limits established in the config file on either side
        of the width and height
    title : str
        plot title
    d_lat : float
        distance between lattitudes on the y-axis ticks
    d_lon : float
        distance between longitudes on the x-axis ticks
    est_latlon : bool
        whether locs_est are provided in lat/lon (True) or X/Y (False)
    comp_latlon : bool
        whether locs_comp are provided in lat/lon (True) or X/Y (False)
    sensors : bool
        whether or not to plot the sensors
    est_name : str
        name of the type of estimate
    comp_name : str
        name of the type of comparison

    Returns
    -------
    : bool
        whether or not there were common acoustic and visual detections during the flight duration
    : figure
        either None or a figure object depending on if an axis object was passed
    """
    # filter points to be within the duration of the comp_path
    est_times_new = []
    comp_times_new = []
    locs_est_new = []
    locs_comp_new = []
    for i, et in enumerate(est_times):
        if et >= np.min(comp_path_times) and et <= np.max(comp_path_times):
            est_times_new.append(et)
            locs_est_new.append(locs_est[i])
    for i, ct in enumerate(comp_times):
        if ct >= np.min(comp_path_times) and ct <= np.max(comp_path_times):
            comp_times_new.append(ct)
            locs_comp_new.append(locs_comp[i])
    est_times = est_times_new
    comp_times = comp_times_new
    locs_est = np.asarray(locs_est_new)
    locs_comp = np.asarray(locs_comp_new)

    if len(locs_est) == 0:
        return False, None

    # random list of color for plotting
    rng = np.random.default_rng(386729407)
    colors = [rng.uniform(0, 1, size=3) for _ in range(2)]

    # load constants
    TOSSIT_locations = np.asarray([config['TOSSIT']['TOSSIT_y'], config['TOSSIT']['TOSSIT_x']]).T
    min_x = config['scaling']['min_x']
    max_x = config['scaling']['max_x']
    min_y = config['scaling']['min_y']
    max_y = config['scaling']['max_y']

    # instantiate subplots
    if ax is None:
        return_fig = True
        fig, ax = plt.subplots(1,1)
    else:
        return_fig = False

    ax.set_title(title)

    # define basemap
    width = abs(config['scaling']['max_x'] - config['scaling']['min_x']) + 2*buffer
    height = abs(config['scaling']['max_y'] - config['scaling']['min_y']) + 2*buffer
    m = Basemap(width=width, height=height, projection='aeqd',
                lat_0=config['TOSSIT']['map_origin_latlon'][0], lon_0=config['TOSSIT']['map_origin_latlon'][1], resolution="f", ellps="WGS84", ax=ax)
    # fill background.
    m.drawmapboundary(fill_color='white')
    # draw coasts and fill continents.
    m.drawcoastlines(linewidth=0.8)
    m.fillcontinents(color='white',lake_color='white')

    # define the x/y offsets because Basemap defines the origin in the lower left of the map
    x_offset = width / 2
    y_offset = height / 2

    # get lon/lat of map edges
    lon, lat = m([(-width / 2) + x_offset, (width / 2) + x_offset], [(-height / 2) + y_offset, (height / 2) + y_offset], inverse=True)

    # plot sensors
    if sensors:
        m.plot(TOSSIT_locations[:,1] + x_offset, -TOSSIT_locations[:,0] + y_offset, markerfacecolor='red', linestyle="None", marker='^', markeredgecolor='black', latlon=False, label='sensors', zorder=1)

    if est_latlon:
        # convert lat/lon to x/y bins so we can bin if we need to
        x, y = locs_est[:,1], locs_est[:,0]
        x, y = m(x, y)
    else:
        x, y = locs_est[:,1] + x_offset, -locs_est[:,0] + y_offset

    if comp_latlon:
        # convert lat/lon to x/y bins so we can bin if we need to
        x_c, y_c = locs_comp[:,1], locs_comp[:,0]
        x_c, y_c = m(x_c, y_c)
        x_p, y_p = comp_path[:,1], comp_path[:,0]
        x_p, y_p = m(x_p, y_p)
    else:
        x_c, y_c = locs_comp[:,1] + x_offset, -locs_comp[:,0] + y_offset
        x_p, y_p = comp_path[:,1] + x_offset, -comp_path[:,0] + y_offset

    # plot flightpath color coded by time
    tmin = np.min(comp_path_times)
    #if comp_path_times is None:
    #    m.plot(x_p, y_p, latlon=False, label="flight path")
    #else:
    colors_cpt = np.asarray([(cpt - tmin).total_seconds() for cpt in comp_path_times])
    norm = np.max(colors_cpt)
    colors_cpt /= norm

    path_points = np.array([x_p, y_p]).T.reshape(-1, 1, 2)
    segments = np.concatenate([path_points[:-1], path_points[1:]], axis=1)
    lc = LineCollection(segments, cmap='viridis', norm=plt.Normalize(colors_cpt.min(), colors_cpt.max()), zorder=1)
    lc.set_array(colors_cpt)
    lc.set_linewidth(0.5)
    line = ax.add_collection(lc)


    # plot detections color coded by time
    # if est_times is not None and comp_times is not None:
    ct_diff = np.asarray([(ct - tmin).total_seconds() for ct in comp_times])
    et_diff = np.asarray([(et - tmin).total_seconds() for et in est_times])

    colors_ct = ct_diff / norm
    colors_et = et_diff / norm
    # else:
    #     colors_ct, colors_et = None, None

    sp1 = m.scatter(x, y, marker='P', s=36, label=f"{est_name} esimate", c=colors_et, norm=lc.norm, edgecolors='black', linewidth=0.25, zorder=3)
    sp2 = m.scatter(x_c, y_c, marker='o', s=36, label=f"{comp_name} estimate", c=colors_ct, norm=lc.norm, edgecolors='black', linewidth=0.25, zorder=2)
    cb = m.colorbar(line, location='bottom', label='Normalized Time')

    ax.legend(loc='lower left')
    handles, labels = ax.get_legend_handles_labels()
    if sensors:
        new_handles = [handles[0], \
                       Line2D([0], [0], marker='P', markerfacecolor='black', markeredgecolor='black', markersize=6, ls=''), \
                       Line2D([0], [0], marker='o', markerfacecolor='black', markeredgecolor='black', markersize=6, ls='')]
    else:
        new_handles = [Line2D([0], [0], marker='P', markerfacecolor='black', markeredgecolor='black', markersize=6, ls=''), \
                       Line2D([0], [0], marker='o', markerfacecolor='black', markeredgecolor='black', markersize=6, ls='')]
    ax.legend(new_handles, labels, loc='lower left')

    if return_fig:
        return True, fig
    else:
        return True, None

def plot_localization(locs_est, ax=None, buffer=6000, title=None, bins=None, d_lat=0.1, d_lon=0.1, est_latlon=False, points=False, sensors=False, name='est'): 
    """
    Plot estimated source locations.

    Parameters
    ----------
    locs_est : List[np.array] or np.array, with each subarray of shape N X 2
        list of lists of estimated locations where the first column stores the Y
        coordinate and the second stores the X coordinates
    ax : Axis
        axis for plot. if not provided returns a figure
    buffer : float
        how much to plot outside of the limits established in the config file on either side
        of the width and height
    title : str
        title for plot
    bins : flot
        if None plots the points, else plots regions of detections using bins and the dimensions
    d_lat : float
        distance between lattitudes on the y-axis ticks
    d_lon : float
        distance between longitudes on the x-axis ticks
    est_latlon : bool
        whether locs_est are provided in lat/lon (True) or X/Y (False)
    points : bool
        whether to plot the location points (True) or not (False)
    sensors : bool
        whether or not to plot the sensors
    name : str
        name of the type of estimate
    """

    # if we're not plotting bins, we need to plot points
    if bins is None:
        points = True

    # random list of color for plotting
    rng = np.random.default_rng(386729407)
    colors = [rng.uniform(0, 1, size=3) for _ in range(2)]

    # load constants
    TOSSIT_locations = np.asarray([config['TOSSIT']['TOSSIT_y'], config['TOSSIT']['TOSSIT_x']]).T
    min_x = config['scaling']['min_x']
    max_x = config['scaling']['max_x']
    min_y = config['scaling']['min_y']
    max_y = config['scaling']['max_y']

    # instantiate subplots
    if ax is None:
        return_fig = True
        fig, ax = plt.subplots(1,1)
    else:
        return_fig = False

    ax.set_title(title)

    # define basemap
    width = abs(config['scaling']['max_x'] - config['scaling']['min_x']) + 2*buffer
    height = abs(config['scaling']['max_y'] - config['scaling']['min_y']) + 2*buffer
    m = Basemap(width=width, height=height, projection='aeqd',
                lat_0=config['TOSSIT']['map_origin_latlon'][0], lon_0=config['TOSSIT']['map_origin_latlon'][1], resolution="f", ellps="WGS84")
    # fill background.
    m.drawmapboundary(fill_color='aqua' if bins is None else 'white')
    # draw coasts and fill continents.
    m.drawcoastlines(linewidth=0.8)
    m.fillcontinents(color='coral' if bins is None else 'white',lake_color='aqua' if bins is None else 'white')

    # define the x/y offsets because Basemap defines the origin in the lower left of the map
    x_offset = width / 2
    y_offset = height / 2

    # get lon/lat of map edges
    lon, lat = m([(-width / 2) + x_offset, (width / 2) + x_offset], [(-height / 2) + y_offset, (height / 2) + y_offset], inverse=True)

    # plot sensors
    if sensors:
        m.plot(TOSSIT_locations[:,1] + x_offset, -TOSSIT_locations[:,0] + y_offset, markerfacecolor='aqua' if bins else 'green', linestyle="None", marker='^', markeredgecolor='blue' if bins else 'black', latlon=False, label='sensors')
    
    # for i, l in enumerate(locs_est):
    if est_latlon:
        # convert lat/lon to x/y bins so we can bin if we need to
        x, y = locs_est[:,1], locs_est[:,0]
        x, y = m(x, y)
    else:
        x, y = locs_est[:,1] + x_offset, -locs_est[:,0] + y_offset
    
    if bins:
        # bin the locs
        bins_list = [np.arange(0, height + bins, bins), np.arange(0, width + bins, bins)]
        H, x_edges, y_edges = np.histogram2d(y, x, bins=bins_list)
        H = H / np.max(H)
        # H[H == 0] = np.nan

        # make heatmap
        xx, yy = np.meshgrid(y_edges, x_edges)
        cmap = cmap = ListedColormap(['white', 'black'])
        colormesh = m.pcolormesh(xx, yy, H > 0, latlon=False, cmap=cmap)
    
    # plot points
    if points:
        m.plot(x, y, 'x', color=(colors[0][0], colors[0][1], colors[0][2]), latlon=False, label='estimate')

    # colorbar
    if bins:
        cb = m.colorbar(colormesh, location='bottom')
        tick_locs = (np.arange(2) + 0.5)/2
        cb.set_ticks(tick_locs)
        cb.set_ticklabels([f"No {name.capitalize()} Detection", f"{name.capitalize()} Detection"])

    # mark lat/lons and draw legend
    m.drawparallels(np.arange(np.floor(lat[0]), np.ceil(lat[1]), d_lat), labels=[1, 0, 0, 0], color="None")
    m.drawmeridians(np.arange(np.floor(lon[0]), np.ceil(lon[1]), d_lon), labels=[0, 0, 0, 1], color="None")
    plt.legend(loc='upper left')
    
    if return_fig:
        return fig

def one_plot_comparison(locs_est, locs_comp, comp_path=None, ax=None, buffer=6000, bins=2500, d_lat=0.1, d_lon=0.1, title=None, est_latlon=False, compare_latlon=False, points=False, sensors=False, est_name='est', comp_name='comp'):
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
    ax : Axis
        axis to put the plot on
    buffer : float
        how much to plot outside of the limits established in the config file on either side
        of the width and height
    bins : float
        dimensions of location bins
    d_lat : float
        distance between lattitudes on the y-axis ticks
    d_lon : float
        distance between longitudes on the x-axis ticks    
    title : str
        title for plot
    est_latlon : bool
        whether locs_est are provided in lat/lon (True) or X/Y (False)
    compare_latlon : bool
        whether locs_comp are provided in lat/lon (True) or X/Y (False)
    points : bool
        whether to plot the location points (True) or not (False)
    sensors : bool
        whether to plot the sensor locations (True) or not (False)
    est_name : str
        name of the type of estimate
    comp_name : str
        name of the type of comparison
    """
    # random list of color for plotting
    rng = np.random.default_rng(386729407)
    colors = [rng.uniform(0, 1, size=3) for _ in range(2)]

    # load constants
    TOSSIT_locations = np.asarray([config['TOSSIT']['TOSSIT_y'], config['TOSSIT']['TOSSIT_x']]).T
    min_x = config['scaling']['min_x']
    max_x = config['scaling']['max_x']
    min_y = config['scaling']['min_y']
    max_y = config['scaling']['max_y']

    if ax is None:
        return_fig = True
        fig, ax = plt.subplots(1,1)
    else:
        return_fig = False

    ax.set_title(title)

    # define basemap
    width = abs(config['scaling']['max_x'] - config['scaling']['min_x']) + 2*buffer
    height = abs(config['scaling']['max_y'] - config['scaling']['min_y']) + 2*buffer
    m = Basemap(width=width, height=height, projection='aeqd',
                lat_0=config['TOSSIT']['map_origin_latlon'][0], lon_0=config['TOSSIT']['map_origin_latlon'][1], resolution="f", ellps="WGS84", ax=ax)
    # fill background.
    m.drawmapboundary(fill_color='aqua' if bins is None else 'white')
    # draw coasts and fill continents.
    m.drawcoastlines(linewidth=0.8)
    m.fillcontinents(color='coral' if bins is None else 'white',lake_color='aqua' if bins is None else 'white')

    # define the x/y offsets because Basemap defines the origin in the lower left of the map
    x_offset = width / 2
    y_offset = height / 2

    # get lon/lat of map edges
    lon, lat = m([(-width / 2) + x_offset, (width / 2) + x_offset], [(-height / 2) + y_offset, (height / 2) + y_offset], inverse=True)

    # plot sensors
    if sensors:
        m.plot(TOSSIT_locations[:,1] + x_offset, -TOSSIT_locations[:,0] + y_offset, markerfacecolor='aqua', linestyle="None", marker='^', markeredgecolor='blue', latlon=False, label='sensors')

    # construct bins
    bins_list = [np.arange(0, height + bins, bins), np.arange(0, width + bins, bins)]

    # get counts for location estimates
    # for i, l in enumerate(locs_est):
    if est_latlon:
        # convert lat/lon to x/y bins so we can bin if we need to
        x, y = locs_est[:,1], locs_est[:,0]
        x, y = m(x, y)
    else:
        x, y = locs_est[:,1] + x_offset, -locs_est[:,0] + y_offset
    
    # bin the locs
    H_est, x_edges, y_edges = np.histogram2d(y, x, bins=bins_list)

    # plot points
    if points:
        m.plot(x, y, 'x', color=(colors[0][0], colors[0][1], colors[0][2]), latlon=False, label=f'{est_name} estimate')

    # get counts for comparison
    # for i, l in enumerate(locs_comp):
    if compare_latlon:
        # convert lat/lon to x/y bins so we can bin if we need to
        x, y = locs_comp[:,1], locs_comp[:,0]
        x, y = m(x, y)

        if comp_path is not None:
            x_p, y_p = comp_path[:,1], comp_path[:,0]
            x_p, y_p = m(x_p, y_p)
    else:
        x, y = locs_comp[:,1] + x_offset, -locs_comp[:,0] + y_offset
        
        if comp_path is not None:
            x_p, y_p = comp_path[:,1] + x_offset, -comp_path[:,0] + y_offset

    # bin the locs
    H_comp, x_edges, y_edges = np.histogram2d(y, x, bins=bins_list)

    # plot points
    if points:
        m.plot(x, y, 'x', color=(colors[1][0], colors[1][1], colors[1][2]), latlon=False, label=f'{comp_name} estimate')

    # identify regions of detection
    H_est = H_est > 0
    H_comp = H_comp > 0

    # heat map
    H = np.zeros(H_est.shape)

    # in common map
    r, c = np.where((H_comp == 1) & (H_est == 1))
    H[r, c] = 1

    # just est map
    r, c = np.where((H_comp == 0) & (H_est == 1))
    H[r, c] = 2

    # just comp map
    r, c = np.where((H_comp == 1) & (H_est == 0))
    H[r, c] = 3
    
    xx, yy = np.meshgrid(y_edges, x_edges)
    cmap = ListedColormap(['white', 'r', 'g', 'b'])
    colormesh = m.pcolormesh(xx, yy, H, latlon=False, cmap=cmap)

    # set up colorbar
    cb = m.colorbar(colormesh, location='bottom', pad=0.4)
    tick_locs = (np.arange(4) + 0.5)*(3)/4
    cb.set_ticks(tick_locs)
    cb.set_ticklabels(["None", "Both", est_name.capitalize(), comp_name.capitalize()])

    # mark lat/lons and draw legend
    m.drawparallels(np.arange(np.floor(lat[0]), np.ceil(lat[1]), d_lat), labels=[1, 0, 0, 0], color="None")
    m.drawmeridians(np.arange(np.floor(lon[0]), np.ceil(lon[1]), d_lon), labels=[0, 0, 0, 1], color="None")

    if comp_path is not None:
        m.plot(x_p, y_p, latlon=False, label="flight path")

    ax.legend(loc='lower left')

    if return_fig:
        return fig

def plot_spatial_density(locs_est, TOSSIT_ids=None, ax=None, buffer=6000, bins=2500, d_lat=0.1, d_lon=0.1, title=None, est_latlon=False, points=False, sensors=False, norm=False, bathym=False, GDOP_norm=False):
    """
    Plot heatmap of binned detections.

    Parameters
    ----------
    locs_est : List[np.array] or np.array, with each subarray of shape N X 2
        list of lists of estimated locations where the first column stores the Y
        coordinate and the second stores the X coordinates
    ax : Axis
        axis to put the plot on
    buffer : float
        how much to plot outside of the limits established in the config file on either side
        of the width and height
    bins : float
        dimensions of location bins
    d_lat : float
        distance between lattitudes on the y-axis ticks
    d_lon : float
        distance between longitudes on the x-axis ticks    
    title : str
        title for plot
    est_latlon : bool
        whether locs_est are provided in lat/lon (True) or X/Y (False)
    points : bool
        whether to plot the location points (True) or not (False)
    sensors : bool
        whether to plot the sensor locations (True) or not (False)
    norm : bool
        whether to normalize the heatmap (True) or not (False) by dividing by the maximum value
    """


    # load constants
    TOSSIT_locations = np.asarray([config['TOSSIT']['TOSSIT_y'], config['TOSSIT']['TOSSIT_x']]).T
    min_x = config['scaling']['min_x']
    max_x = config['scaling']['max_x']
    min_y = config['scaling']['min_y']
    max_y = config['scaling']['max_y']

    rng = np.random.default_rng(386729407)
    colors = rng.uniform(0, 1, size=3)

    if ax is None:
        return_fig = True
        fig, ax = plt.subplots(1,1)
    else:
        return_fig = False

    ax.set_title(title)

    # define basemap
    width = abs(config['scaling']['max_x'] - config['scaling']['min_x']) + 2*buffer
    height = abs(config['scaling']['max_y'] - config['scaling']['min_y']) + 2*buffer
    m = Basemap(width=width, height=height, projection='aeqd',
                lat_0=config['TOSSIT']['map_origin_latlon'][0], lon_0=config['TOSSIT']['map_origin_latlon'][1], resolution="f", ellps="WGS84", ax=ax)
    # fill background.
    m.drawmapboundary(fill_color='aqua' if bins is None else 'white')
    # draw coasts and fill continents.
    m.drawcoastlines(linewidth=1.5)
    if not bathym:
        m.fillcontinents(color='coral' if bins is None else 'white',lake_color='aqua' if bins is None else 'white', zorder=1)

    # define the x/y offsets because Basemap defines the origin in the lower left of the map
    x_offset = width / 2
    y_offset = height / 2

    # get lon/lat of map edges
    lon, lat = m([(-width / 2) + x_offset, (width / 2) + x_offset], [(-height / 2) + y_offset, (height / 2) + y_offset], inverse=True)

    # plot sensors
    if sensors:
        m.plot(TOSSIT_locations[:,1] + x_offset, -TOSSIT_locations[:,0] + y_offset, markerfacecolor='grey', linestyle="None", marker='o', markeredgecolor='black', latlon=False, label='sensors')

    # get counts for location estimates
    # for i, l in enumerate(locs_est):
    if est_latlon:
        # convert lat/lon to x/y bins so we can bin if we need to
        x, y = locs_est[:,1], locs_est[:,0]
        x, y = m(x, y)
    else:
        x, y = locs_est[:,1] + x_offset, -locs_est[:,0] + y_offset
    
    # construct bins
    bins_list = [np.arange(0, height + bins, bins), np.arange(0, width + bins, bins)]
    
    # bin the locs
    if GDOP_norm:
        TL_transform = np.concatenate((TOSSIT_locations[:,[1]] + x_offset, -TOSSIT_locations[:,[0]] + y_offset), axis=1)
        weights = calculate_GDOP_locs(x, y, TL_transform, TOSSIT_ids)
    else:
        weights = None

    H_est, x_edges, y_edges = np.histogram2d(y, x, bins=bins_list, weights=weights, density=False)

    # calculate center positions of bins
    x_centers = x_edges[1:] - (bins / 2)
    y_centers = y_edges[1:] - (bins / 2)
    
    # normalize by GDOP
    # if GDOP_norm:
    #     TL_transform = np.concatenate((TOSSIT_locations[:,[1]] + x_offset, -TOSSIT_locations[:,[0]] + y_offset), axis=1)
    #     GDOP = calculate_GDOP(m, y_centers, x_centers, TL_transform)
    #     H_est = H_est * GDOP

    # normalize
    if norm:
        H_est = H_est / np.max(H_est)

    # set 0 detection bins as nan
    H_est[H_est == 0] = float('nan')

    # plot heatmap
    xx, yy = np.meshgrid(y_edges, x_edges)
    cmap = colormaps['summer']
    cmap.set_bad("white")
    colormesh = m.pcolormesh(xx, yy, H_est - H_est_orig, latlon=False, cmap=cmap, vmin=np.nanmin(H_est))

    # set up colorbar
    cb = m.colorbar(colormesh, location='right', label="Normalized Density")

    # plot points
    if points:
        m.plot(x, y, 'x', color=(colors[0], colors[1], colors[2]), latlon=False, label='estimate')

    # mark lat/lons and draw legend
    m.drawparallels(np.arange(np.floor(lat[0]), np.ceil(lat[1]), d_lat), labels=[1, 0, 0, 0], color="None")
    m.drawmeridians(np.arange(np.floor(lon[0]), np.ceil(lon[1]), d_lon), labels=[0, 0, 0, 1], color="None")

    if bathym:
        # load bathymetry map
        Image.MAX_IMAGE_PIXELS = 729744000
        bathym = Image.open(os.path.join(config['dataset']['data_directory'], "mikesbathym.tif"))

        lat_bounds_idx = [(lat[0] - config['TOSSIT']['map_lat_limits'][0]) / config['TOSSIT']['cell_lat'], (lat[1] - config['TOSSIT']['map_lat_limits'][0]) / config['TOSSIT']['cell_lat']]
        lon_bounds_idx = [(lon[0] - config['TOSSIT']['map_lon_limits'][0]) / config['TOSSIT']['cell_lon'], (lon[1] - config['TOSSIT']['map_lon_limits'][0]) / config['TOSSIT']['cell_lon']]

        bathym = bathym.crop((round(lon_bounds_idx[0]), 
                            round(bathym.size[1] - lat_bounds_idx[1]), 
                            round(lon_bounds_idx[1]), 
                            round(bathym.size[1] - lat_bounds_idx[0])))

        bathym_np = np.array(bathym, dtype=float)
        bathym_np[(bathym_np >= 145) & (bathym_np <= 149)] = float('nan')
        lat_vec = np.linspace(lat[0], lat[1], bathym_np.shape[0])
        lon_vec = np.linspace(lon[0], lon[1], bathym_np.shape[1])

        xxx, yyy = np.meshgrid(lon_vec, lat_vec)

        cmapb = colormaps['jet']
        cmapb.set_bad("white")

        cp = m.contour(xxx, yyy, np.flipud(np.ma.array(bathym_np, mask=np.isnan(bathym_np))), latlon=True, cmap=cmapb, levels=20)
        cbb = m.colorbar(cp, location='bottom', pad=0.4, label="Depth [m]")

    ax.legend(loc='lower left')

    if return_fig:
        return fig