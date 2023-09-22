from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import numpy as np
import pyproj as proj

from whale_gunshot_localization import config

def plot_localization_binary(locs_est, locs_comp=None, buffer=6000, title_est=None, title_comp=None, dates=None, bins=None, d_lat=0.1, d_lon=0.1, est_latlon=False, compare_latlon=False, points=False, sensors=False, one_plot=False, est_name='est', comp_name='comp', save=None): 
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
    title_est : str
        title for the estimated locations
    title_comp : str
        title for the locations for comparison
    dates : List[datetime.datetime] or datetime.datatime
        list of dates associated with each sublist of location estiamtes
        in locs_est
    d_lat : float
        distance between lattitudes on the y-axis ticks
    d_lon : float
        distance between longitudes on the x-axis ticks
    est_latlon : bool
        whether locs_est are provided in lat/lon (True) or X/Y (False)
    compare_latlon : bool
        whether locs_comp are provided in lat/lon (True) or X/Y (False)
    points : bool
        whether to plot the location points (True) or not (False)
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
    
    # assert the inputs match properly and set the number of subplots
    if locs_comp:
        assert len(locs_comp) == len(locs_est) == len(dates), "locs_est, locs_comp, and dates lists must have a one-to-one correspondence"
        n_subplots = 2
    else:
        assert len(locs_est) == len(dates), "locs_est and dates lists must have a one-to-one correspondence"
        n_subplots = 1

    # figure
    fig = plt.figure()

    # random list of color for plotting
    rng = np.random.default_rng(386729407)
    colors = [rng.uniform(0, 1, size=3) for _ in range(len(locs_est) + 1)]

    # load constants
    TOSSIT_locations = np.asarray([config['TOSSIT']['TOSSIT_y'], config['TOSSIT']['TOSSIT_x']]).T
    min_x = config['scaling']['min_x']
    max_x = config['scaling']['max_x']
    min_y = config['scaling']['min_y']
    max_y = config['scaling']['max_y']

    # define projection objectalgorithm_sandbox
    # pargs = proj.Proj(proj="aeqd", lat_0=config['TOSSIT']['map_origin_latlon'][0], lon_0=config['TOSSIT']['map_origin_latlon'][1], datum="WGS84", units="m")

    if one_plot:
        assert bins is not None, "bins argument required when using one_plot=True"
        assert locs_comp is not None, "locs_comp argument is required when using one_plot=True" 

        ax = fig.add_subplot(111)
        ax.set_title(title_est)

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

        # construct bins
        bins_list = [np.arange(0, height + bins, bins), np.arange(0, width + bins, bins)]

        # get counts for location estimates
        for i, l in enumerate(locs_est):
            if est_latlon:
                # convert lat/lon to x/y bins so we can bin if we need to
                x, y = l[:,1], l[:,0]
                x, y = m(x, y)
            else:
                x, y = l[:,1] + x_offset, -l[:,0] + y_offset
            
            # bin the locs
            H_est, x_edges, y_edges = np.histogram2d(y, x, bins=bins_list)

            # plot points
            if points:
                m.plot(x, y, 'x', color=(colors[i][0], colors[i][1], colors[i][2]), latlon=est_latlon, label=f'{est_name} estimate ({dates[i]})' if dates[0] else 'estimate')


        # get counts for comparison
        for i, l in enumerate(locs_comp):
            if compare_latlon:
                # convert lat/lon to x/y bins so we can bin if we need to
                x, y = l[:,1], l[:,0]
                x, y = m(x, y)
            else:
                x, y = l[:,1] + x_offset, -l[:,0] + y_offset

            # bin the locs
            H_comp, x_edges, y_edges = np.histogram2d(y, x, bins=bins_list)

            # plot points
            if points:
                m.plot(x, y, 'x', color=(colors[i + 1][0], colors[i + 1][1], colors[i + 1][2]), latlon=est_latlon, label=f'{comp_name} estimate ({dates[i]})' if dates[0] else 'estimate')

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
        colormesh = m.pcolormesh(xx, yy, H, latlon=est_latlon, cmap=cmap)

        # set up colorbar
        cb = m.colorbar(colormesh, location='bottom')
        tick_locs = (np.arange(4) + 0.5)*(3)/4
        cb.set_ticks(tick_locs)
        cb.set_ticklabels(["None", "Both", est_name.capitalize(), comp_name.capitalize()])

        ax.legend(loc='upper left')

    else:
        # instantiate subplots
        ax = fig.add_subplot(111 if n_subplots == 1 else 211)
        ax.set_title(title_est)

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
        
        for i, l in enumerate(locs_est):
            if est_latlon:
                # convert lat/lon to x/y bins so we can bin if we need to
                x, y = l[:,1], l[:,0]
                x, y = m(x, y)
            else:
                x, y = l[:,1] + x_offset, -l[:,0] + y_offset
            
            if bins:
                # bin the locs
                bins_list = [np.arange(0, height + bins, bins), np.arange(0, width + bins, bins)]
                H, x_edges, y_edges = np.histogram2d(y, x, bins=bins_list)
                H = H / np.max(H)
                # H[H == 0] = np.nan

                # make heatmap
                xx, yy = np.meshgrid(y_edges, x_edges)
                cmap = plt.get_cmap('binary')
                # cmap.set_bad("white")
                colormesh = m.pcolormesh(xx, yy, H > 0, latlon=est_latlon, cmap=cmap)
            
            # plot points
            if points:
                m.plot(x, y, 'x', color=(colors[i][0], colors[i][1], colors[i][2]), latlon=est_latlon, label=f'{est_name} estimate ({dates[i]})' if dates[0] else 'estimate')

        # colorbar
        # if bins:
        #     cb = m.colorbar(colormesh, location='right', label="Normalized Detections")

        # mark lat/lons and draw legend
        m.drawparallels(np.arange(np.floor(lat[0]), np.ceil(lat[1]), d_lat), labels=[1, 0, 0, 0], color="None")
        m.drawmeridians(np.arange(np.floor(lon[0]), np.ceil(lon[1]), d_lon), labels=[0, 0, 0, 1], color="None")
        plt.legend(loc='upper left')

        if locs_comp:
            ax = fig.add_subplot(212)
            ax.set_title(title_comp)

            # define basemap
            m = Basemap(width=width, height=height, projection='aeqd',
                        lat_0=config['TOSSIT']['map_origin_latlon'][0], lon_0=config['TOSSIT']['map_origin_latlon'][1], resolution="f")
            # fill background.
            m.drawmapboundary(fill_color='aqua' if bins is None else 'white')
            # draw coasts and fill continents.
            m.drawcoastlines(linewidth=0.8)
            m.fillcontinents(color='coral' if bins is None else 'white',lake_color='aqua' if bins is None else 'white')

            # plot sensors
            if sensors:
                m.plot(TOSSIT_locations[:,1] + x_offset, -TOSSIT_locations[:,0] + y_offset, markerfacecolor='aqua' if bins else 'green', linestyle="None", marker='^', markeredgecolor='blue' if bins else 'black', latlon=False, label='sensors')

            for i, l in enumerate(locs_comp):
                if compare_latlon:
                    # convert lat/lon to x/y bins so we can bin if we need to
                    x, y = l[:,1], l[:,0]
                    x, y = m(x, y)
                else:
                    x, y = l[:,1] + x_offset, -l[:,0] + y_offset

                if bins:
                    # bin the locs
                    bins_list = [np.arange(0, height + bins, bins), np.arange(0, width + bins, bins)]                
                    H, x_edges, y_edges = np.histogram2d(y, x, bins=bins_list)
                    H = H / np.max(H)
                    # H[H == 0] = np.nan
                    
                    # make heatmap
                    xx, yy = np.meshgrid(y_edges, x_edges)
                    cmap = plt.get_cmap('binary')
                    # cmap.set_bad("white")
                    colormesh = m.pcolormesh(xx, yy, H > 0, latlon=False, cmap=cmap)
            
            # plot points
            if points:
                m.plot(x, y, 'x', color=(colors[i][0], colors[i][1], colors[i][2]), latlon=False, label=f'{comp_name} estimate ({dates[i]})' if dates[0] else 'estimate')

            # colorbar
            # if bins:
            #     cb = m.colorbar(colormesh, location='right', label="Normalized Detections")

            # mark lat/lons and draw legend
            m.drawparallels(np.arange(np.floor(lat[0]), np.ceil(lat[1]), d_lat), labels=[1, 0, 0, 0], color="None")
            m.drawmeridians(np.arange(np.floor(lon[0]), np.ceil(lon[1]), d_lon), labels=[0, 0, 0, 1], color="None")
            plt.legend(loc='lower left')
    
    if save is not None:
        plt.savefig(save)
    
    return fig