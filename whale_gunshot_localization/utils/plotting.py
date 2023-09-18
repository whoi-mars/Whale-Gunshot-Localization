from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt

import numpy as np
import pyproj as proj
# import pygmt

from whale_gunshot_localization import config

def plot_localization(locs_est, buffer=6000, title=None, dates=None, save=None, legend_transparency=0, bins=None): 
    """
    Plot estimated source locations.

    Parameters
    ----------
    locs_est : List[np.array], with each subarray of shape N X 2
        list of lists of estimated locations where the first column stores the Y
        coordinate and the second stores the X coordinates
    buffer : float
        how much to plot outside of the limits established in the config file
    title : str
        plot title
    dates : List[datetime.datetime]
        list of dates associated with each sublist of location estiamtes
        in locs_est
    save : str
        path at which to save the plot if desired
    """

    # format inputs
    if not isinstance(locs_est, list):
        locs_est = [locs_est]
    if not isinstance(dates, list):
        dates = [dates]
    assert len(locs_est) == len(dates), "locs_est and dates lists must have a one-to-one correspondence"

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
    
    # get lon/lat bounds for the map
    # lon, lat = pargs([min_x-buffer, max_x+buffer], [min_y-buffer, max_y+buffer], inverse=True)
    # region = [*lon, *lat]

    # convert sensor locs to lat/lon
    lon_TOSSIT, lat_TOSSIT = pargs(TOSSIT_locations[:,1], -TOSSIT_locations[:,0], inverse=True)

    # define basemap
    width = abs(config['scaling']['max_x'] - config['scaling']['min_x']) + 2*buffer
    height = abs(config['scaling']['max_y'] - config['scaling']['min_y']) + 2*buffer
    m = Basemap(width=width, height=height, projection='aeqd',
                lat_0=41.9108, lon_0=-70.4292, resolution="f")
    lon, lat = pargs([-width / 2, width / 2], [-height / 2, height / 2], inverse=True)

    # fill background.
    m.drawmapboundary(fill_color='white')
    # draw coasts and fill continents.
    m.drawcoastlines(linewidth=0.5)
    m.fillcontinents(color='white',lake_color='white')


    m.plot(lon_TOSSIT, lat_TOSSIT, 'g^', markeredgecolor='black', latlon=True, label='sensors')
    
    for i, l in enumerate(locs_est):
        lon_est, lat_est = pargs(l[:,1], -l[:,0], inverse=True)
        if bins:
            H, x_edges, y_edges = np.histogram2d(lat_est, lon_est, bins=bins)
            xx, yy = np.meshgrid(y_edges, x_edges)
            colormesh = m.pcolormesh(xx, yy, H, latlon=True, cmap='Greens')
        m.plot(lon_est, lat_est, 'x', color=(colors[i][0], colors[i][1], colors[i][2]), latlon=True, label=f'estimate ({dates[i]})' if dates[0] else 'estimate')

    cb = m.colorbar(colormesh, location='right', label="")

    m.drawparallels(np.arange(np.floor(lat[0]), np.ceil(lat[1]), 0.2), labels=[1, 0, 0, 0], color="None")
    m.drawmeridians(np.arange(np.floor(lon[0]), np.ceil(lon[1]), 0.1), labels=[0, 0, 0, 1], color="None")
    plt.legend(loc='upper left')
    plt.savefig(save)
    plt.close()

    # fig = pygmt.Figure()
    # fig.basemap(region=region, projection="Cyl_stere10c", frame=True)
    # fig.coast(land="black", water="skyblue4")
    # fig.plot(x=lon_TOSSIT, y=lat_TOSSIT, style="t0.3c", fill="green", pen="black", label="sensors")
    # for i, l in enumerate(locs_est):
    #     lon_est, lat_est = pargs(l[:,1], -l[:,0], inverse=True)
    #     if bins:
    #         H, x_edges, y_edges = np.histogram2d(lon, lat, bins=bins)
    #         print(np.diff(x_edges), np.diff(y_edges))
    #         fig.histogram(H, series=bins, fill=True)
    #     else:
    #         fig.plot(x=lon_est, y=lat_est, style="x0.3c", pen=f"1p,{colors[i][0]}/{colors[i][1]}/{colors[i][2]}", label=f'estimate ({dates[i]})' if dates[0] else 'estimate')
    
    # fig.legend(transparency=legend_transparency)

    # if save is not None:
    #     fig.savefig(save)