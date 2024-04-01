import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.geometry import MultiPoint, Point, Polygon

from whale_gunshot_localization import config

class PointsInPoly:
    """
    Generate random points within the convex hull of 
    a network of sensors via rejection sampling.

    ...

    Attributes
    ----------
    polygon : shapely.geometry.Polygon
        polygon object
    rng : np.random.default_rng
        random generator object
    """

    def __init__(self, TOSSIT_locations, rng):
        """
        Construct attributes.

        Parameters
        ----------
        TOSSIT_locations : np.ndarray
            sensor coorindinates
        rng : np.random.default_rng
            random generator object
        """
        
        TOSSIT_locations = np.asarray([TOSSIT_locations[:,1], -TOSSIT_locations[:,0]]).T
        mpt = MultiPoint(TOSSIT_locations)
        self.polygon = Polygon(mpt.convex_hull)
        self.rng = rng

    def generate(self, n):
        """
        Generate random points in polygon

        Parameters
        ----------
        n : int
            number of points to generate

        Returns
        -------
          : np.ndarray
            randomly generated points
        """

        minx, miny, maxx, maxy = self.polygon.bounds
        points = []
        while len(points) < n:
            pnt = Point(self.rng.uniform(minx, maxx), self.rng.uniform(miny, maxy))
            if self.polygon.contains(pnt):
                points.append([-pnt.xy[1][0], pnt.xy[0][0]])
        return np.asarray(points)

def generate_measurements(num_sources, rng, var=10, num_delete=0, in_sensors=False, TOSSIT_locations=None):
    """
    Generate some synthetic measurements to test with data association/localization algoritms.
    
    Parameters
    ----------
    num_sources : int
        maximum number of sources
    rng : numpy.random._generator.Generator
        RNG object
    var : float
        variance of Gaussian noise added to range measurements
    num_delete : int
        maximum number of measurements to hide/delete at each source
    in_sensors : bool
        whether or not to limit sampled locations to be in the rectangle that the network inscribes
    TOSSIT_locations : np.ndarray
        N X 2 matrix of sensor locations
        
    Returns
    -------
    range_measurements : List[array-like]
        each sublist contains range measurements and is associated with a particular TOSSIT
    source_associations : List[array-like]
        same shape as range_measurements. each sublist contains numbers which associate the
        range_measurement in the corresponding spot in the data structure with a source.
    TOSSIT_association : List[array-like]
        same shape as range_measurements. each sublist contains numbers which associate the
        range_measurement in the corresponding spot in the data structure with a TOSSIT.    
    source_locs : array-like[array-like]
        matrix of generated source locations
    """ 

    # check inputs
    assert num_sources > 0, "number of sources must be non-negative"
    assert num_delete >= 0, "max signals to delete at each sensor must be non-negative"
    
    # get TOSSIT locations and extreme coordinate values
    TOSSIT_locations = TOSSIT_locations if TOSSIT_locations is not None else np.asarray([config['TOSSIT']['TOSSIT_y'], config['TOSSIT']['TOSSIT_x']]).T
    
    # choose number of sources and generate source locations
    if in_sensors:
        G = PointsInPoly(TOSSIT_locations, rng)
        source_locs = G.generate(num_sources)
    else:
        min_x = config['scaling']['min_x']
        max_x = config['scaling']['max_x']
        min_y = config['scaling']['min_y']
        max_y = config['scaling']['max_y']
        source_locs = np.concatenate((rng.uniform(min_y, max_y, size=(num_sources,1)), rng.uniform(min_x, max_x, size=(num_sources,1))), axis=1)
    
    # generate range measurements and log associations
    range_measurements, TOSSIT_associations, del_list, source_associations = [], [], [], []
    for t in range(TOSSIT_locations.shape[0]):
        
        # calculate range measurements from all sources to TOSSIT t, adding Gaussian noise
        r = np.abs(np.linalg.norm(source_locs - TOSSIT_locations[t,:], axis=1) + \
            rng.normal(loc=0, scale=np.sqrt(var), size=num_sources))
        range_measurements.append(r)
        
        # generate arrays for the TOSSIT associations of the measuremnts.
        TOSSIT_associations.append(np.ones((num_sources,), dtype=int) * t)
        
        # generate arrays of source associations for the measurements
        source_associations.append(np.arange(num_sources))
    
    # delete from each source
    # if num_delete:
    #     num_delete = rng.choice(num_delete)
    #     for s in range(num_sources): 
    #         distances = np.sqrt(((TOSSIT_locations - source_locs[s,:]) ** 2).sum(axis=1))
    #         to_delete = np.argsort(distances)[::-1][:num_delete]
    #         for d in to_delete:
    #             idx = np.argwhere(source_associations[d] == s)
    #             range_measurements[d] = np.delete(range_measurements[d], idx)
    #             source_associations[d] = np.delete(source_associations[d], idx)
    #             TOSSIT_associations[d] = np.delete(TOSSIT_associations[d], idx)
    if num_delete:
        for i in range(num_delete):
            while True:
                try:
                    sen = rng.choice(TOSSIT_locations.shape[0])
                    idx = rng.choice(len(range_measurements[sen]))
                except:
                    continue
                range_measurements[sen] = np.delete(range_measurements[sen], idx)
                source_associations[sen] = np.delete(source_associations[sen], idx)
                TOSSIT_associations[sen] = np.delete(TOSSIT_associations[sen], idx)
                break

                
    # make sure smallest association value is 0
    bias = min([a[0] for a in source_associations if len(a)])
    if bias > 0:
        for i, a in enumerate(source_associations):
            source_associations[i] = a - bias
    
    return range_measurements, source_associations, TOSSIT_associations, source_locs

def generate_simple_paths(source_params, var, TOSSIT_locations, rng, loc0=None, bearing0=None, beamwidth=None, n_del=0):

    assert (n_del == 0) or (TOSSIT_locations.shape[0] - n_del >= 3), "cannot make source impossible to localize"

    # generate initial locations for sources
    if loc0 is None or bearing0 is None:
        PIP = PointsInPoly(TOSSIT_locations=TOSSIT_locations, rng=rng)
        curr_pts = PIP.generate(len(source_params))
        headings = rng.uniform(low=0, high=360, size=(len(curr_pts)))
    else:
        curr_pts = loc0
        headings = bearing0

    range_measurements_list = []
    source_associations_list = []
    TOSSIT_associations_list = []
    source_locs_list = []
    source_ids_list = []
    for ts in range(1, np.max(source_params)+1):
        
        # storage lists
        range_measurements = [] 
        TOSSIT_associations = [] 
        source_associations = []
        source_ids = []
        
        for idx, interval in enumerate(source_params):
            if 1 in interval and interval[0] <= ts <= interval[1]:
                source_ids.append(idx)
            elif interval[0] < ts <= interval[1]:
                source_ids.append(idx)
        
        # get present source locs
        source_locs = curr_pts[source_ids,:]
                
        # get number of sources
        num_sources = len(source_ids)

        for t in range(TOSSIT_locations.shape[0]):
            # calculate range measurements from all sources to TOSSIT t, adding Gaussian noise
            r = np.abs(np.linalg.norm(source_locs - TOSSIT_locations[t,:], axis=1) + \
                rng.normal(loc=0, scale=np.sqrt(var), size=num_sources))
            
            range_measurements.append(r)

            # generate arrays for the TOSSIT associations of the measuremnts.
            TOSSIT_associations.append(np.ones((num_sources,), dtype=int) * t)

            # generate arrays of source associations for the measurements
            source_associations.append(np.arange(num_sources))

        # optional delete measurements
        # if n_del > 0:
        #     for s in range(num_sources):
        #         for _ in range(n_del):
        #             while True:
        #                 tidx = np.random.choice(TOSSIT_locations.shape[0])
        #                 tassocs = source_associations[tidx]
        #                 if s in tassocs:
        #                     midx = np.where(tassocs == s)[0]
        #                     range_measurements[tidx] = np.delete(range_measurements[tidx], midx)
        #                     source_associations[tidx] = np.delete(source_associations[tidx], midx)
        #                     TOSSIT_associations[tidx] = np.delete(TOSSIT_associations[tidx], midx)
        #                     break
        if n_del > 0:
            for s in range(num_sources):
                tidxs = np.random.choice(TOSSIT_locations.shape[0], replace=False, size=(n_del,))
                for tidx in tidxs:
                    tassocs = source_associations[tidx]
                    midx = np.where(tassocs == s)[0]
                    range_measurements[tidx] = np.delete(range_measurements[tidx], midx)
                    source_associations[tidx] = np.delete(source_associations[tidx], midx)
                    TOSSIT_associations[tidx] = np.delete(TOSSIT_associations[tidx], midx)

        range_measurements_list.append(range_measurements)
        source_associations_list.append(source_associations)
        TOSSIT_associations_list.append(TOSSIT_associations)
        source_locs_list.append(source_locs)
        source_ids_list.append(source_ids)

        # move present sources
        if beamwidth is not None:
            headings += rng.uniform(low=-beamwidth, high=beamwidth, size=len(headings))
        unit_vecs = np.asarray([-np.sin(np.radians(headings)), np.cos(np.radians(headings))]).T
        curr_pts[source_ids,:] += 70 * unit_vecs[source_ids,:]
    
    return range_measurements_list, source_associations_list, TOSSIT_associations_list, source_locs_list, source_ids_list
