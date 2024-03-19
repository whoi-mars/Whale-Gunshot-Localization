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

def generate_simple_paths(source_params, var, TOSSIT_locations, rng):

    # generate initial locations for sources
    PIP = PointsInPoly(TOSSIT_locations=TOSSIT_locations, rng=rng)
    curr_pts = PIP.generate(len(source_params))
    headings = rng.uniform(low=0, high=360, size=(len(curr_pts)))

    range_measurements_list = []
    source_associations_list = []
    TOSSIT_associations_list = []
    source_locs_list = []
    source_ids_list = []
    for ts in range(1, np.max(source_params)):
        
        # storage lists
        range_measurements = [] 
        TOSSIT_associations = [] 
        source_associations = []
        source_ids = []
        
        for idx, interval in enumerate(source_params):
            if interval[0] <= ts <= interval[1]:
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

        range_measurements_list.append(range_measurements)
        source_associations_list.append(source_associations)
        TOSSIT_associations_list.append(TOSSIT_associations)
        source_locs_list.append(source_locs)
        source_ids_list.append(source_ids)

        # move present sources
        unit_vecs = np.asarray([-np.sin(np.radians(headings)), np.cos(np.radians(headings))])
        curr_pts[source_ids,:] += 100 * unit_vecs[source_ids,:]
    
    return range_measurements_list, source_associations_list, TOSSIT_associations_list, source_locs_list, source_ids_list

def generate_trajectory(num_points, beam_width=20, measurement_stats=(0, 500000), timing_stats=(1, 0.5), repetition_time=0.5, num_repetitions=1, whale_speed=1.3, chunk_size=2, max_channel_offset=40, rng=None, in_sensors=False):
    """
    Generate a simulated trajectory/path of source locations along with time-stamped measurements
    
    Paramters
    ---------
    rng : numpy.random._generator.Generator
        RNG object
    num_points : int
        number of points in the path
    beam_width : float
        arc within which we generate the next source point
    timeing_stats : Tuple[float, float]
        mean/std of call generation (in minutes)
    repetition_time : float
        time between repetitions when > 1
    repetitions : int
        number of repetitions
    whale_speed : float
        speed of simulated whale in km/hr
    chunk_size : float
        chunk of data to analyze at once (in minutes)
    in_sensors : bool
        whether or not to generate soure locations only in the sensor network

    Returns
    -------
    source_locs : array-like, of shape N X 2
        locations of sources in trajectory/path
    measurements_list : List[List[array-like]]
        list of measurements collected at each source location in the path
        on all sensors
    """
    
    # check inputs
    assert num_points > 0, "number of sources must be non-negative"

    if rng is None:
        rng = np.random
    
    # convert whale speed to m / minute
    whale_speed *= (1000 / 60)
    # convert timing_stats to seconds
    timing_stats = tuple([i*60 for i in timing_stats])
    # convert chunk size to seconds
    chunk_size *= 60
    
    # get TOSSIT locations and extreme coordinate values
    TOSSIT_locations = np.asarray([config['TOSSIT']['TOSSIT_y'], config['TOSSIT']['TOSSIT_x']]).T

    if in_sensors:
        min_x = np.min(TOSSIT_locations[:,1])
        max_x = np.max(TOSSIT_locations[:,1])
        min_y = np.min(TOSSIT_locations[:,0])
        max_y = np.max(TOSSIT_locations[:,0])
    else:
        min_x = config['scaling']['min_x']
        max_x = config['scaling']['max_x']
        min_y = config['scaling']['min_y']
        max_y = config['scaling']['max_y']
    
    # choose number of sources and generate source locations
    source_locs = np.concatenate((rng.uniform(min_y, max_y, size=(1,1)), rng.uniform(min_x, max_x, size=(1,1))), axis=1)
    time_stamps = np.arange(0, num_repetitions * repetition_time, repetition_time)
    heading = rng.uniform(0, 359, size=1)
    
    curr_loc = source_locs[[0]]
    measurements = [np.asarray([]) for _ in range(TOSSIT_locations.shape[0])]
    source_associations = [np.asarray([]) for _ in range(TOSSIT_locations.shape[0])]
    TOSSIT_associations = [np.asarray([]) for _ in range(TOSSIT_locations.shape[0])]

    s_counter = 0
    while True:
        
        # get measurements
        for t in range(TOSSIT_locations.shape[0]):
            # calculate range measurements from all sources to TOSSIT t, adding Gaussian noise
            r = np.linalg.norm(curr_loc - TOSSIT_locations[t,:], axis=1) + \
                rng.normal(loc=measurement_stats[0], scale=np.sqrt(measurement_stats[1]), size=num_repetitions)
            measurements[t] = np.append(measurements[t], r)
            source_associations[t] = np.append(source_associations[t], s_counter)
            TOSSIT_associations[t] = np.append(TOSSIT_associations[t], t)
        s_counter += 1

        if source_locs.shape[0] == num_points:
            break
        
        # time delta to next source
        new_time_delta = rng.normal(loc=timing_stats[0], scale=np.sqrt(timing_stats[1]))
        
        # timestamp(s) for current source location
        time_stamps = np.append(time_stamps, [time_stamps[-1] + new_time_delta + repetition_time*i for i in range(num_repetitions)])

        # calculate magnitude of transition vector
        vec_mag = whale_speed * new_time_delta
    
        # transition vector to get from current location to the next one
        transition_vector = (vec_mag * np.asarray([-np.sin(np.radians(heading)), np.cos(np.radians(heading))])).T
        
        # get new location and save
        curr_loc += transition_vector
        source_locs = np.concatenate((source_locs, curr_loc), axis=0)
        
        # get new heading
        heading += rng.uniform(-beam_width / 2, beam_width / 2)
    
    # create offset timestamps for each sensor
    time_stamps = time_stamps[np.newaxis,:]
    for t in range(TOSSIT_locations.shape[0]):
        offset = rng.uniform(low=0, high=max_channel_offset)
        time_stamps = np.concatenate((time_stamps, time_stamps[[0],:] + offset), axis=0)    
    
    # split measurements
    measurements_list = []
    source_assocaitions_list = []
    TOSSIT_associations_list = []
    pointer = 0
    while pointer <= time_stamps[:,-1].max():
        
        new_measurements = []
        new_source_associations = []
        new_TOSSIT_associations = []
        for t in range(TOSSIT_locations.shape[0]):
        
            # get measurements indices of time chunk
            idx = np.where((time_stamps[t] >= pointer) & (time_stamps[t] < pointer + chunk_size))[0]

            # save in list
            new_measurements.append(measurements[t][idx])
            new_source_associations.append(source_associations[t][idx])
            new_TOSSIT_associations.append(TOSSIT_associations[t][idx])

        # append location measurements to overall list        
        measurements_list.append(new_measurements)
        source_assocaitions_list.append(new_source_associations)
        TOSSIT_associations_list.append(new_TOSSIT_associations)
        
        # iterate pointer
        pointer += chunk_size
                        
    return source_locs, measurements_list, source_assocaitions_list, TOSSIT_associations_list