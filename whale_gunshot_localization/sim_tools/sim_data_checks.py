import numpy as np

def is_in_bay(source_locs, bathym, map_origin, dx, dy):
    """
    Check if source locations are in CCB.
    
    Parameters
    ----------
    source_locs : array-like[array-like]
        matrix of generated source locations'
    bathym : PIL.Image
        geotiff of the bathymetry
    map_origin : array-like of shape 1 X 2
        pixels coordinates of the origin of the bathymetry map
    dx : float
        approximate change in meters when going in the X direction
    dy : float
        approximate change in meters when going in the Y direction
    
    Returns
    -------
    : bool
        whether all source_locs are in water (True) or not (False)
    """

    for loc in source_locs:
        # line cutting off locations on the open-ocean side of Provincetown
        line1 = loc[0] - 0.88434446716*loc[1] < -38672.47
        # line cutting off locations on the west side of the Cape Cod Canal
        line2 = loc[0] - 0.84448322668*loc[1] > 19385.3 
        pixel = bathym.getpixel((round(map_origin[1] + loc[1]/dx), round(map_origin[0] + loc[0]/dy)))
        if pixel == 147 or line1 or line2:
            return False
    return True