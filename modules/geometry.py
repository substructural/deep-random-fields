#---------------------------------------------------------------------------------------------------

import numpy as N

#---------------------------------------------------------------------------------------------------


def voxel( *coordinates ) :

    if len( coordinates ) == 1 and isinstance( coordinates[ 0 ], N.ndarray ) :
        return coordinates[ 0 ].astype( N.int16 )

    else :
        return N.array( coordinates, dtype=N.int16 )



def cuboid( minima, maxima ) :

    if isinstance( minima, tuple ):
        return N.array( [ minima, maxima ] )

    if isinstance( minima, N.ndarray ):
        return N.array( [ minima, maxima ] )


def in_bounds( point, bounds ):

    minimum, maximum = bounds
    dimensions = range( 0, len( point ) )
    for d in dimensions:
        if ( point[ d ] < minimum[ d ] ) or ( maximum[ d ] < point[ d ] ):
            return False
    return True


def mask( outer, minimum, maximum, value_in_mask = 1 ):

    dimensions = len( outer )
    mask = N.zeros( tuple( outer ) ).astype( 'int64' )

    if dimensions == 2:
        mask[ minimum[ 0 ] : maximum[ 0 ] + 1,
              minimum[ 1 ] : maximum[ 1 ] + 1 ] = value_in_mask

        return mask

    if dimensions == 3:
        mask[ minimum[ 0 ] : maximum[ 0 ] + 1,
              minimum[ 1 ] : maximum[ 1 ] + 1,
              minimum[ 2 ] : maximum[ 2 ] + 1 ] = value_in_mask

        return mask

    else:
        raise Exception( f'only 2D and 3D masks are supported (dimensions = {dimensions})' )
 

#---------------------------------------------------------------------------------------------------
