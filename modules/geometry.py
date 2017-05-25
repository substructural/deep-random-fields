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

#---------------------------------------------------------------------------------------------------
