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


def mask( outer, inner_0, inner_n, value_in_mask = 1 ):

    dimensions = len( outer )
    bounds = cuboid( inner_0, inner_n )

    if dimensions == 2:
        return N.array(
            [ [ ( value_in_mask if in_bounds( (y, x), bounds ) else 0 )
                for x in range( outer[ 1 ] ) ]
              for y in range( outer[ 0 ] ) ] )

    if dimensions == 3:
        return N.array(
            [ [ [ ( value_in_mask if in_bounds( (z, y, x), bounds ) else 0 )
                  for x in range( outer[ 2 ] ) ]
                for y in range( outer[ 1 ] ) ] 
              for z in range( outer[ 0 ] ) ] )

    else:
        raise Exception( f'only 2D and 3D masks are supported (dimensions = {dimensions})' )


#---------------------------------------------------------------------------------------------------
