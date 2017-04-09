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


#---------------------------------------------------------------------------------------------------
