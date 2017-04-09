#===================================================================================================
# common utilty functions

import numpy as N

import operator
import math

from functools import reduce

import pdb

#---------------------------------------------------------------------------------------------------



#---------------------------------------------------------------------------------------------------

def product( xs ):

    return reduce( operator.mul, xs, 1 )


#---------------------------------------------------------------------------------------------------

def slices( range_list ):
    
    if range_list == []:
        return []

    if len( range_list[ 0 ] ) == 3:
        return [ slice( a, b, s ) for a, b, s in range_list ] 

    if len( range_list[ 0 ] ) == 2:
        return [ slice( a, b, None ) for a, b in range_list ]

    else:
        error = 'the slice for a tuple a:b:s must have the form (a,b,c) (or (a,b) for a:b)' 
        raise TypeError( error )


#---------------------------------------------------------------------------------------------------

def slice_count( s ):

    assert( isinstance( s , slice ) )
    i = s.step if s.step else 1
    m = s.stop - s.start
    n = 1 + math.floor( m / i )
    return n


#---------------------------------------------------------------------------------------------------

def cartesian_product( *xs ):

    if isinstance( xs[ 0 ], tuple ):
        s = slices( xs ) 
        return cartesian_product( *s )

    if isinstance( xs[ 0 ], slice ):
        pdb.set_trace()
        s = [ slice_count( x ) for x in xs ] 
        n = product( s )
        p = N.mgrid[ xs ]
        return p.reshape( ( 2, n ) ).T

    if isinstance( xs[ 0 ], list ):
        s = [ slice_count( x ) for x in xs ] 
        n = product( s )
        p = N.array( N.meshgrid( xs ) )
        return p.reshape( ( 2, n ) ).T

    raise TypeError( 'slice or list required' ) 


#---------------------------------------------------------------------------------------------------




#---------------------------------------------------------------------------------------------------

import unittest

class Tests( unittest.TestCase ):

    def test_cartesian_product_of_empty_sets_is_empty( self ):
        empty = N.array( [] ).reshape( ( 0, 2 ) )
        result = cartesian_product( [], [] )
        self.assertTrue( N.array_equal( result, empty ) )

    def test_cartesian_product_of_two_sets( self ):
        pdb.set_trace()
        computed = cartesian_product( ( 0, 2, 2 ), ( 1, 4, 1 ) )
        expected = N.array( [
            [ 0, 1 ],
            [ 0, 2 ],
            [ 0, 3 ], 
            [ 2, 1 ],
            [ 2, 2 ],
            [ 2, 3 ] ] )
        self.assertTrue( N.array_equal( computed, expected ) )
        

#---------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    unittest.main()
    
#---------------------------------------------------------------------------------------------------
