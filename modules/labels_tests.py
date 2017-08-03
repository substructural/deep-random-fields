#===================================================================================================
# labels module tests


import labels

import numpy
import unittest
import pdb


#---------------------------------------------------------------------------------------------------


class TestData :


    @staticmethod
    def index_volume() :

        return numpy.arange( 64 ).reshape( ( 4, 4, 4 ) )


    @staticmethod
    def index_patches() :

        return numpy.array( [

            [ [ [  0,  1 ],
                [  4,  5 ] ],
              [ [ 16, 17 ],
                [ 20, 21 ] ] ],

            [ [ [  2,  3 ],
                [  6,  7 ] ],
              [ [ 18, 19 ],
                [ 22, 23 ] ] ],

            [ [ [  8,  9 ],
                [ 12, 13 ] ],
              [ [ 24, 25 ],
                [ 28, 29 ] ] ],

            [ [ [ 10, 11 ],
                [ 14, 15 ] ],
              [ [ 26, 27 ],
                [ 30, 31 ] ] ],

            [ [ [ 32, 33 ],
                [ 36, 37 ] ],
              [ [ 48, 49 ],
                [ 52, 53 ] ] ],

            [ [ [ 34, 35 ],
                [ 38, 39 ] ],
              [ [ 50, 51 ],
                [ 54, 55 ] ] ],

            [ [ [ 40, 41 ],
                [ 44, 45 ] ],
              [ [ 56, 57 ],
                [ 60, 61 ] ] ],

            [ [ [ 42, 43 ],
                [ 46, 47 ] ],
              [ [ 58, 59 ],
                [ 62, 63 ] ] ]

            ] )


    @staticmethod
    def disjoint_distribution_patches() :

        indices = TestData.index_patches()
        distribution = numpy.zeros( ( 8, 2, 2, 2, 64 ) )
        for p in range( 0, 8 ) :
            for z in range( 0, 2 ) :
                for y in range( 0, 2 ) :
                    for x in range( 0, 2 ) :
                        d = indices[ p, z, y, x ]
                        distribution[ p, z, y, x, d ] = 1.0
        return distribution


    @staticmethod
    def overlapping_distribution_patches() :

        distribution = numpy.zeros( ( 27, 2, 2, 2, 64 ) )
        for k in range( 0, 3 ) :
            for j in range( 0, 3 ) :
                for i in range( 0, 3 ) :
                    for z in range( 0, 2 ) :
                        for y in range( 0, 2 ) :
                            for x in range( 0, 2 ) :
                                d = ( ( k + z ) * 16 ) + ( ( j + y ) * 4 ) + ( i + x )
                                p = k * 9 + j * 3 + i
                                distribution[ p, z, y, x, d ] = 1.0
        return distribution


    @staticmethod
    def distribution_volume() :

        volume = numpy.zeros( ( 4, 4, 4, 64 ) )
        for z in range( 0, 4 ) :
            for y in range( 0, 4 ) :
                for x in range( 0, 4 ) :
                    i = z*16 + y*4 + x
                    volume[ z, y, x, i ] = 1.0
        return volume



#---------------------------------------------------------------------------------------------------

class DenseLabelTests( unittest.TestCase ) :


    def test_dense_patch_indices_to_dense_patch_distribution( self ) :

        indices = TestData.index_patches()
        expected_distribution = TestData.disjoint_distribution_patches()
        computed_distribution = labels.dense_patch_indices_to_dense_patch_distribution( indices, 64 )

        self.assertTrue( numpy.array_equal( computed_distribution, expected_distribution ) )


    def test_dense_patch_distribution_to_dense_volume_distribution( self ) :

        patches = TestData.disjoint_distribution_patches()
        volume_shape = TestData.index_volume().shape
        margin = 0
        expected_volume = TestData.distribution_volume()
        computed_volume = labels.dense_patch_distribution_to_dense_volume_distribution(
            patches,
            volume_shape,
            margin )

        self.assertTrue( numpy.array_equal( computed_volume, expected_volume ) )


    def test_dense_volume_distribution_to_dense_volume_indices( self ) :

        distribution = TestData.distribution_volume()
        computed_indices = labels.dense_volume_distribution_to_dense_volume_indices( distribution )
        expected_indices = TestData.index_volume().reshape( ( 4, 4, 4 ) )

        self.assertTrue( numpy.array_equal( computed_indices, expected_indices ) )


    def test_dense_volume_indices_to_dense_volume_masks( self ) :

        indices = TestData.index_volume().reshape( ( 4, 4, 4 ) )
        group_by_mask = ( 3, 0, 1, 2 )
        distributions_as_masks = TestData.distribution_volume().astype( 'int16' )
        expected_masks = numpy.transpose( distributions_as_masks, group_by_mask )
        computed_masks = labels.dense_volume_indices_to_dense_volume_masks( indices, 64 )

        self.assertTrue( numpy.array_equal( computed_masks, expected_masks ) )


#---------------------------------------------------------------------------------------------------


class SparseLabelTests( unittest.TestCase ) :


    def test_dense_patch_distribution_to_sparse_patch_distribution( self ) :

        dense_distribution = TestData.disjoint_distribution_patches()
        expected_samples = dense_distribution[ :, 1, 1, 1, : ].reshape( 8, 64 )
        computed_samples = labels.dense_patch_distribution_to_sparse_patch_distribution( dense_distribution )

        self.assertTrue( numpy.array_equal( computed_samples, expected_samples ) )


    def test_sparse_patch_distribution_to_dense_volume_distribution( self ) :

        dense_patches = TestData.overlapping_distribution_patches()
        sparse_patches = dense_patches[ :, 1, 1, 1, : ].reshape( 27, 64 )
        computed_volume = labels.sparse_patch_distribution_to_dense_volume_distribution(
            sparse_patches,
            ( 3, 3, 3 ) )
        expected_volume = TestData.distribution_volume()[ 1:, 1:, 1:, : ].reshape( ( 3, 3, 3, 64 ) )

        self.assertTrue( numpy.array_equal( computed_volume, expected_volume ) )


#---------------------------------------------------------------------------------------------------
