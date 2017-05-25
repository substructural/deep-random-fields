#===================================================================================================
# data tests

import unittest
import pdb

import numpy

import data
import geometry


#---------------------------------------------------------------------------------------------------

class Mock :


    class Aquisition( data.Aquisition ) :

        def __init__( self, aquisition_id, subject, subject_age_at_aquisition, volume ) :
            super( Mock.Aquisition, self ).__init__( aquisition_id, subject, subject_age_at_aquisition )
            self.volume = volume

        def read_volume( self ) :
            return self.volume


    class Dataset( object ) :

        def __init__( self, training_set = None, validation_set = None, test_set = None ) :
            self.training_set = training_set
            self.validation_set = validation_set
            self.test_set = test_set


    class Batch( object ) :

        def __init__( self, image_patches = None, label_patches = None, mask_patches = None ) :
            self.image_patches = image_patches
            self.label_patches = label_patches
            self.mask_patches = mask_patches


def arrays_are_equal( computed, expected ):

    equal = numpy.array_equal( computed, expected )
    if not equal:
        print( "\narrays differ\n" )
        print( "expected:\n" + str( expected ) + "\n" )
        print( "computed:\n" + str( computed ) + "\n" )
        if computed.shape != expected.shape:
            print( "shapes differ: {0} != {1}".format( computed.shape, expected.shape ) )
        else:
            print( "differences:\n" + str( computed - expected ) + "\n" )

    return equal

#---------------------------------------------------------------------------------------------------

class VolumeTests( unittest.TestCase ) :


    @staticmethod
    def volume_under_test( j0, j1, jN, i0, i1, iN ):

        I = range( 0, iN )
        J = range( 0, jN )
        masked = lambda j, i : 1 if ( i > i0 and i < i1 ) and ( j > j0 and j < j1 ) else 0
        mask   = numpy.array( [ [ [ masked( j, i ) for i in I ] for j in J ] ] )
        image  = numpy.array( [ [ [ ( j * 10 + i ) for i in I ] for j in J ] ] )
        labels = numpy.array( [ [ [ i for i in I ] for j in J ] ] )
        return data.Volume( image, labels, mask )


    def test_unmasked_bounds( self ) :

        j0, j1, jN = 2, 7, 10
        i0, i1, iN = 3, 6, 8

        volume = VolumeTests.volume_under_test( j0, j1, jN, i0, i1, iN )
        bounds = volume.unmasked_bounds
        expected_bounds = geometry.cuboid( ( 0, j0 + 1, i0 + 1 ), ( 0, j1 - 1, i1 - 1 ) )

        self.assertTrue( numpy.array_equal( bounds, expected_bounds ) )


    def test_centre( self ):

        j0, j1, jN = 2, 7, 10
        i0, i1, iN = 3, 6, 8

        volume = VolumeTests.volume_under_test( j0, j1, jN, i0, i1, iN )
        centre = volume.centre

        j = j0 + ( ( j1 - j0 ) / 2.0 )
        i = i0 + ( ( i1 - i0 ) / 2.0 )
        expected_centre = geometry.voxel( 0, j, i )

        self.assertTrue( numpy.array_equal( centre, expected_centre ) )


#---------------------------------------------------------------------------------------------------

class DatasetTests( unittest.TestCase ):


    def test_that_all_aquisitions_are_assigned_when_equal_to_count( self ) :

        aquisitions = [
            Mock.Aquisition( i, data.Subject( "S_" + str( i ), i % 2, None ), 20 + i, [] )
            for i in range( 0, 20 ) ]

        dataset = data.Dataset( aquisitions, 10, 5, 5, 42 )
        computed = set( dataset.training_set + dataset.validation_set + dataset.test_set )
        expected = set( aquisitions )

        self.assertEqual( computed, expected )


    def test_that_the_specified_split_is_used_absent_subject_constraints( self ) :

        aquisitions = [
            Mock.Aquisition( i, data.Subject( "S_" + str( i ), i % 2, None ), 20 + i, [] )
            for i in range( 0, 20 ) ]

        dataset = data.Dataset( aquisitions, 10, 5, 5, 42 )
        training_set = set( dataset.training_set )
        validation_set = set( dataset.validation_set )
        test_set = set( dataset.test_set )

        self.assertEqual( len( training_set ), 10 )
        self.assertEqual( len( validation_set ), 5 )
        self.assertEqual( len( test_set ), 5 )


    def test_that_all_aquisitions_for_a_single_subject_are_assigned_to_a_single_subset( self ):

        subjects = [ data.Subject( "S_" + str( i ), i % 2, None ) for i in range( 0, 5 ) ]
        aquisitions = [
            Mock.Aquisition( i, subjects[ i % 5 ], 20 + i, [] ) for i in range( 0, 20 ) ]

        dataset = data.Dataset( aquisitions, 10, 5, 5, 42 )
        training_subjects = sorted( [ str( a.subject ) for a in dataset.training_set ] )
        validation_subjects = sorted( [ str( a.subject ) for a in dataset.validation_set ] )
        test_subjects = sorted( [ str( a.subject ) for a in dataset.test_set ] )

        for subject in training_subjects:
            self.assertTrue( subject not in validation_subjects )
            self.assertTrue( subject not in test_subjects )

        for subject in validation_subjects:
            self.assertTrue( subject not in training_subjects )
            self.assertTrue( subject not in test_subjects )

        for subject in test_subjects:
            self.assertTrue( subject not in training_subjects )
            self.assertTrue( subject not in validation_subjects )


#---------------------------------------------------------------------------------------------------

class BatchTests( unittest.TestCase ):


    def test_volumes_for_batch_uses_correct_batch_offset( self ):

        aquisitions = [
            Mock.Aquisition( i, data.Subject( "S_" + str( i ), i % 2, None ), 20 + i, [ i ] )
            for i in range( 0, 20 ) ]

        batch_parameters = data.Parameters( 3 )
        computed = data.Batch.volumes_for_batch( aquisitions, 2, batch_parameters )
        expected = [ [ i ] for i in [ 6, 7, 8 ] ]

        self.assertEqual( computed, expected )


    def test_that_normalised_bounds_of_unmasked_region_matches_the_smallest_volume( self ):

        images = numpy.zeros( ( 5, 10, 10, 10 ) )
        labels = numpy.zeros( ( 5, 10, 10, 10 ) )

        mask = lambda m, d, k, j, i : (
            ( ( m <= i <= m + d ) and ( m <= j <= m + d ) and ( k == 1 ) ) )

        masks = numpy.array(
            [ [ [ [ 1 if mask( 2 + m, m, k, j, i ) else 0
                    for i in range( 0, 10 ) ]
                  for j in range( 0, 10 ) ]
                for k in range( 0, 3 ) ]
              for m in range( 0, 4 ) ] )

        volumes = [ data.Volume( images[ i ], labels[ i ], masks[ i ] ) for i in range( 0, 4 ) ]

        span = int( ( ( 8 - 5 ) + 1 ) / 2 )
        centres = [ ( 2+0 + 0/2 ), ( 2+1 + 1/2 ), ( 2+2 + 2/2 ), ( 2+3 + 3/2 ) ]
        minima = [ geometry.voxel( 1, int( c ) - span, int( c ) - span ) for c in centres ]
        maxima = [ geometry.voxel( 1, int( c ) + span, int( c ) + span ) for c in centres ]

        expected = numpy.transpose( numpy.array( ( minima, maxima ) ), axes=( 1, 0, 2 ) )
        computed = data.Batch.normalised_bounds_of_unmasked_regions( volumes )

        self.assertTrue( arrays_are_equal( computed, expected ) )


    def test_that_offsets_covers_the_exact_region_at_the_target_grid_points( self ):

        parameters = data.Parameters( 1 ).with_patch_stride( 3 ).with_patch_shape( ( 6, 4, 3 ) )
        volume_shape = ( 15, 16, 17 )
        bounds = numpy.array( [ [ 0, 1, 2 ], [ 10, 10, 10 ] ] )

        computed = data.Batch.offsets( volume_shape, bounds, parameters )
        expected = numpy.array( [
            [ 0, 1, 2 ],
            [ 0, 1, 5 ],
            [ 0, 1, 8 ],
            [ 0, 4, 2 ],
            [ 0, 4, 5 ],
            [ 0, 4, 8 ],
            [ 0, 7, 2 ],
            [ 0, 7, 5 ],
            [ 0, 7, 8 ],
            [ 3, 1, 2 ],
            [ 3, 1, 5 ],
            [ 3, 1, 8 ],
            [ 3, 4, 2 ],
            [ 3, 4, 5 ],
            [ 3, 4, 8 ],
            [ 3, 7, 2 ],
            [ 3, 7, 5 ],
            [ 3, 7, 8 ] ] )

        self.assertTrue( arrays_are_equal( expected, computed ) )


    def test_that_patches_are_positioned_at_specified_offsets_and_have_correct_size( self ):

        parameters = data.Parameters( 2 ).with_patch_shape( ( 1, 2, 2 ) )

        I = J = K = range( 0, 5 )
        V = range( 1, 3 )
        volume_data = numpy.array( [
            [ [ [ v*1000 + k*100 + j*10 + i for i in I ] for j in J ] for k in K ] for v in V ] )

        offsets_per_volume = numpy.array( [
            [ [ 2, 1, 0 ],
              [ 3, 3, 1 ] ],
            [ [ 0, 1, 2 ],
              [ 1, 2, 3 ] ] ] )

        expected_patches = numpy.array( [
            [ [ [ [ 1210, 1211 ],
                  [ 1220, 1221 ] ] ],
              [ [ [ 1331, 1332 ],
                  [ 1341, 1342 ] ] ] ],
            [ [ [ [ 2012, 2013 ],
                  [ 2022, 2023 ] ] ],
              [ [ [ 2123, 2124 ],
                  [ 2133, 2134 ] ] ] ] ] )

        computed_patches = data.Batch.patches( volume_data, offsets_per_volume, parameters )

        self.assertTrue( arrays_are_equal( expected_patches, computed_patches ) )


    def test_that_patches_generated_for_batch_are_positioned_at_specified_offsets( self ):

        I = J = K = range( 0, 5 )
        V = range( 0, 2 )
        bounds = [ [ [ 0, 1, 2 ], [ 0, 3, 4 ] ], [ [ 2, 1, 0 ], [ 2, 3, 2 ] ] ]
        mask = lambda v, k, j, i : geometry.in_bounds( ( k, j, i ), bounds[ v ] )

        mask_data = numpy.array(
            [ [ [ [ mask( v, k, j, i ) for i in I ] for j in J ] for k in K ] for v in V ] )

        image_data = numpy.array(
            [ [ [ [ (v + 1)*1000 + k*100 + j*10 + i for i in I ] for j in J ] for k in K ] for v in V ] )

        label_data = image_data # in order to better distinguish the labels

        volumes = [
            None, # we will test the second batch, ignoring the first two volumes
            None,
            data.Volume( image_data[ 0 ], label_data[ 0 ], mask_data[ 0 ] ),
            data.Volume( image_data[ 1 ], label_data[ 1 ], mask_data[ 1 ] ) ]

        subjects = [ data.Subject( "s" + str( i ), "m", None ) for i in range( 0, 4 ) ]
        aquisitions = [ Mock.Aquisition( "a" + str( i ), subjects[ i ], 99, volumes[ i ] ) for i in range( 0, 4 ) ]
        parameters = data.Parameters( 2 ).with_patch_stride( 2 ).with_patch_shape( ( 1, 1, 1 ) )
        batch_index = 1
        batch = data.Batch( aquisitions, batch_index, parameters )

        expected_patches = numpy.array( [
            [ [[[1012]]], [[[1014]]],
              [[[1032]]], [[[1034]]] ],
            [ [[[2210]]], [[[2212]]],
              [[[2230]]], [[[2232]]] ] ] )

        self.assertTrue( arrays_are_equal( batch.image_patches, expected_patches ) )
        self.assertTrue( arrays_are_equal( batch.label_patches, expected_patches ) )


#---------------------------------------------------------------------------------------------------

class ParametersTests( unittest.TestCase ):


    def test_create_with_volume_count( self ):

        parameters = data.Parameters( 42 )
        self.assertEqual( parameters.volume_count, 42 )


    def test_with_patch_shape( self ):

        parameters = data.Parameters( 42 ).with_patch_shape( ( 6, 9 ) )
        self.assertEqual( parameters.patch_shape, ( 6, 9 ) )


    def test_with_patch_stride( self ):

        parameters = data.Parameters( 42 ).with_patch_stride( 7 )
        self.assertEqual( parameters.patch_stride, 7 )


    def test_with_constrain_to_mask( self ):

        for flag in [ True, False ]:
            parameters = data.Parameters( 42 ).with_constrain_to_mask( flag )
            self.assertEqual( parameters.constrain_to_mask, flag )


    def test_with_window_margin( self ):

        parameters = data.Parameters( 42 ).with_window_margin( 3 )
        self.assertEqual( parameters.window_margin, 3 )


#---------------------------------------------------------------------------------------------------

if __name__ == '__main__' :

    unittest.main()


#---------------------------------------------------------------------------------------------------
