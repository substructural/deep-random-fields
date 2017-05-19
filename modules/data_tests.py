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

        expected = numpy.array( ( minima, maxima ) )
        computed = data.Batch.normalised_bounds_of_unmasked_regions( volumes )

        pdb.set_trace()
        self.assertTrue( numpy.array_equal( computed, expected ) )


    def test_that_offsets_covers_the_exact_region_at_the_target_grid_points( self ):
        pass


    def test_that_patch_offsets_match_the_unmasked_region( self ):
        pass


    def test_that_image_patches_match_the_patch_offsets( self ):
        pass


    def test_that_label_patches_match_the_patch_offsets( self ):
        pass


    def test_that_mask_patches_match_the_patch_offsets( self ):
        pass


#---------------------------------------------------------------------------------------------------

class ParametersTests( unittest.TestCase ):


    def test_create_with_volume_count( self ):
        pass


    def test_with_patch_shape( self ):
        pass


    def test_with_patch_stride( self ):
        pass


    def test_with_constrain_to_mask( self ):
        pass


    def test_with_window_margin( self ):
        pass


#---------------------------------------------------------------------------------------------------

if __name__ == '__main__' :

    unittest.main()


#---------------------------------------------------------------------------------------------------
