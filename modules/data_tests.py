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

if __name__ == '__main__' :

    unittest.main()


#---------------------------------------------------------------------------------------------------
