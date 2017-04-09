#===================================================================================================
# data tests

import numpy
import data

import unittest
import pdb

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

class LoaderTests( unittest.TestCase ) :


    def test_dense_to_patch_based_labels_yields_the_correct_label_per_patch( self ) :

        dense_labels = numpy.array( [
            [ [ k * 100 + j * 10 + i for i in range( 0, 10 ) ] for j in range( 0, 10 ) ]
            for k in range( 0, 20 ) ] )

        expected_patch_based_labels = numpy.array( [ k * 100 + 55 for k in range( 0, 20 ) ] )
        computed_patch_based_labels = data.Loader.dense_to_patch_based_labels( dense_labels )

        self.assertTrue( numpy.array_equal( computed_patch_based_labels, expected_patch_based_labels ) )


    def test_dense_labels_in_centred_window_has_the_correct_offset_and_dimensions( self ) :

        X = 10
        Y = 10
        N = 20
        d = 2

        dense_labels = numpy.array( [
            [ [ k * 100 + j * 10 + i for i in range( 0, X ) ] for j in range( 0, Y ) ]
            for k in range( 0, N ) ] )

        expected_patch_based_labels = numpy.array( [
            [ [ k * 100 + j * 10 + i for i in range( 0 + d, X - d ) ] for j in range( 0 + d, Y - d ) ]
            for k in range( 0, N ) ] )

        computed_patch_based_labels = data.Loader.dense_labels_in_centered_window( dense_labels, d )

        self.assertTrue( numpy.array_equal( computed_patch_based_labels, expected_patch_based_labels ) )


    def test_labels_to_distribution_assigns_the_correct_probability_to_the_correct_class( self ) :

        labels = numpy.array( [ ( i % 5 ) for i in range( 0, 20 ) ] )

        computed_distribution = data.Loader.labels_to_distribution( labels, 5 )
        expected_distribution = numpy.array(
            [ [ ( 1.0 if ( i % 5 ) == k else 0.0 ) for k in range( 0, 5 ) ] for i in range( 0, 20 ) ] )

        self.assertTrue( numpy.array_equal( computed_distribution, expected_distribution ) )


    def test_per_patch_distribution_over_labels( self ) :

        initial_images = numpy.array(
            [ [ [ ( n * 100 + j * 10 + i )
                for i in range( 0, 10 ) ]
              for j in range( 0, 10 ) ]
            for n in range( 0, 20 ) ] )

        initial_labels = numpy.array(
            [ [ [ ( ( n % 5 ) if ( i == j and j == 5 ) else 6 )
                for i in range( 0, 10 ) ]
              for j in range( 0, 10 ) ]
            for n in range( 0, 20 ) ] )

        mock_batch = Mock.Batch( image_patches = initial_images, label_patches = initial_labels )

        output_image_patches, computed_distribution = data.Loader.per_patch_distribution_over_labels( mock_batch, 5 )
        expected_distribution = numpy.array(
            [ [ ( 1.0 if ( i % 5 ) == k else 0.0 ) for k in range( 0, 5 ) ] for i in range( 0, 20 ) ] )

        self.assertTrue( numpy.array_equal( computed_distribution, expected_distribution ) )
        self.assertTrue( numpy.array_equal( output_image_patches, initial_images ) )


    def test_dense_distribution_over_labels( self ) :

        K = 5
        d = 2

        initial_images = numpy.array(
            [ [ [ ( n * 100 + j * 10 + i )
                for i in range( 0, 10 ) ]
              for j in range( 0, 10 ) ]
            for n in range( 0, 20 ) ] )

        initial_labels = numpy.array(
            [ [ [ ( ( n + j + i ) % K ) 
                for i in range( 0, 10 ) ]
              for j in range( 0, 10 ) ]
            for n in range( 0, 20 ) ] )

        expected_distribution = numpy.array(
            [ [ [ [ ( 1.0 if ( ( ( n + j + i ) % K ) == k ) else 0.0 )
                  for k in range( 0, K ) ]
                for i in range( 0 + d, 10 - d ) ]
              for j in range( 0 + d, 10 - d ) ]
            for n in range( 0, 20 ) ] )

        mock_batch = Mock.Batch( image_patches = initial_images, label_patches = initial_labels )
        output_image_patches, computed_distribution = data.Loader.dense_distribution_over_labels( mock_batch, d, K )

        self.assertTrue( numpy.array_equal( computed_distribution, expected_distribution ) )
        self.assertTrue( numpy.array_equal( output_image_patches, initial_images ) )

        
    def test_that_accessors_are_consistent_with_constructor( self ) :

        dataset = Mock.Dataset( training_set = [ 1, 2, 3, 4 ], validation_set = [ 1, 2, 3, 4 ] )
        label_count = 5
        window_margin = 2
        training_set_parameters = data.Parameters( 20, ( 10, 10 ) ).with_patch_stride( 2 ).with_window_margin( 2 )
        loader = data.Loader( dataset, training_set_parameters, label_count, window_margin ) 

        self.assertEqual( loader.dataset, dataset )
        self.assertEqual( loader.label_count, label_count )

        
    @staticmethod
    def loader_under_test( image_data, label_data, mask_data, parameters, split = 2, label_count = 2 ) :

        volumes = [ data.Volume( image_data[ c ], label_data[ c ], mask_data[ c ] ) for c in range( 0, 4 ) ]
        aquisitions = [ Mock.Aquisition( c, c, 1, volumes[ c ] ) for c in range( 0, 4 ) ]
        dataset = Mock.Dataset( training_set = aquisitions[ 0 : split ], validation_set = aquisitions[ split : ] )

        return data.Loader( dataset, parameters, label_count ) 

        
    @staticmethod
    def define_volume( f, K, J, I, C ) :

        return numpy.array(
            [ [ [ [ f( c, k, j, i )
                    for i in range( 0, I ) ]
                  for j in range( 0, J ) ]
                for k in range( 0, K ) ]
              for c in range( 0, C ) ] )


    @staticmethod
    def define_patches( f, C, Z, Y, X, patch_offsets ) :

        return numpy.array(
            [ [ [ [ [ f( c, k + z, j + y, i + x )
                      for x in range( 0, X ) ]
                    for y in range( 0, Y ) ]
                  for z in range( 0, Z ) ]
                for ( k, j, i ) in patch_offsets ]
              for c in range( 0, C ) ] )


    @staticmethod
    def define_labels( f, C, z, y, x, patch_offsets ) :

        return numpy.array(
            [ [ f( c, k + z, j + y, i + x )
                for ( k, j, i ) in patch_offsets ]
              for c in range( 0, C ) ] )

            
    def test_that_training_batches_have_the_correct_per_patch_labels( self ) :

        C, K, J, I = 4, 1, 6, 5
        Z, Y, X = 1, 3, 3
        training_split = 2

        image_value = lambda c, k, j, i : ( ( j * 10 ) + i + c )
        label_value = lambda c, k, j, i : ( ( i + j + c ) % 3 )
        mask_value  = lambda c, k, j, i : ( j < 5 )

        image_data = LoaderTests.define_volume( image_value, K, J, I, C )
        label_data = LoaderTests.define_volume( label_value, K, J, I, C )
        mask_data  = LoaderTests.define_volume( mask_value,  K, J, I, C )

        patch_offsets = [ ( 0, 0, 0 ), ( 0, 0, 2 ), ( 0, 2, 0 ), ( 0, 2, 2 ) ]
        expected_image_patches = LoaderTests.define_patches( image_value, training_split, Z, Y, X, patch_offsets )
        expected_per_patch_labels = LoaderTests.define_labels( label_value, training_split, 0, 1, 1, patch_offsets )

        expected_per_patch_distribution_over_labels = data.Loader.labels_to_distribution(
            expected_per_patch_labels, label_count = 3 )

        parameters = data.Parameters( 1, patch_shape = ( 1, 3, 3 ), patch_stride = 2 )
        loader = LoaderTests.loader_under_test(
            image_data, label_data, mask_data, parameters, split = training_split, label_count = 3 )

        for batch_index in range( 0, training_split ) :
            
            image_patches, label_distribution = (
                loader.load_training_batch_with_per_patch_distribution_over_labels( batch_index ) )
            
            self.assertTrue( numpy.array_equal( image_patches, expected_image_patches[ batch_index ] ) )
            self.assertTrue( numpy.array_equal( label_distribution,
                                                expected_per_patch_distribution_over_labels[ batch_index ] ) )


    def test_that_validation_batches_have_the_correct_per_patch_labels( self ) :

        C, K, J, I = 4, 1, 6, 5
        Z, Y, X = 1, 3, 3
        training_split = 2

        image_value = lambda c, k, j, i : ( ( j * 10 ) + i + c )
        label_value = lambda c, k, j, i : ( ( i + j + c ) % 3 )
        mask_value  = lambda c, k, j, i : ( j < 5 )

        image_data = LoaderTests.define_volume( image_value, K, J, I, C )
        label_data = LoaderTests.define_volume( label_value, K, J, I, C )
        mask_data  = LoaderTests.define_volume( mask_value,  K, J, I, C )

        patch_offsets = [ ( 0, 0, 0 ), ( 0, 0, 2 ), ( 0, 2, 0 ), ( 0, 2, 2 ) ]
        all_image_patches = LoaderTests.define_patches( image_value, C, Z, Y, X, patch_offsets )
        all_per_patch_labels = LoaderTests.define_labels( label_value, C, 0, 1, 1, patch_offsets )

        expected_image_patches = numpy.array( [ p for v in all_image_patches[ training_split : C ] for p in v ] )
        expected_per_patch_labels = numpy.array( [ l for v in all_per_patch_labels[ training_split : C ] for l in v ] )
        expected_per_patch_distribution_over_labels = data.Loader.labels_to_distribution(
            expected_per_patch_labels, label_count = 3 )

        parameters = data.Parameters( 1, patch_shape = ( 1, 3, 3 ), patch_stride = 2 )
        loader = LoaderTests.loader_under_test(
            image_data, label_data, mask_data, parameters, split = 2, label_count = 3 )

        image_patches, label_distribution = loader.load_validation_set_with_per_patch_distribution_over_labels()
        print( "\n\ncomputed labels:\n" + str( label_distribution ) + "\n" )
        print( "\n\nexpected labels:\n" + str( expected_per_patch_distribution_over_labels ) + "\n" )
        print( "\n\ncomputed patches:\n" + str( image_patches ) + "\n" )
        print( "\n\nexpected patches:\n" + str( expected_image_patches ) + "\n" )
        self.assertTrue( numpy.array_equal( image_patches, expected_image_patches ) )
        self.assertTrue( numpy.array_equal( label_distribution, expected_per_patch_distribution_over_labels ) )


#---------------------------------------------------------------------------------------------------

if __name__ == '__main__' :

    unittest.main()


#---------------------------------------------------------------------------------------------------
