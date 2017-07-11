#===================================================================================================
# sample tests

import unittest

import numpy

import data
import geometry
import sample

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

class ParametersTests( unittest.TestCase ):


    def test_create_with_volume_count( self ):

        parameters = sample.Parameters().with_volume_count( 42 )
        self.assertEqual( parameters.volume_count, 42 )


    def test_with_patch_shape( self ):

        parameters = sample.Parameters( 42 ).with_patch_shape( ( 6, 9 ) )
        self.assertEqual( parameters.patch_shape, ( 6, 9 ) )


    def test_with_patch_stride( self ):

        parameters = sample.Parameters( 42 ).with_patch_stride( 7 )
        self.assertEqual( parameters.patch_stride, 7 )


    def test_with_constrain_to_mask( self ):

        for flag in [ True, False ]:
            parameters = sample.Parameters( 42 ).with_constrain_to_mask( flag )
            self.assertEqual( parameters.constrain_to_mask, flag )


    def test_with_window_margin( self ):

        parameters = sample.Parameters( 42 ).with_window_margin( 3 )
        self.assertEqual( parameters.window_margin, 3 )


#---------------------------------------------------------------------------------------------------

class PatchSetTests( unittest.TestCase ):


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
        computed = sample.PatchSet.normalised_bounds_of_unmasked_regions( volumes )

        self.assertTrue( arrays_are_equal( computed, expected ) )


    def test_that_offsets_covers_the_exact_region_at_the_target_grid_points( self ):

        patch_stride = 3
        patch_shape = ( 6, 4, 3 ) 
        bounds = numpy.array( [ [ 0, 1, 2 ], [ 10, 10, 10 ] ] )

        computed = sample.PatchSet.offsets( bounds, patch_shape, patch_stride )
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

        I = J = K = range( 0, 5 )
        V = range( 1, 3 )
        volume_data = numpy.array( [
            [ [ [ v*1000 + k*100 + j*10 + i for i in I ] for j in J ] for k in K ] for v in V ] )

        offsets_per_volume = numpy.array( [
            [ [ 2, 1, 0 ],
              [ 3, 3, 1 ] ],
            [ [ 0, 1, 2 ],
              [ 1, 2, 3 ] ] ] )

        patch_shape = ( 1, 2, 2 )

        expected_patches = numpy.array( [
            [ [ [ [ 1210, 1211 ],
                  [ 1220, 1221 ] ] ],
              [ [ [ 1331, 1332 ],
                  [ 1341, 1342 ] ] ] ],
            [ [ [ [ 2012, 2013 ],
                  [ 2022, 2023 ] ] ],
              [ [ [ 2123, 2124 ],
                  [ 2133, 2134 ] ] ] ] ] )

        computed_patches = sample.PatchSet.extract( volume_data, offsets_per_volume, patch_shape )

        self.assertTrue( arrays_are_equal( expected_patches, computed_patches ) )


#---------------------------------------------------------------------------------------------------

class MockRandomGenerator( object ):

    def __init__( self, start, increment ):
        self.start = start
        self.increment = increment

    def permutation( self, xs ):
        # this is not guaranteed to be a perutation of course, but this is fune for a test
        count = xs.shape[0]
        index = [ ( self.start + i * self.increment ) % count for i in range( count ) ]
        permuted = numpy.array( [ xs[ index[ i ] ] for i in range( count ) ] )
        return permuted


class RandomPatchSetTests( unittest.TestCase ):


    def test_volumes_for_batch_uses_correct_batch_offset( self ):

        aquisitions = [
            Mock.Aquisition( i, data.Subject( "S_" + str( i ), i % 2, None ), 20 + i, [ i ] )
            for i in range( 0, 10 ) ]

        batches_per_iteration = 4
        volumes_per_batch = 3
        computed_batch = lambda i: sample.RandomPatchSet.volumes_for_batch(
            aquisitions, i, batches_per_iteration, volumes_per_batch )

        expected0 = [ [ i ] for i in [ 0, 1, 2 ] ]
        computed_volumes_0 = computed_batch( 0 )
        self.assertEqual( computed_volumes_0, expected0 )

        expected3 = [ [ i ] for i in [ 9 ] ]
        computed_volumes_3 = computed_batch( 3 )
        self.assertEqual( computed_volumes_3, expected3 )

        expected4 = [ [ i ] for i in [ 0, 1, 2 ] ]
        computed_volumes_4 = computed_batch( 4 )
        self.assertEqual( computed_volumes_4, expected4 )


    @staticmethod
    def construct_aquisitions_with_pixels_values_set_to_indices():

        I = range( 8 )
        J = range( 5 )
        K = range( 5 )
        V = range( 3 )

        bounds = [ 
            [ [ 1, 1, 1 ], [ 1, 3, 7 ] ],
            [ [ 2, 1, 1 ], [ 2, 3, 7 ] ],
            [ [ 3, 1, 1 ], [ 3, 3, 7 ] ] ]

        mask = lambda v, k, j, i : geometry.in_bounds( ( k, j, i ), bounds[ v ] )
        mask_data = numpy.array(
            [ [ [ [ mask( v, k, j, i ) for i in I ] for j in J ] for k in K ] for v in V ] )

        image_data = numpy.array(
            [ [ [ [ (v + 1)*1000 + k*100 + j*10 + i for i in I ] for j in J ] for k in K ]
              for v in V ] )

        label_data = image_data # in order to better distinguish the labels

        volumes = [data.Volume( image_data[ v ], label_data[ v ], mask_data[ v ] ) for v in V ]
        aquisitions = [ Mock.Aquisition( str( v ), v, v, volumes[ v ] ) for v in V ]

        return aquisitions


    def test_that_patches_generated_for_batch_are_positioned_at_specified_offsets( self ):
        
        aquisitions = RandomPatchSetTests.construct_aquisitions_with_pixels_values_set_to_indices()

        parameters = ( sample.Parameters()
                       .with_volume_count( 2 )
                       .with_patch_count( 4 )
                       .with_patch_stride( 2 )
                       .with_patch_shape( ( 1, 2, 2 ) )
                       .with_seed( 42 ) )

        patches_before_permutation = numpy.array( [

            # volume 1
            [ [ [ [ 1111, 1112 ] , # patch 1
                  [ 1121, 1122 ] ] ],
              [ [ [ 1113, 1114 ] , # patch 2
                  [ 1123, 1124 ] ] ] , 
              [ [ [ 1115, 1116 ] , # patch 3
                  [ 1125, 1126 ] ] ] ],

            # volume 2
            [ [ [ [ 2211, 2212 ] , # patch 1
                  [ 2221, 2222 ] ] ],
              [ [ [ 2213, 2214 ] , # patch 2
                  [ 2223, 2224 ] ] ],
              [ [ [ 2215, 2216 ] , # patch 3
                  [ 2225, 2226 ] ] ] ],

            # volume 3
            [ [ [ [ 3311, 3312 ] , # patch 1
                  [ 3321, 3322 ] ] ],
              [ [ [ 3313, 3314 ] , # patch 2
                  [ 3323, 3324 ] ] ],
              [ [ [ 3315, 3316 ] , # patch 3
                  [ 3325, 3326 ] ] ] ]
        ] )
            
        non_random_generator = MockRandomGenerator( 1, 2 )
        patches_after_permutation = numpy.array([
            non_random_generator.permutation( patches_before_permutation[i] )
            for i in range( 3 ) ])

        batch0 = sample.RandomPatchSet( aquisitions, 0, parameters, non_random_generator )
        expected0 = patches_after_permutation[ 0:2, 0:2 ] 
        self.assertEqual( batch0.image_patches.shape, expected0.shape )
        self.assertTrue( arrays_are_equal( batch0.image_patches, expected0 ) )
        self.assertTrue( arrays_are_equal( batch0.label_patches, expected0 ) )

        batch1 = sample.RandomPatchSet( aquisitions, 1, parameters, non_random_generator )
        expected1 = patches_after_permutation[ 2:3, 0:2 ] 
        self.assertEqual( batch1.image_patches.shape, expected1.shape )
        self.assertTrue( arrays_are_equal( batch1.image_patches, expected1 ) )
        self.assertTrue( arrays_are_equal( batch1.label_patches, expected1 ) )

        batch2 = sample.RandomPatchSet( aquisitions, 2, parameters, non_random_generator )
        expected2 = patches_after_permutation[ 0:2, 2:3 ] 
        self.assertEqual( batch2.image_patches.shape, expected2.shape )
        self.assertTrue( arrays_are_equal( batch2.image_patches, expected2 ) )
        self.assertTrue( arrays_are_equal( batch2.label_patches, expected2 ) )

        batch3 = sample.RandomPatchSet( aquisitions, 3, parameters, non_random_generator )
        expected3 = patches_after_permutation[ 2:3, 2:3 ] 
        self.assertEqual( batch3.image_patches.shape, expected3.shape )
        self.assertTrue( arrays_are_equal( batch3.image_patches, expected3 ) )
        self.assertTrue( arrays_are_equal( batch3.label_patches, expected3 ) )


#---------------------------------------------------------------------------------------------------

class ContiguousPatchSetTests( unittest.TestCase ):


    @staticmethod
    def construct_aquisitions_with_pixels_values_set_to_indices():

        I = range( 8 )
        J = range( 5 )
        K = range( 5 )
        V = range( 3 )

        bounds = [ 
            [ [ 1, 1, 1 ], [ 1, 3, 7 ] ],
            [ [ 2, 1, 1 ], [ 2, 3, 7 ] ],
            [ [ 3, 1, 1 ], [ 3, 3, 7 ] ] ]

        mask = lambda v, k, j, i : geometry.in_bounds( ( k, j, i ), bounds[ v ] )
        mask_data = numpy.array(
            [ [ [ [ mask( v, k, j, i ) for i in I ] for j in J ] for k in K ] for v in V ] )

        image_data = numpy.array(
            [ [ [ [ (v + 1)*1000 + k*100 + j*10 + i for i in I ] for j in J ] for k in K ]
              for v in V ] )

        label_data = image_data # in order to better distinguish the labels

        volumes = [data.Volume( image_data[ v ], label_data[ v ], mask_data[ v ] ) for v in V ]
        aquisitions = [ Mock.Aquisition( str( v ), v, v, volumes[ v ] ) for v in V ]

        return aquisitions


    def test_target_bounds( self ):

        prime_bounds = sample.Parameters().with_target_bounds( 2, 3, 5 )


        
        even_bounds = sample.Parameters().with_target_bounds( 2, 4, 8 )
    

    def test_that_volume_and_patch_index_are_correct_for_initial_patch( self ):
        
        aquisitions = ContiguousPatchSetTests.construct_aquisitions_with_pixels_values_set_to_indices()

        parameters = ( sample.Parameters()
                       .with_volume_count( 2 )
                       .with_patch_count( 4 )
                       .with_patch_stride( 2 )
                       .with_patch_shape( ( 1, 2, 2 ) )
                       .with_seed( 42 ) )

        patch_set = sample.ContiguousPatchSet( 
        expected_patches = numpy.array( [

            # volume 1
            [ [ [ [ 1111, 1112 ] , # patch 1
                  [ 1121, 1122 ] ] ],
              [ [ [ 1113, 1114 ] , # patch 2
                  [ 1123, 1124 ] ] ] , 
              [ [ [ 1115, 1116 ] , # patch 3
                  [ 1125, 1126 ] ] ] ],

            # volume 2
            [ [ [ [ 2211, 2212 ] , # patch 1
                  [ 2221, 2222 ] ] ],
              [ [ [ 2213, 2214 ] , # patch 2
                  [ 2223, 2224 ] ] ],
              [ [ [ 2215, 2216 ] , # patch 3
                  [ 2225, 2226 ] ] ] ],

            # volume 3
            [ [ [ [ 3311, 3312 ] , # patch 1
                  [ 3321, 3322 ] ] ],
              [ [ [ 3313, 3314 ] , # patch 2
                  [ 3323, 3324 ] ] ],
              [ [ [ 3315, 3316 ] , # patch 3
                  [ 3325, 3326 ] ] ] ]
        ] )
            

        self.assertTrue( False )


    def test_that_volume_and_patch_index_are_correct_for_initial_patch( self ):

        self.assertTrue( False )


    def test_that_volume_and_patch_index_are_correct_for_initial_patch( self ):

        self.assertTrue( False )


    


#---------------------------------------------------------------------------------------------------
