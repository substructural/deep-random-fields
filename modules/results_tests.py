import os
import os.path
import unittest

import matplotlib.pyplot
import numpy

import geometry
import results

import pdb



#---------------------------------------------------------------------------------------------------


def input_directory():

    path = os.path.abspath( os.path.dirname( __file__ ) ) + '/test/input'
    if not os.path.exists( path ):
        raise Exception( f'input directory "{path}" does not exist' )

    return path


def output_directory():

    path = os.path.abspath( os.path.dirname( __file__ ) ) + '/test/output'
    if not os.path.exists( path ):
        os.mkdir( path )

    return path



#---------------------------------------------------------------------------------------------------


class ArchiveTests( unittest.TestCase ):


    def test_save_array_output( self ):

        I = J = K = range( 0, 100 )
        original_data = numpy.array(
            [ [ [ k*100 + j*10 + i for i in I ] for j in J ] for k in K ] )

        archive_name = 'archive'
        archive = results.Archive( output_directory(), archive_name )
        tag = "array-test"

        archive.save_array_output( original_data, tag )
        saved_name = archive.saved_object_file_name( tag ) + ".npy"
        reconstituted_data = numpy.load( saved_name )
        self.assertTrue( numpy.array_equal( reconstituted_data, original_data ) )

        archive.save_array_output( original_data, tag, 42 )
        saved_name_42 = archive.saved_object_file_name( tag ) + ".npy"
        reconstituted_data_42 = numpy.load( saved_name_42 )
        self.assertTrue( numpy.array_equal( reconstituted_data_42, original_data ) )



#---------------------------------------------------------------------------------------------------

class ImagesTests( unittest.TestCase ):

    '''
    These are not proper tests, they simply regress on an expected output.

    '''


    def test_pretty_much_everything_by_unintelligent_regression( self ):

        input_path = input_directory() + '/oasis_0001_axial.gif'
        output_path = output_directory() + '/difference_of_masks.png' 
        regression_path = input_directory() + '/difference_of_masks.png' 

        greyscale_image = matplotlib.pyplot.imread( input_path )

        a = geometry.mask( (208, 176), (40, 40), (100, 100) )
        b = geometry.mask( (208, 176), (70, 70), (150, 150) )
        difference_overlay = results.Images.difference_of_masks( a, b, include_true_negatives = True )
        difference_image = results.Images.overlay( greyscale_image, difference_overlay )

        results.Images.save_image( difference_image, output_path )

        generated = matplotlib.pyplot.imread( output_path )
        regression = matplotlib.pyplot.imread( regression_path )
        differences = regression - generated
        self.assertFalse( numpy.any( differences ) )


    
#---------------------------------------------------------------------------------------------------


class MetricsTests( unittest.TestCase ):

    
    def test_that_the_dice_score_of_segmentation_with_itself_is_equal_to_1( self ):

        a = geometry.mask( (10, 10, 10), (2, 2, 2), (8, 8, 8) )
        d = results.Metrics.dice_score( a, a )
        self.assertEqual( d, 1.0 )


    def test_that_the_dice_score_of_disjoint_segmentations_is_equal_to_zero( self ):

        a = geometry.mask( (10, 10, 10), (2, 2, 2), (4, 5, 6) )
        b = geometry.mask( (10, 10, 10), (5, 6, 7), (8, 8, 8) )
        d = results.Metrics.dice_score( a, b )
        self.assertEqual( d, 0.0 )


    def test_that_the_dice_is_twice_the_intersection_over_the_sum_of_the_sizes( self ):

        a = geometry.mask( (10, 10, 10), (1, 1, 1), (4, 4, 4) )
        b = geometry.mask( (10, 10, 10), (4, 4, 4), (5, 5, 5) )

        computed = results.Metrics.dice_score( a, b )
        expected = 2*1 / ( 4**3 + 2**3 )
        self.assertEqual( computed, expected )


    def test_confusion_matrix( self ):

        reference = numpy.array([
            [ 1, 1, 2, 2 ],
            [ 1, 1, 2, 2 ],
            [ 3, 3, 4, 4 ],
            [ 3, 3, 4, 4 ] ])

        predicted = numpy.array([
            [ 1, 0, 2, 1 ],
            [ 0, 0, 2, 0 ],
            [ 3, 3, 4, 4 ],
            [ 2, 0, 4, 4 ] ])
            
        classes = [ 0, 1, 2, 3, 4 ]
        reference_masks = [ reference == c for c in classes ]
        predicted_masks = [ predicted == c for c in classes ]

        computed = results.Metrics.confusion_matrix( predicted_masks, reference_masks )
        expected = numpy.array([
            [ 0, 0, 0, 0, 0 ],
            [ 3, 1, 0, 0, 0 ],
            [ 1, 1, 2, 0, 0 ],
            [ 1, 0, 1, 2, 0 ],
            [ 0, 0, 0, 0, 4 ] ])

        differences = computed - expected
        self.assertFalse( numpy.any( differences ) )
        



#---------------------------------------------------------------------------------------------------
