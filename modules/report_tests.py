#===================================================================================================
# report tests



#---------------------------------------------------------------------------------------------------

import os
import unittest

import numpy

import geometry

from results import SegmentationResults
from report import Report


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


class MockVolume( object ):


    @staticmethod
    def define( bounds, offset, outer_span ):

        neg = numpy.logical_not

        base_image = numpy.array(
            [ [ [ z + y + x
                  for x in range( bounds[2] ) ]
                for y in range( bounds[1] ) ]
              for z in range( bounds[0] ) ] )

        border = numpy.array(( 10, 10, 10 ))
        mask1 = geometry.mask( bounds, offset + border, offset + outer_span - border )
        mask2 = geometry.mask( bounds, offset, offset + outer_span ) & neg( mask1 )
        image = base_image + ( mask1 * 40 ) + ( mask2 * 40 )

        labels = numpy.zeros(( bounds )) + ( mask1 * 1 ) + ( mask2 * 2 )

        return image, labels


    def __init__( self, bounds, offset, outer_span ):

        images, labels = MockVolume.define( bounds, offset, outer_span )
        self.images = images
        self.labels = labels


class MockAquisition( object ):

    def __init__( self, volume ):

        self.volume = volume


    def read_volume( self ):

        return self.volume


class MockDataset( object ):

    def __init__( self, bounds, offsets, reference_spans ):

        assert len( offsets.shape ) == 2
        assert offsets.shape == reference_spans.shape

        count = offsets.shape[0]
        self.validation_set = [
            MockAquisition( 
                MockVolume( bounds, offsets[ i ], reference_spans[ i ] ) )
            for i in range( count ) ]


class MockResults( SegmentationResults ):

    
    @staticmethod
    def distribution( target, span, value ):

        neg = numpy.logical_not

        border = numpy.array(( 10, 10, 10 )).astype( 'int64' )
        zeros = numpy.zeros(( 3, )).astype( 'int64' )
        mask1 = geometry.mask( target, border, span - border )
        mask2 = geometry.mask( target, zeros, span ) & neg( mask1 )
        mask0 = neg( mask1 | mask2 )

        distribution0 = mask0 * value
        distribution1 = mask1 * value
        distribution2 = mask2 * value

        distribution = numpy.array([ distribution0, distribution1, distribution2 ])
        return numpy.transpose( distribution, ( 1, 2, 3, 0 ) )


    def __init__( self, target, offsets, predicted_spans, reference_spans, data_path, name ):

        assert len( offsets.shape ) == 2
        assert offsets.shape == predicted_spans.shape

        class_count = 3

        super( MockResults, self ).__init__( data_path, name, 0, class_count )

        self.offsets = offsets
        self.volume_count = offsets.shape[0]

        self.predicted = [
            MockResults.distribution( target, predicted_spans[ i ], 0.5 )
            for i in range( self.volume_count ) ]

        self.reference = [
            MockResults.distribution( target, reference_spans[ i ], 1.0 )
            for i in range( self.volume_count ) ]

        for i in range( self.volume_count ):
            self.append( self.predicted[ i ], self.reference[ i ] )
        

    def predicted_distribution( self, i ):

        return ( self.predicted[ i ], self.offsets[ i ] )


class MockExperimentDefinition( experiment.ExperimentDefinition ):


    @property
    def label_count( self ):

        return 3
    

    @property
    def sample_parameters( self ):

        raise NotImplementedError()


    def dataset( self, input_path, log ):

        raise NotImplementedError()


    def architecture( self ):

        raise NotImplementedError() 

    
    def optimiser( self, dataset, log ):

        raise NotImplementedError() 
    



#---------------------------------------------------------------------------------------------------


def write_report():

    unit = numpy.ones(( 3, )).astype( 'int64' )
    bounds = 200 * unit
    target = 150 * unit

    offsets = numpy.array([ ( i * 10 + 10 ) * unit for i in range( 5 ) ])
    outer_spans = numpy.array([ ( i * 20 + 60 ) * unit for i in range( 5 ) ])  
    inner_spans = numpy.array([ ( i * 10 + 60 ) * unit for i in range( 5 ) ])

    dataset = MockDataset( bounds, offsets, outer_spans )
    results = MockResults(
        target, offsets, inner_spans, outer_spans, output_directory(), 'report_test' )

    experiment_instance = None

    Report.write( results, experiment_instance )



class ReportTests( unittest.TestCase ):


    def test_by_regression( self ):

        pass # write_report()

         



#---------------------------------------------------------------------------------------------------
