#===================================================================================================
# experiment tests



#---------------------------------------------------------------------------------------------------

import os
import sys
import unittest

import activation
import costs
import experiments
import layers
import learning_rates
import network
import oasis
import optimisation
import output
import sample

import numpy.random


#---------------------------------------------------------------------------------------------------


class RealDataExperiment( experiments.ExperimentDefinition ):


    @property
    def experiment_id( self ):

        return 'real_data_experiment'


    @property
    def label_count( self ):

        return 4


    def dataset( self, input_path, log ):

        return oasis.OasisDataSet( input_path, 8, 1, 1, 42, maybe_log = log )


    @property
    def sample_parameters( self ):

        return ( sample.Parameters()
                 .with_volume_count( 1 )
                 .with_window_margin( 5 )
                 .with_target_shape(( 110, 110, 110 ))
                 .with_patch_shape(( 20, 20, 20 ))
                 .with_patch_count( 100 )
                 .with_patch_stride( 10 ) )


    def model( self ):

        leaky_relu = activation.LeakyRectifiedLinearUnit( 0.1 )
        architecture = network.Architecture(
            [ layers.ConvolutionalLayer( 1, 8, ( 3, 3, 3 ), 1, leaky_relu ),
              layers.ConvolutionalLayer( 8, 8, ( 3, 3, 3 ), 1, leaky_relu ),
              layers.ConvolutionalLayer( 8, 8, ( 3, 3, 3 ), 1, leaky_relu ),
              layers.ConvolutionalLayer( 8, 8, ( 3, 3, 3 ), 1, leaky_relu ),
              layers.ConvolutionalLayer( 8, 4, ( 3, 3, 3 ), 1, leaky_relu ),
              layers.Softmax(),
              layers.ScalarFeatureMapsProbabilityDistribution()
            ],
            input_dimensions = 4,
            output_dimensions = 5 )

        return network.Model( architecture, seed = 42 )


    def optimiser( self, log ):

        distribution_axis = 1
        cost_function = costs.CategoricalCrossEntropyCost( distribution_axis, 0.1, 0.1 )
        learning_rate = learning_rates.RMSPropLearningRate( 0.001, 0.9 )
        parameters = optimisation.Parameters( weight_L1 = 0.1, weight_L2 = 0.1 )
        return optimisation.StochasticGradientDescent(
            parameters, cost_function, learning_rate, log )

    


#---------------------------------------------------------------------------------------------------

class RealDataTests( unittest.TestCase ):



    @staticmethod
    def train_simple_cnn_on_oasis( input_path, output_path ):

        log = output.Log( sys.stdout )
        definition = RealDataExperiment()
        experiment = experiments.SegmentationByDenseInferenceExperiment(
            definition,
            input_path,
            output_path,
            log )

        random_generator = numpy.random.RandomState( seed = 54 )
        return experiment.run( random_generator ) 
    

    def test_complete_experiment_stack( self ):

        input_path = os.environ.get( 'REAL_DATA_INPUT_PATH', None )
        output_path = os.environ.get( 'REAL_DATA_OUTPUT_PATH', None )
        if input_path  and output_path:
            RealDataTests.train_simple_cnn_on_oasis( input_path, output_path )

        


#---------------------------------------------------------------------------------------------------

if __name__ == '__main__' :

    assert len( sys.argv ) > 2
    oasis_input_path = sys.argv[ 1 ]
    test_output_path = sys.argv[ 2 ]

    RealDataTests.train_simple_cnn_on_oasis( oasis_input_path, test_output_path )
    

#---------------------------------------------------------------------------------------------------
