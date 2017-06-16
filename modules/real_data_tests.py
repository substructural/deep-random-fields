#===================================================================================================
# experiment tests



#---------------------------------------------------------------------------------------------------

import os
import sys
import tempfile
import unittest

import numpy

import activation
import data
import experiment
import labels
import layers
import network
import oasis
import optimisation


#---------------------------------------------------------------------------------------------------

class RealDataTests( unittest.TestCase ):


    @staticmethod
    def train_simple_cnn_on_oasis( input_path, output_path ):

        leaky_relu = activation.LeakyRectifiedLinearUnit( 0.1 )
        architecture = network.Architecture(
            [ layers.ConvolutionalLayer( 1, 8, ( 3, 3, 3 ), 1, leaky_relu ),
              layers.PoolingLayer( layers.PoolingType.MAX,  ( 2, 2, 2 ) ),
              layers.ConvolutionalLayer( 8, 8, ( 3, 3, 3 ), 1, leaky_relu ),
              layers.PoolingLayer( layers.PoolingType.MAX,  ( 2, 2, 2 ) ),
              layers.ConvolutionalLayer( 8, 8, ( 3, 3, 3 ), 1, leaky_relu ),
              layers.PoolingLayer( layers.PoolingType.MAX,  ( 2, 2, 2 ) ),
              layers.DenseLayer( 8, 3, leaky_relu ),
              layers.Softmax() ],
            input_dimensions = 4,
            output_dimensions = 2 )

        distribution_axis = 1
        cost_function = optimisation.CategoricalCrossEntropyCost( distribution_axis, 0.1, 0.1 )
        optimiser = optimisation.SimpleGradientDescentOptimiser( 0.1 )
        model = network.Model( architecture, cost_function, optimiser )

        label_conversion = labels.SparseLabelConversions( 4 )
        batch_parameters = data.Parameters( 4, ( 22, 22, 22 ), 1 )
        oasis_dataset = oasis.OasisDataSet( input_path, 8, 1, 1, 42 )

        experiment_parameters = experiment.Parameters( "real-data-test", output_path, 2, 0.0001, 4 )
        test_run = experiment.Experiment(
            model,
            label_conversion,
            oasis_dataset,
            batch_parameters,
            experiment_parameters )

        return test_run.run() 
    

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
