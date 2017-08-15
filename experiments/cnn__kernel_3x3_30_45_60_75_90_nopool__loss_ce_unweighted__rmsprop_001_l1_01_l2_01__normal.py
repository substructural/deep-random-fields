#===================================================================================================
# experiment tests



#---------------------------------------------------------------------------------------------------

import os
import sys
import unittest

import activation
import costs
import data
import experiment
import layers
import learning_rates
import network
import oasis
import optimisation
import output
import sample
import report


#---------------------------------------------------------------------------------------------------


class Definition( experiment.ExperimentDefinition ):


    @property
    def label_count( self ):

        return 4


    @staticmethod
    def dataset( input_path, log ):

        return oasis.OasisDataSet( input_path, 100, 50, 250, 42, maybe_log = log )


    @property
    def sample_parameters( self ):

        return ( sample.Parameters()
                 .with_volume_count( 10 )
                 .with_window_margin( 6 )
                 .with_target_shape(( 172, 202, 162 ))
                 .with_patch_shape(( 22, 22, 22 ))
                 .with_patch_count( 500 )
                 .with_patch_stride( 10 ) )


    @staticmethod
    def architecture():

        leaky_relu = activation.LeakyRectifiedLinearUnit( 0.1 )
        convolution = lambda i, o, k : layers.ConvolutionalLayer(
            i, o, k, 1, leaky_relu, uniform_weights = False, orthogonal_weights = False )

        return network.Architecture(
            [ convolution(  1, 30, ( 3, 3, 3 ) ),
              convolution( 30, 45, ( 3, 3, 3 ) ),
              convolution( 45, 60, ( 3, 3, 3 ) ),
              convolution( 60, 75, ( 3, 3, 3 ) ),
              convolution( 75, 90, ( 3, 3, 3 ) ),
              convolution( 90,  4, ( 3, 3, 3 ) ),
              layers.Softmax(),
              layers.ScalarFeatureMapsProbabilityDistribution()
            ],
            input_dimensions = 4,
            output_dimensions = 5 )


    def optimiser( self, dataset, log ):

        distribution_axis = 1
        cost_function = costs.CategoricalCrossEntropyCost(
            distribution_axis, weight_L1=0.01, weight_L2=0.01 )

        learning_rate = learning_rates.RMSPropLearningRate( 0.001, 0.9 )
        parameters = optimisation.Parameters( maximum_epochs = 15 )
        return optimisation.StochasticGradientDescent(
            parameters, cost_function, learning_rate, log )


#---------------------------------------------------------------------------------------------------
        
