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
                 .with_patch_count( 2000 )
                 .with_patch_stride( 10 ) )


    @staticmethod
    def architecture():

        leaky_relu = activation.LeakyRectifiedLinearUnit( 0.1 )
        convolution = lambda i, o, k : layers.ConvolutionalLayer(
            i, o, k, 1, leaky_relu, uniform_weights = False, orthogonal_weights = False )

        return network.Architecture(
            [ convolution(  1, 20, ( 5, 5, 5 ) ),
              convolution( 20, 40, ( 5, 5, 5 ) ),
              convolution( 40,  4, ( 5, 5, 5 ) ),
              layers.Softmax(),
              layers.ScalarFeatureMapsProbabilityDistribution()
            ],
            input_dimensions = 4,
            output_dimensions = 5 )


    def optimiser( self, dataset, log ):

        distribution_axis = 1
        cost_function = costs.CategoricalCrossEntropyCost(
            distribution_axis, weight_L1=0.005, weight_L2=0.0 )

        learning_rate = learning_rates.RMSPropLearningRate( 0.001, 0.9 )
        parameters = optimisation.Parameters( maximum_epochs = 5 )
        return optimisation.StochasticGradientDescent(
            parameters, cost_function, learning_rate, log )


#---------------------------------------------------------------------------------------------------
        
