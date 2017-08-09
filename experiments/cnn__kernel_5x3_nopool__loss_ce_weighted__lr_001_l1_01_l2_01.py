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
    def model():

        leaky_relu = activation.LeakyRectifiedLinearUnit( 0.1 )
        architecture = network.Architecture(
            [ layers.ConvolutionalLayer(  1, 16, ( 5, 5, 5 ), 1, leaky_relu ),
              layers.ConvolutionalLayer( 16, 32, ( 5, 5, 5 ), 1, leaky_relu ),
              layers.ConvolutionalLayer( 32,  4, ( 5, 5, 5 ), 1, leaky_relu ),
              layers.Softmax(),
              layers.ScalarFeatureMapsProbabilityDistribution()
            ],
            input_dimensions = 4,
            output_dimensions = 5 )

        return network.Model( architecture, seed = 42 )


    def optimiser( self, dataset, log ):

        distribution_axis = 1
        prior_distribution = data.Normalisation.class_distribution_in_data_subset(
            dataset.training_set, self.label_count, log )

        cost_function = costs.WeightedCategoricalCrossEntropyCost(
            prior_distribution, distribution_axis, weight_L1=0.01, weight_L2=0.01 )

        learning_rate = learning_rates.RMSPropLearningRate( 0.005, 0.9 )
        parameters = optimisation.Parameters( maximum_epochs=2 )
        return optimisation.StochasticGradientDescent(
            parameters, cost_function, learning_rate, log )


#---------------------------------------------------------------------------------------------------
        
