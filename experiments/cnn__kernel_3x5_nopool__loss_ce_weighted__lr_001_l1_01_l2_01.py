#===================================================================================================
# experiment tests



#---------------------------------------------------------------------------------------------------

import os
import sys
import unittest

import activation
import costs
import data
import experiments
import layers
import learning_rates
import network
import oasis
import optimisation
import output
import sample
import report


#---------------------------------------------------------------------------------------------------


class Definition( experiments.ExperimentDefinition ):


    @property
    def experiment_id( self ):

        return 'cnn__kernel_3x5_nopool__loss_ce_weighted__lr_001_l1_01_l2_01'


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
                 .with_window_margin( 5 )
                 .with_target_shape(( 170, 200, 160 ))
                 .with_patch_shape(( 20, 20, 20 ))
                 .with_patch_count( 1000 )
                 .with_patch_stride( 10 ) )


    @staticmethod
    def model():

        leaky_relu = activation.LeakyRectifiedLinearUnit( 0.1 )
        architecture = network.Architecture(
            [ layers.ConvolutionalLayer(  1, 16, ( 3, 3, 3 ), 1, leaky_relu ),
              layers.ConvolutionalLayer( 16, 16, ( 3, 3, 3 ), 1, leaky_relu ),
              layers.ConvolutionalLayer( 16, 32, ( 3, 3, 3 ), 1, leaky_relu ),
              layers.ConvolutionalLayer( 32, 32, ( 3, 3, 3 ), 1, leaky_relu ),
              layers.ConvolutionalLayer( 32,  4, ( 3, 3, 3 ), 1, leaky_relu ),
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

        learning_rate = learning_rates.RMSPropLearningRate( 1e-3, 0.9 )
        parameters = optimisation.Parameters( maximum_epochs=1 )
        return optimisation.StochasticGradientDescent(
            parameters, cost_function, learning_rate, log )


#---------------------------------------------------------------------------------------------------
        
