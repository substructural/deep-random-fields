#===================================================================================================
# experiment tests



#---------------------------------------------------------------------------------------------------

import unittest
from experiment import Experiment
import network
import layer
import activation


#---------------------------------------------------------------------------------------------------

class ExperimentTestHarness( class Experiment ) :

    def __init__(
            self,
            dataset,
            architecture,
            cost_function,
            optimiser,
            training_batch_parameters,
            validation_batch_parameters,
            experiment_parameters,
            expected_labels ) :

        super().__init__(
            dataset,
            architecture,
            cost_function,
            optimiser,
            training_batch_parameters,
            validation_batch_parameters,
            experiment_parameters )

        self.expected_labels = expected_labels
        self.on_batch_events = []
        self.on_epoch_events = []
        
        
    def format_labels( self, patches ) :

        return self.expected_labels


    def on_batch_event( self, batch_index, training_output, training_costs ) :

        self.on_batch_events.append( ( batch_index, training_output, training_costs ) )


    def on_epoch_event( self, epoch_index, validation_output, validation_costs, training_costs ) :

        self.on_epoch_events.append( ( batch_index, validationn_output, validation_costs, training_costs ) )
    


#---------------------------------------------------------------------------------------------------

class BaseExperimentTests( unittest.TestCase ) :

    def test_that_experiment_loads_the_correct_training_batch( self ) :

        layers = [
            layers.DenseLayer( ( 2, 2 ), activation.LeakyRectifiedLinearUnit( 0.1 ) ),
            layers.Softmax() ]
        architecture = network.Architecture( layers, 2, 1 )

        cross_entropy_cost_function = optimiser.CategoricalCrossEntropyCost( 0.1, 0.1 )
        gradient_descent_optimiser = optimiser.SimpleGradientDescentOptimiser( 0.1 )
        experiment = ExperimentTestHarness(
            dataset,
            architecture,
            cross_entropy_cost_function,
            gradient_descent_optimiser,
            training_batch_parameters,
            validation_batch_parameters,
            experiment_parameters,
        )
        


#---------------------------------------------------------------------------------------------------

if __name__ == '__main__' :

    unittest.main()

#---------------------------------------------------------------------------------------------------
