#===================================================================================================
# network frameowrk tests


#---------------------------------------------------------------------------------------------------

import unittest
import numpy
import theano 
import theano.tensor as T

import network

import pdb


#---------------------------------------------------------------------------------------------------

class Mock :

    class Layer( network.Layer ) :

        def __init__( self, i ) :
            self.initial_value = float( i )

        def graph( self, parameters, inputs ) :
            return parameters[ 0 ] * inputs

        def initial_parameter_values( self ) :
            return [ ( 'W', self.initial_value ), ( 'b', 42.0 ) ]

        
    class CostFunction( object ) :

        def __init__( self, output_shape = (1, ) ) :
            self.output_shape = output_shape
            
        def __call__( self, outputs, labels, parameters ) :
            return T.sum( ( outputs - labels ) ** 2 )


    def FixedOffsetOptimiser( parameter, cost ) :

        return parameter + 1.0 + ( cost / 1000.0 )


    def SimpleGradientOptimiser( parameter, cost ) :

        return parameter - 0.5 * T.grad( cost, wrt=parameter, disconnected_inputs='ignore' )


    class ModelWithPresetOutputs( object ) :

        def __init__(
                self,
                training_outputs,
                training_costs,
                validation_outputs,
                validation_costs ) :

            self.training_costs = training_costs
            self.training_outputs = training_outputs
            self.validation_costs = validation_costs
            self.validation_outputs = validation_outputs

            self.batches_per_epoch = len( training_outputs[ 0 ] )

            self.epoch = 0
            self.batch = 0
            self.inputs_passed_to_optimise = []
            self.labels_passed_to_optimise = []
            self.inputs_passed_to_validate = []
            self.labels_passed_to_validate = []


        def optimise( self, training_inputs, training_labels ) :

            self.inputs_passed_to_optimise.append( training_inputs.tolist() )
            self.labels_passed_to_optimise.append( training_labels.tolist() )

            i = self.epoch
            j = self.batch

            if self.batch + 1 == self.batches_per_epoch :
                self.batch = 0
                self.epoch += 1
            else :
                self.batch += 1

            return ( self.training_outputs[ i ][ j ], self.training_costs[ i ][ j ] )
            

        def validate( self, validation_inputs, validation_labels ) :

            self.inputs_passed_to_validate.append( validation_inputs.tolist() )
            self.labels_passed_to_validate.append( validation_labels.tolist() )

            assert( self.epoch > 0 ) # validation for an epoch is called after training for the epoch completes
            return self.validation_outputs[ self.epoch - 1 ], self.validation_costs[ self.epoch - 1 ]
    

        
    

#---------------------------------------------------------------------------------------------------

def parameter_values( model ) :

    return [
        [ ( p.name, float( p.get_value() ) ) for p in parameter_set ]
        for parameter_set in model.parameters ]
            

#---------------------------------------------------------------------------------------------------

class ModelTests( unittest.TestCase ) :

             
    def test_that_the_model_computes_the_correct_initial_values_if_not_supplied( self ) :

        seed = 56
        layer_count = 3
        layers = [ Mock.Layer( i ) for i in range( 0, layer_count ) ]
        architecture = network.Architecture( layers, input_dimensions = 0 )

        model = network.Model( architecture, Mock.CostFunction(), Mock.FixedOffsetOptimiser, seed = seed  )

        initial_parameter_values = architecture.initial_parameter_values( seed )
        actual_parameters = parameter_values( model )
        self.assertEqual( actual_parameters, initial_parameter_values )

            
    def test_that_the_model_uses_the_learned_parameters_if_supplied( self ) :

        layer_count = 3
        layers = [ Mock.Layer( i ) for i in range( 0, layer_count ) ]
        architecture = network.Architecture( layers, input_dimensions = 0 )
        learned_parameters = [ [ ( 'W', 6.0 * 9 * i ), ( 'b', 56.0 ) ] for i in range( 0, layer_count ) ]

        model = network.Model(
            architecture,
            Mock.CostFunction(),
            Mock.FixedOffsetOptimiser,
            learned_parameters )

        actual_parameters = parameter_values( model )
        self.assertEqual( actual_parameters, learned_parameters )

            
    def test_that_the_model_computes_the_correct_outputs( self ) :

        layers = [ Mock.Layer( i ) for i in [ 1, 2, 3 ] ]
        architecture = network.Architecture( layers, input_dimensions = 1, output_dimensions = 1 )

        model = network.Model( architecture, Mock.CostFunction(), Mock.FixedOffsetOptimiser )

        inputs = numpy.array( [ 2, 3, 5, 7 ] ).astype( theano.config.floatX ) 
        labels = numpy.array( [ 6, 9, 15, 21 ] ).astype( theano.config.floatX ) 
        expected_outputs = ( 1 * 2 * 3 ) * inputs

        validation_outputs, validation_costs = model.validate( inputs, labels )
        self.assertTrue( numpy.array_equal( validation_outputs, expected_outputs ) )

        training_outputs, training_costs = model.optimise( inputs, labels )
        self.assertTrue( numpy.array_equal( training_outputs, expected_outputs ) )

            
    def test_that_the_model_computes_the_correct_cost( self ) :

        layers = [ Mock.Layer( i ) for i in [ 1, 2, 3 ] ]
        architecture = network.Architecture( layers, input_dimensions = 1, output_dimensions = 1 )
        cost_function = Mock.CostFunction()

        model = network.Model( architecture, cost_function, Mock.FixedOffsetOptimiser )

        inputs = numpy.array( [ 2, 3, 5, 7 ] ).astype( theano.config.floatX ) 
        labels = numpy.array( [ 6, 9, 15, 21 ] ).astype( theano.config.floatX ) 
        expected_outputs = ( 1 * 2 * 3 ) * inputs
        expected_cost = sum( ( expected_outputs - labels ) ** 2 )

        validation_outputs, validation_cost = model.validate( inputs, labels )
        training_outputs, training_cost = model.optimise( inputs, labels )

        self.assertEqual( validation_cost, expected_cost )
        self.assertEqual( training_cost, expected_cost )

            
    def test_that_the_model_parameter_updates_are_set_by_the_optimiser( self ) :

        layers = [ Mock.Layer( i ) for i in [ 1, 2, 3 ] ]
        architecture = network.Architecture( layers, input_dimensions = 1, output_dimensions = 1 )
        cost_function = Mock.CostFunction()
        initial_parameters = [ [ ( 'W', float( i ) ), ( 'b', 56.0 ) ] for i in [ 0.5, 2.1, 3.5 ] ]

        model = network.Model( architecture, cost_function, Mock.FixedOffsetOptimiser, initial_parameters )

        for i in range( 0, len( layers ) ) :
            for j in range( 0, 2 ) :
                expected_value = initial_parameters[ i ][ j ][ 1 ]
                actual_value = model.parameters[ i ][ j ].get_value()
                self.assertEqual( actual_value, expected_value )

        inputs = numpy.array( [ 2, 3, 5, 7 ] ).astype( theano.config.floatX ) 
        labels = numpy.array( [ 6, 9, 15, 21 ] ).astype( theano.config.floatX ) 
        outputs, cost = model.validate( inputs, labels )

        model.optimise( inputs, labels )

        for i in range( 0, len( layers ) ) :
            for j in range( 0, 2 ) :
                initial_value = initial_parameters[ i ][ j ][ 1 ]
                expected_value = ( initial_value + 1.0 ) + ( cost / 1000.0 )
                actual_value = model.parameters[ i ][ j ].get_value()
                error = abs( expected_value - actual_value )
                tolerance = 1e-5
                self.assertTrue( error < 1e-5 )

#---------------------------------------------------------------------------------------------------

class TrainingTests( unittest.TestCase ) :


    def test_that_train_for_epoch_performs_the_expected_sequence_of_optimisations( self ) :

        training_inputs = [ 1, 2, 3 ]
        training_labels = [ 2, 3, 5 ]

        mock_model = Mock.ModelWithPresetOutputs(
            training_outputs   = [ [ 100, 101, 102 ], [ 200, 201, 202 ] ],
            training_costs     = [ [ 0.9, 0.8, 0.7 ], [ 0.5, 0.4, 0.3 ] ],
            validation_outputs = [ 103, 203 ],
            validation_costs   = [ 1.0, 0.6 ] )

        def load_training_set( i ) :
            return ( numpy.array( training_inputs[ i ] ), numpy.array( training_labels[ i ] ) )

        class on_batch_event :

            training_outputs_passed_to_event = []
            training_costs_passed_to_event = []

            @staticmethod
            def callback( i, training_output, training_costs ) :
                on_batch_event.training_outputs_passed_to_event.append( training_output ) 
                on_batch_event.training_costs_passed_to_event.append( training_costs )

        costs_for_epoch = network.train_for_epoch(
            mock_model,
            load_training_set,
            mock_model.batches_per_epoch,
            on_batch_event.callback )

        self.assertEqual( costs_for_epoch, mock_model.training_costs[ 0 ] )
        self.assertEqual( on_batch_event.training_costs_passed_to_event, mock_model.training_costs[ 0 ] )
        self.assertEqual( on_batch_event.training_outputs_passed_to_event, mock_model.training_outputs[ 0 ] )
        self.assertEqual( mock_model.inputs_passed_to_optimise, training_inputs )
        self.assertEqual( mock_model.labels_passed_to_optimise, training_labels )


    def test_that_train_terminates_on_convergence( self ) :

        training_inputs = [ 1, 2, 3 ]
        training_labels = [ 2, 3, 5 ]
        validation_inputs = [ 4, 5, 6, 7 ]
        validation_labels = [ 7, 11, 13, 17 ]

        mock_model = Mock.ModelWithPresetOutputs(
            training_outputs   = [ [ 100, 101, 102 ], [ 200, 201, 202 ], [ 300, 301, 302 ], [ 400, 401, 402 ] ],
            training_costs     = [ [ 2.5, 2.0, 1.5 ], [ 0.8, 0.6, 0.4 ], [ 0.3, 0.2, 0.1 ], [ 0.08, 0.06, 0.04 ] ],
            validation_outputs = [ 103, 203, 303, 404 ],
            validation_costs   = [ 0.5, 0.2, 0.1, 0.5 ] )

        def load_training_set( j ) : 
            return ( numpy.array( training_inputs[ j ] ), numpy.array( training_labels[ j ] ) )

        def load_validation_set( i ) :
            return ( numpy.array( validation_inputs[ i ] ), numpy.array( validation_labels[ i ] ) )

        class on_epoch_event :

            validation_outputs_passed_to_event = []
            validation_costs_passed_to_event = []
            training_costs_passed_to_event = []

            @staticmethod
            def callback( i, validation_output, validation_costs, training_costs ) :
                on_epoch_event.validation_outputs_passed_to_event.append( validation_output ) 
                on_epoch_event.validation_costs_passed_to_event.append( validation_costs )
                on_epoch_event.training_costs_passed_to_event.append( training_costs )

        validation_output, validation_cost, training_costs = network.train(
            mock_model,
            load_training_set,
            load_validation_set,
            cost_threshold_for_convergence = 0.2,
            tail_length_for_convergence = 2,
            batch_count = mock_model.batches_per_epoch,
            epoch_count = len( validation_labels ),
            on_batch_event = ( lambda i, o, c : None ),
            on_epoch_event = on_epoch_event.callback )

        self.assertEqual( validation_output, mock_model.validation_outputs[ 2 ] )
        self.assertEqual( validation_cost, mock_model.validation_costs[ 0 : 3 ] )
        self.assertEqual( on_epoch_event.validation_outputs_passed_to_event, mock_model.validation_outputs[ 0 : 3 ] )
        self.assertEqual( on_epoch_event.validation_costs_passed_to_event, mock_model.validation_costs[ 0 : 3 ] )
        self.assertEqual( on_epoch_event.training_costs_passed_to_event, mock_model.training_costs[ 0 : 3 ] )
        self.assertEqual( mock_model.inputs_passed_to_validate, validation_inputs[ 0 : 3 ] )
        self.assertEqual( mock_model.labels_passed_to_validate, validation_labels[ 0 : 3 ] )



    def test_that_train_terminates_on_overfitting( self ) :

        training_inputs = [ 1, 2, 3 ]
        training_labels = [ 2, 3, 5 ]
        validation_inputs = [ 4, 5, 6, 7 ]
        validation_labels = [ 7, 11, 13, 17 ]

        mock_model = Mock.ModelWithPresetOutputs(
            training_outputs   = [ [ 100, 101, 102 ], [ 200, 201, 202 ], [ 300, 301, 302 ], [ 400, 401, 402 ] ],
            training_costs     = [ [ 2.5, 2.0, 1.5 ], [ 0.8, 0.6, 0.4 ], [ 0.3, 0.2, 0.1 ], [ 0.08, 0.06, 0.04 ] ],
            validation_outputs = [ 103, 203, 303, 404 ],
            validation_costs   = [ 0.5, 0.2, 0.4, 0.5 ] )

        def load_training_set( j ) : 
            return ( numpy.array( training_inputs[ j ] ), numpy.array( training_labels[ j ] ) )

        def load_validation_set( i ) :
            return ( numpy.array( validation_inputs[ i ] ), numpy.array( validation_labels[ i ] ) )

        class on_epoch_event :

            validation_outputs_passed_to_event = []
            validation_costs_passed_to_event = []
            training_costs_passed_to_event = []

            @staticmethod
            def callback( i, validation_output, validation_costs, training_costs ) :
                on_epoch_event.validation_outputs_passed_to_event.append( validation_output ) 
                on_epoch_event.validation_costs_passed_to_event.append( validation_costs )
                on_epoch_event.training_costs_passed_to_event.append( training_costs )

        validation_output, validation_cost, training_costs = network.train(
            mock_model,
            load_training_set,
            load_validation_set,
            cost_threshold_for_convergence = 0.2,
            tail_length_for_convergence = 2,
            tail_length_for_overfitting = 2,
            batch_count = mock_model.batches_per_epoch,
            epoch_count = len( validation_labels ),
            on_batch_event = ( lambda i, o, c : None ),
            on_epoch_event = on_epoch_event.callback )

        self.assertEqual( validation_output, mock_model.validation_outputs[ 2 ] )
        self.assertEqual( validation_cost, mock_model.validation_costs[ 0 : 3 ] )
        self.assertEqual( on_epoch_event.validation_outputs_passed_to_event, mock_model.validation_outputs[ 0 : 3 ] )
        self.assertEqual( on_epoch_event.validation_costs_passed_to_event, mock_model.validation_costs[ 0 : 3 ] )
        self.assertEqual( on_epoch_event.training_costs_passed_to_event, mock_model.training_costs[ 0 : 3 ] )
        self.assertEqual( mock_model.inputs_passed_to_validate, validation_inputs[ 0 : 3 ] )
        self.assertEqual( mock_model.labels_passed_to_validate, validation_labels[ 0 : 3 ] )



#---------------------------------------------------------------------------------------------------

if __name__ == '__main__' :

    unittest.main()


#---------------------------------------------------------------------------------------------------
