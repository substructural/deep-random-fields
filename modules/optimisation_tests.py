#===================================================================================================
# optiisation tests


#---------------------------------------------------------------------------------------------------

import collections
import random
import sys
import unittest

import numpy
import theano
import theano.tensor as T 

import costs
import learning_rates
import network
import optimisation
import output


#---------------------------------------------------------------------------------------------------

FloatType = theano.config.floatX


#---------------------------------------------------------------------------------------------------


class CostsTests( unittest.TestCase ):


    def test_that_has_converged_is_false_when_sample_size_is_greater_than_length( self ):

        costs = [ 1.0, 0.5 ]
        parameters = optimisation.Parameters( cost_sample_size = 3, recent_cost_sample_size = 2 )

        self.assertFalse( optimisation.Cost.has_converged( costs, parameters ) )


    def test_that_has_converged_is_false_when_change_is_above_threshold( self ):

        costs = [ 2.0 ** (-i) for i in range( 4 ) ]
        parameters = optimisation.Parameters(
            cost_sample_size = 3,
            recent_cost_sample_size = 2,
            convergence_threshold = 0.1 )

        for i in range( 4 ):
            self.assertFalse( optimisation.Cost.has_converged( costs[ : i ], parameters ) )


    def test_that_has_converged_is_true_when_change_is_below_threshold( self ):

        costs = [ 2.0 ** (-i) for i in range( 5 ) ]
        parameters = optimisation.Parameters(
            cost_sample_size = 3,
            recent_cost_sample_size = 2,
            convergence_threshold = 0.1 )

        self.assertTrue( optimisation.Cost.has_converged( costs[ : 5 ], parameters ) )

        
    def test_that_has_overfit_is_false_when_sample_size_is_greater_than_length( self ):

        costs = [ 1.0, 0.5 ]
        parameters = optimisation.Parameters( cost_sample_size = 3, recent_cost_sample_size = 2 )

        self.assertFalse( optimisation.Cost.has_overfit( costs, parameters ) )


    def test_that_has_overfit_is_false_when_recent_mean_is_less_than_background_mean( self ):

        costs = [ 1.0, 0.8, 0.7, 0.65, 0.7, 0.8 ]
        parameters = optimisation.Parameters(
            cost_sample_size = 3,
            recent_cost_sample_size = 2 )

        for i in range( 5 ):
            self.assertFalse( optimisation.Cost.has_overfit( costs[ : i ], parameters ) )


    def test_that_has_overfit_is_true_when_recent_mean_is_greater_than_background_mean( self ):

        costs = [ 1.0, 0.8, 0.7, 0.65, 0.7, 0.8 ]
        parameters = optimisation.Parameters(
            cost_sample_size = 3,
            recent_cost_sample_size = 2 )

        self.assertTrue( optimisation.Cost.has_overfit( costs[ : 6 ], parameters ) )



#---------------------------------------------------------------------------------------------------

class Mock :

    class LinearLayer( network.Layer ) :

        def __init__( self, w, b ) :
            self.initial_w = float( w )
            self.initial_b = float( b )

        def graph( self, parameters, inputs ) :
            return parameters[ 0 ] * inputs + parameters[ 1 ]

        def initial_parameter_values( self ) :
            return [ ( 'W', self.initial_w ), ( 'b', self.initial_b ) ]

        
    class SimpleDifferenceCostFunction( object ) :
            
        def __call__( self, outputs, labels, parameters = None ) :
            return T.mean( outputs - labels )


    class MeanSquaredDifferenceCost( costs.CostFunction ):

        def cost( self, outputs, labels ):
            return T.mean( ( outputs - labels ) ** 2 )


    class MonitorWhichRecordsCallsMadeToIt( optimisation.Monitor ):

        def __init__( self, verbosity = 0 ):
            self.on_batch_arguments = []
            self.on_epoch_arguments = []
            self.verbosity = verbosity

        def on_batch( self, epoch, batch, predicted_distribution, reference_distribution, offsets ):
            self.on_batch_arguments.append((
                epoch, batch, predicted_distribution, reference_distribution, offsets ))
            if self.verbosity > 1:
                print( self.on_batch_arguments[ -1 ] )

        def on_epoch( self, epoch, mean_cost, model ):
            self.on_epoch_arguments.append(( epoch, mean_cost, model ))
            if self.verbosity > 0:
                print( self.on_epoch_arguments[ -1 ] )


#---------------------------------------------------------------------------------------------------


class OptimiserUnderTest( optimisation.Optimiser ):

    def __init__( self, update_functor ):
        self.__update_functor = update_functor
        simple_difference_cost = Mock.SimpleDifferenceCostFunction()
        optimisation_parameters = optimisation.Parameters()
        super( OptimiserUnderTest, self ).__init__(
            simple_difference_cost, optimisation_parameters )

    def updates( self, model, cost ):
        return self.__update_functor( model, cost )


class OptimiserWithMockSteps( optimisation.Optimiser ):

    Step = collections.namedtuple( 'Step', [ 'step', 'step_name', 'epoch', 'data', 'monitor' ] )

    def __init__( self, costs_per_epoch, parameters ):
        self.steps = []
        self.costs = costs_per_epoch
        simple_difference = Mock.SimpleDifferenceCostFunction()
        super( OptimiserWithMockSteps, self ).__init__( simple_difference, parameters )

    def updates( self, model, cost ):
        raise Exception( "The optimisation step for this class does not update the model.")

    def optimisation_step( self, model ):
        return "optimisation_step"

    def validation_step( self, model ):
        return "validation_step"

    def iterate( self, step, step_name, epoch, data, monitor, model ):
        self.steps.append(
            OptimiserWithMockSteps.Step( step, step_name, epoch, data, monitor ) )
        return self.costs[ epoch ]


#---------------------------------------------------------------------------------------------------


class OptimiserTests( unittest.TestCase ):
    

    def test_that_optimisation_step_computes_the_model_prediction_function_and_its_cost( self ):

        model = network.Model( network.Architecture( [ Mock.LinearLayer( 2.0, 0.0 ) ], 1, 1 ) )
        updates = lambda model, cost : (
            [ ( p, p - 0.5 * cost ) for layer in model.parameters for p in layer ] ) 

        optimiser = OptimiserUnderTest( updates )
        optimisation_step = optimiser.optimisation_step( model )

        inputs           = [ 2.0, 3.0, 4.0 ]
        target_outputs   = [ 3.0, 4.5, 6.0 ]
        expected_outputs = [ 4.0, 6.0, 8.0 ]
        expected_cost    = 1.5 # mean([ (2*2)-3 + (2*3)-4.5 + (2*4)-6 ])

        outputs, cost = optimisation_step( inputs, target_outputs )
        self.assertEqual( cost, expected_cost )
        self.assertEqual( list( outputs ), expected_outputs )
    

    def test_that_optimisation_step_applies_the_correct_updates( self ):

        model = network.Model( network.Architecture( [ Mock.LinearLayer( 2.0, 0.0 ) ], 1, 1 ) )
        updates = lambda model, cost : (
            [ ( p, p - 0.5 * cost ) for layer in model.parameters for p in layer ] ) 

        optimiser = OptimiserUnderTest( updates )
        optimisation_step = optimiser.optimisation_step( model )

        inputs         = [ 2.0, 3.0, 4.0 ]
        target_outputs = [ 3.0, 4.5, 6.0 ]
        expected_cost  = 1.5 # mean([ (2*2)-3 + (2*3)-4.5 + (2*4)-6 ])

        optimisation_step( inputs, target_outputs )

        updated_weights = model.parameter_values
        expected_weights = numpy.array( [[ p - 0.5 * expected_cost for p in [ 2.0, 0.0 ] ]] )
        errors = numpy.abs( expected_weights - updated_weights )
        failures = errors > 0
        self.assertFalse( numpy.any( failures ) )


    def test_that_validation_step_computes_the_model_prediction_function_and_its_cost( self ):

        model = network.Model( network.Architecture( [ Mock.LinearLayer( 2.0, 0.0 ) ], 1, 1 ) )
        updates = lambda model, cost : (
            [ ( p, p - 0.5 * cost ) for layer in model.parameters for p in layer ] ) 

        optimiser = OptimiserUnderTest( updates )
        validation_step = optimiser.validation_step( model )

        inputs           = [ 2.0, 3.0, 4.0 ]
        target_outputs   = [ 3.0, 4.5, 6.0 ]
        expected_outputs = [ 4.0, 6.0, 8.0 ]
        expected_cost    = 1.5 # mean([ 4-3 + 6-4.5 + 8-6 ])

        outputs, cost = validation_step( inputs, target_outputs )
        self.assertEqual( cost, expected_cost )
        self.assertEqual( list( outputs ), expected_outputs )


    def test_that_validation_step_applies_no_update_to_the_parameters( self ):

        model = network.Model( network.Architecture( [ Mock.LinearLayer( 2.0, 0.0 ) ], 1, 1 ) )
        updates = lambda model, cost : (
            [ ( p, p - 0.5 * cost ) for layer in model.parameters for p in layer ] ) 

        optimiser = OptimiserUnderTest( updates )
        validation_step = optimiser.validation_step( model )

        inputs         = [ 2.0, 3.0, 4.0 ]
        target_outputs = [ 3.0, 4.5, 6.0 ]

        weights_before_computation = model.parameter_values
        validation_step( inputs, target_outputs )

        weights_after_computation = model.parameter_values
        differences = numpy.abs( weights_after_computation - weights_before_computation )
        failures = differences > 0
        self.assertFalse( numpy.any( failures ) )


    def test_that_iterate_passes_correct_results_to_the_monitor( self ):

        model = network.Model( network.Architecture( [ Mock.LinearLayer( 2.0, 0.0 ) ], 1, 1 ) )
        updates = lambda model, cost : (
            [ ( p, p - 0.1 ) for layer in model.parameters for p in layer ] ) 

        optimiser = OptimiserUnderTest( updates )
        validation_step = optimiser.validation_step( model )

        monitor = Mock.MonitorWhichRecordsCallsMadeToIt()
        data = [ ( [2.0], [3.0], [(1,0)] ), ( [3.0], [4.5], [(1,1)] ), ( [4.0], [6.0], [(2,0)] ) ]
        mean_cost = optimiser.iterate( validation_step, "validation", 42, data, monitor, model )
        
        recorded_positions = numpy.array( [ p for (e, b, pd, rd, p) in monitor.on_batch_arguments ] )
        expected_positions = numpy.array( [ p for (x, y, p) in data ] )
        self.assertTrue( numpy.array_equal( recorded_positions, expected_positions ) )
        
        recorded_predictions = numpy.array( [ pd for ( e, b, pd, rd, p ) in monitor.on_batch_arguments ] )
        expected_predictions = numpy.array( [ [ 2 * x[0] ] for ( x, y, p ) in data ] )
        self.assertTrue( numpy.array_equal( recorded_predictions, expected_predictions ) )

        recorded_distribution = numpy.array( [ rd for ( e, b, pd, rd, p ) in monitor.on_batch_arguments ] )
        expected_distribution = numpy.array( [ rd for ( x, rd, p ) in data ] )
        self.assertTrue( numpy.array_equal( recorded_distribution, expected_distribution ) )

        recorded_mean_costs = [ c for ( e, c, m ) in monitor.on_epoch_arguments ]
        expected_mean_cost = 1.5 # = 1/3 * ( 3-2 + 4.5-3 + 6-4 )
        self.assertEqual( mean_cost, expected_mean_cost )
        self.assertEqual( len( recorded_mean_costs ), 1 )
        self.assertEqual( recorded_mean_costs[ 0 ], mean_cost )


    def test_that_optimise_until_converged_passes_the_correct_functor_data_and_monitor( self ):

        model = 'model'
        costs = [ ( e % 2 ) * 1e3 for e in range( 10 ) ]

        validation_monitor = 'validation_monitor'
        validation_data = 42

        optimisation_monitor = 'optimisation_monitor'
        optimisation_data = 6 * 9

        parameters = optimisation.Parameters(
            maximum_epochs = 10,
            cost_sample_size = 4,
            recent_cost_sample_size = 2 )
        
        optimiser = OptimiserWithMockSteps( costs, parameters )
        optimiser.optimise_until_converged(
            model, optimisation_data, validation_data, optimisation_monitor, validation_monitor )

        self.assertEqual( len( optimiser.steps ), 20 )

        for e in range( 10 ):
            self.assertEqual( optimiser.steps[ e * 2 ].step, 'optimisation_step' )
            self.assertEqual( optimiser.steps[ e * 2 ].step_name, 'optimisation' )
            self.assertEqual( optimiser.steps[ e * 2 ].epoch, e )
            self.assertEqual( optimiser.steps[ e * 2 ].data, optimisation_data )
            self.assertEqual( optimiser.steps[ e * 2 ].monitor, optimisation_monitor )

        for e in range( 10 ):
            self.assertEqual( optimiser.steps[ ( e * 2 ) + 1 ].step, 'validation_step' )
            self.assertEqual( optimiser.steps[ ( e * 2 ) + 1 ].step_name, 'validation' )
            self.assertEqual( optimiser.steps[ ( e * 2 ) + 1 ].epoch, e )
            self.assertEqual( optimiser.steps[ ( e * 2 ) + 1 ].data, validation_data )
            self.assertEqual( optimiser.steps[ ( e * 2 ) + 1 ].monitor, validation_monitor )


    def test_that_optimise_until_converged_terminates_when_the_model_has_converged( self ):

        model = 'model'
        costs = [ 1.0 / float( e + 1 ) for e in range( 10 ) ]

        parameters = optimisation.Parameters(
            maximum_epochs = 10,
            cost_sample_size = 3,
            recent_cost_sample_size = 2,
            convergence_threshold = 0.05 )

        optimiser = OptimiserWithMockSteps( costs, parameters )
        optimiser.optimise_until_converged( model, None, None, None, None )

        self.assertEqual( len( optimiser.steps ), 5 * 2 )

            
                               
#---------------------------------------------------------------------------------------------------


class StochasticGradientDescentTests( unittest.TestCase ):


    def test_that_gradient_descent_converges_on_an_approximate_solution_for_a_simple_task( self ):

        model = network.Model( network.Architecture( [ Mock.LinearLayer( 0.5, 0.5 ) ], 1, 1 ) )

        parameters = optimisation.Parameters(
            weight_L1 = 1.0,
            weight_L2 = 1.0,
            learning_rate = 0.05,
            maximum_epochs = 1000,
            cost_sample_size = 5,
            recent_cost_sample_size = 3,
            convergence_threshold = 0.00001 )
            
        log = output.Log( sys.stdout )
        cost_function = Mock.MeanSquaredDifferenceCost( weight_L1 = 0.1, weight_L2 = 0.1 )
        learning_rate_schedule = learning_rates.RMSPropLearningRate( 0.01, 0.9 )
        optimiser = optimisation.StochasticGradientDescent(
            cost_function, parameters, learning_rate_schedule, log )
        
        r = lambda : random.randrange( -100, 100, 1 ) * 0.05
        n = lambda : random.randrange( -100, 100, 1 ) * 0.0001
        xs = numpy.array([ [ r() for i in range( 0, 20 ) ] for j in range( 0, 10 ) ])
        ys = numpy.array([ [ n() + ( 2 * x + 1 ) for x in xs_i ] for xs_i in xs ])
        ps = numpy.array([ [ (j, i) for i in range( 0, 20 ) ] for j in range( 0, 10 ) ])

        optimisation_data = list( zip( xs, ys, ps ) )
        optimisation_monitor = Mock.MonitorWhichRecordsCallsMadeToIt( verbosity = 0 )

        validation_data = optimisation_data
        validation_monitor = Mock.MonitorWhichRecordsCallsMadeToIt( verbosity = 0 )

        optimiser.optimise_until_converged(
            model, optimisation_data, validation_data, optimisation_monitor, validation_monitor )

        cs = numpy.array( [ model.predict( x ) for x in xs ] )
        es = numpy.array( [ ys[ i ] - cs[ i ] for i in range( len( ys ) ) ] )

        log.subsection( "results" )
        log.entry( "computed:  "  + str( cs ) )
        log.entry( "expected:  "  + str( ys ) )
        log.entry( "errors:    "  + str( es ) )
        log.entry( "parameters:" )
        log.record( dict( model.parameter_names_and_values[ 0 ] ) )
        


#---------------------------------------------------------------------------------------------------
