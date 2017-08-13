#===================================================================================================
# network frameowrk tests


#---------------------------------------------------------------------------------------------------

import unittest
import numpy
import theano 
import theano.tensor as T

import network


#---------------------------------------------------------------------------------------------------

class Mock:

    class Layer( network.Layer ) :

        def __init__( self, i ) :
            self.initial_value = float( i )

        def graph( self, parameters, inputs ) :
            return parameters[ 0 ] * inputs

        def initial_parameter_values( self ) :
            return [ ( 'W', self.initial_value ), ( 'b', 42.0 ) ]

        @property
        def parameter_names( self ):

            return [ 'W', 'b' ]

        
#---------------------------------------------------------------------------------------------------


class ModelTests( unittest.TestCase ) :

             
    def test_that_the_model_parameter_accessors_are_consistent( self ) :

        layer_count = 4
        layers = [ Mock.Layer( i ) for i in range( layer_count ) ]
        architecture = network.Architecture( layers, input_dimensions = 0 )
        model = network.Model( architecture )

        parameter_names_and_values = model.parameter_names_and_values
        parameter_names = model.parameter_names
        parameter_values = model.parameter_values

        self.assertEqual( layer_count, len( parameter_names_and_values ) )
        self.assertEqual( layer_count, len( parameter_names ) )
        self.assertEqual( layer_count, len( parameter_values ) )

        for i in range( layer_count ):
            zipped_names_and_values = list( zip( parameter_names[ i ], parameter_values[ i ] ) )
            self.assertEqual( parameter_names_and_values[ i ], zipped_names_and_values )

             
    def test_that_the_model_computes_the_correct_initial_values_if_not_supplied( self ) :

        seed = 56
        layer_count = 3
        layers = [ Mock.Layer( i ) for i in range( 0, layer_count ) ]
        architecture = network.Architecture( layers, input_dimensions = 0 )

        model = network.Model( architecture, seed = seed  )

        initial_parameter_values = architecture.initial_parameter_values( seed )
        actual_parameters = model.parameter_names_and_values

        self.assertEqual( actual_parameters, initial_parameter_values )

            
    def test_that_the_model_uses_the_learned_parameters_if_supplied( self ) :

        layer_count = 3
        layers = [ Mock.Layer( i ) for i in range( layer_count ) ]
        architecture = network.Architecture( layers, input_dimensions = 0 )

        ws = [ ( 'W', 6.0 * 9 * i ) for i in range( layer_count ) ]
        bs = [ ( 'b', 56.0 ) for i in range( layer_count ) ]

        learned_parameters = {
            f'{i}:{p}' : v
            for i, ( p, v ) in ( list(enumerate(ws)) + list(enumerate(bs)) ) }

        model = network.Model( architecture, learned_parameters )
        actual_parameters = model.parameter_names_and_values
        expected_parameters = [ [ ws[i], bs[i] ] for i in range( layer_count ) ]

        self.assertEqual( actual_parameters, expected_parameters )

            
    def test_that_the_model_computes_the_correct_outputs( self ) :

        layers = [ Mock.Layer( i ) for i in [ 1, 2, 3 ] ]
        architecture = network.Architecture( layers, input_dimensions = 1, output_dimensions = 1 )

        model = network.Model( architecture )

        inputs = numpy.array( [ 2, 3, 5, 7 ] ).astype( theano.config.floatX ) 
        expected_outputs = ( 1 * 2 * 3 ) * inputs
        computed_outputs = model.predict( inputs )

        self.assertTrue( numpy.array_equal( computed_outputs, expected_outputs ) )


#---------------------------------------------------------------------------------------------------
