#===================================================================================================
# network

'''
A namespace for the classes which implement the core components of a neural network

'''


import theano as T
import theano.tensor

import numpy as N
import numpy.random

import output

import ipdb

#---------------------------------------------------------------------------------------------------

def null_function( *args ) : return None

FloatType = T.config.floatX

#---------------------------------------------------------------------------------------------------


class Layer( object ) :

    ''' A single transformation within the network. '''


    def graph( self, parameters, inputs ) :

        raise NotImplementedError()


    def initial_parameter_values( self ) :

        raise NotImplementedError()


    @property
    def parameter_names( self ):

        raise NotImplementedError()



#---------------------------------------------------------------------------------------------------


class Architecture( object ):
    '''
    The graph of transformations, represented as layers, defining the structure of the network.

    '''


    def __init__( self, layers, input_dimensions = 1, output_dimensions = 1 ) :

        assert layers
        self.__layers = layers
        self.__input_dimensions = input_dimensions
        self.__output_dimensions = output_dimensions


    @property
    def input_dimensions( self ) :
        '''
        The expected number of dimensions for the input to the network.

        This number includes any dimensions over which the network operations will be broadcast
        e.g. the batch.

        '''
        return self.__input_dimensions


    @property
    def output_dimensions( self ) :
        '''
        The number of dimensions for the outputs of the network.

        '''
        return self.__output_dimensions


    @property
    def layers( self ) :
        '''
        The sequence of transformations computed by the network.

        '''
        return self.__layers


    def initial_parameter_values( self, seed ) :
        '''
        The intial values for the learnable parameters for each layer in the network.

        '''
        N.random.seed( seed )
        return [ layer.initial_parameter_values() for layer in self.layers ]


    def graph( self, model_parameters, inputs ) :
        '''
        Constructs a computation graph for the network with the specified parameters and inputs.

        '''
        assert( len( model_parameters ) == len( self.layers ) )

        graph = inputs
        for i, layer in enumerate( self.layers ) :

            graph = layer.graph( model_parameters[ i ], graph )

        return graph



#---------------------------------------------------------------------------------------------------


class Model( object ) :
    '''
    A learned parameterisation of a network architecture, trained on labelled data.

    '''


    @staticmethod
    def verify_parameter_values( parameter_values, default_values, log = output.Log() ):

        def assert_equal( a, b, f ):
            if a != b:
                m = f'invalid {f} {a}, expected {b}'
                log.item( m )
                raise Exception( m )

        assert_equal( len( parameter_values ), len( default_values ), f'input length' )

        for i, (learned_layer, default_layer) in enumerate(zip(parameter_values, default_values)):

            assert_equal( len( learned_layer ), len( default_layer ), f'layer {i} length' )

            for ( ln, lv ), ( dn, dv ) in zip( learned_layer, default_layer ):
                assert_equal( ln, dn, f'layer {i}, parameter name' )
                assert_equal( lv.shape, dv.shape, f'layer {i}, parameter {ln} shape' )
                log.item( f'verified layer {i} parameter {ln} {lv.shape}' )

        
    @staticmethod
    def read( learned_parameter_values, architecture, layers = None, log = output.Log() ):

        def name_and_value( i, p ):
            log.item( f'reading layer {i}, parameter {p}' )
            key =  f'{i}:{p}' 
            if key in learned_parameter_values:
                return ( p, learned_parameter_values[ key ] )
            else:
                message =  f'layer {i}, parameter {p}, key {key} is missing' 
                log.item( 'error: ' + message )
                raise Exception( message )

        end_layer = layers if layers else len( architecture.layers ) 
        learned_values = [
            [ name_and_value( i, p ) for p in layer.parameter_names ]
            for i, layer in enumerate( architecture.layers[ 0 : end_layer ] ) ]

        return learned_values


    @staticmethod
    def transfer( existing_model, default_values, architecture, layers, log = output.Log() ):

        transfer_values = Model.read( existing_model, architecture, layers, log )
        combined_values = transfer_values + default_values[ len( transfer_values ) : ]
        log.item( f'transferred {len( transfer_values )} out of {len( combined_values )}' )
        return combined_values


    def __init__(
            self,
            architecture,
            existing_model = None,
            transfer = 0,
            seed = 42,
            log = output.Log() ) :

        assert existing_model or not transfer
        
        defaults = architecture.initial_parameter_values( seed )
        values = (
            Model.transfer( existing_model, defaults, architecture, transfer, log ) if transfer else
            Model.read( existing_model, architecture, log = log ) if existing_model else
            defaults ) 

        if existing_model:
            model_type = 'transferred' if transfer else 'existing'
            log.entry( f'verifying {model_type} model' )
            Model.verify_parameter_values( values, defaults, log )
        else:
            log.entry( 'completed random initialisation' )

        log.entry( 'constructing learnable parameters' )
        parameters = [ [ T.shared( name = n, value = v ) for n, v in subset ]
                       for subset in values ]

        input_broadcast_pattern = ( False, ) * architecture.input_dimensions
        input_type = T.tensor.TensorType( FloatType, input_broadcast_pattern )
        inputs = input_type( 'X' )

        log.entry( 'constructing forward network graph' )
        outputs = architecture.graph( parameters, inputs )

        label_broadcast_pattern = ( False, ) * architecture.output_dimensions
        label_type = T.tensor.TensorType( FloatType, label_broadcast_pattern )
        labels = label_type( 'Y' )

        self.__architecture = architecture
        self.__parameters = parameters
        self.__inputs = inputs
        self.__outputs = outputs
        self.__labels = labels
        self.__predictor = T.function( [ inputs ], outputs, allow_input_downcast = True )


    @property
    def architecture( self ) :

        return self.__architecture


    @property
    def parameters( self ) :

        return self.__parameters


    @property
    def parameter_names_and_values( self ) :

        return [ [ ( w.name, w.get_value() ) for w in layer ] for layer in self.parameters ]


    @property
    def parameter_names( self ) :

        return [ [ w.name for w in layer ] for layer in self.parameters ]


    @property
    def parameter_values( self ) :

        return numpy.array( [ [ w.get_value() for w in layer ] for layer in self.parameters ] )


    @property
    def inputs( self ):

        return self.__inputs


    @property
    def outputs( self ):

        return self.__outputs


    @property
    def labels( self ):

        return self.__labels


    def predict( self, inputs ) :

        return self.__predictor( inputs )


    def save_to_map( self ):

        data = dict(
            [ ( str(i) + ':' + w.name, w.get_value() )
              for i, layer in enumerate( self.parameters )
              for w in layer ] )

        return data



#---------------------------------------------------------------------------------------------------
