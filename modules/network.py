#===================================================================================================
# network

'''
A namespace for the classes which implement the core components of a neural network

'''


import theano as T
import theano.tensor

import numpy as N
import numpy.random



#---------------------------------------------------------------------------------------------------

def null_function( *args ) : return None

FloatType = T.config.floatX

#---------------------------------------------------------------------------------------------------


class Layer( object ) :

    ''' A single transformation within the network. '''


    def graph( self, parameters, inputs ) :

        raise NotImplementedError


    def initial_parameter_values( self ) :

        raise NotImplementedError



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


    def __init__( self, architecture, learned_parameter_values = None, seed = 42 ) :

        initial_values = (
            learned_parameter_values if learned_parameter_values is not None
            else architecture.initial_parameter_values( seed ) )

        parameters = [ [ T.shared( name = n, value = v ) for n, v in subset ]
                       for subset in initial_values ]

        input_broadcast_pattern = ( False, ) * architecture.input_dimensions
        input_type = T.tensor.TensorType( FloatType, input_broadcast_pattern )
        inputs = input_type( 'X' )

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



#---------------------------------------------------------------------------------------------------
