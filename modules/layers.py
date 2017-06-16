#===================================================================================================
# concrete network layer implementations


import enum

import numpy as N
import theano as T

import network


#---------------------------------------------------------------------------------------------------

class Softmax( network.Layer ) :


    def initial_parameter_values( self ) :

        return []


    def graph( self, parameters, inputs ) :

        assert( parameters == [] )

        softmax = T.tensor.exp( inputs ) / T.tensor.sum( T.tensor.exp( inputs ) )
        return softmax


#---------------------------------------------------------------------------------------------------

class DenseLayer( network.Layer ) :


    def __init__(
            self,
            input_size,
            output_size,
            activation_function,
            seed = None ):

        self.__weights_shape = ( input_size, output_size )
        self.__activation = activation_function


    @property
    def activation( self ) :

        return self.__activation


    @property
    def weights_shape( self ) :

        return self.__weights_shape


    @property
    def biases_shape( self ) :

        return self.__weights_shape[ 1 ]


    def initial_parameter_values( self ) :

        initial_weights = N.random.standard_normal( self.weights_shape ).astype( T.config.floatX )
        initial_biases = N.zeros( self.biases_shape ).astype( T.config.floatX )

        return [ ( 'W', initial_weights ), ( 'b', initial_biases ) ]


    def graph( self, parameters, inputs ) :

        assert( len( parameters ) == 2 )
        weights, biases = parameters
        return self.activation.graph( T.dot( inputs, weights ) + biases )


#---------------------------------------------------------------------------------------------------

class ConvolutionalLayer( network.Layer ) :


    def __init__(
            self,
            input_feature_maps,
            output_feature_maps,
            kernel_shape,
            stride,
            activation_function,
            seed = None,
            add_input_feature_map_axis = False ) :

        assert not ( input_feature_maps > 1 and add_input_feature_map_axis )

        self.__input_feature_maps = input_feature_maps
        self.__output_feature_maps = output_feature_maps
        self.__kernel_shape = kernel_shape
        self.__stride = stride
        self.__activation = activation_function
        self.__add_input_feature_map_axis = add_input_feature_map_axis


    @property
    def input_feature_maps( self ) :

        return self.__input_feature_maps


    @property
    def output_feature_maps( self ) :

        return self.__output_feature_maps


    @property
    def kernel_shape( self ) :

        return self.__kernel_shape


    @property
    def weights_shape( self ):
        
        return ( self.output_feature_maps, self.input_feature_maps ) + self.kernel_shape


    @property
    def stride( self ) :

        return self.__stride


    @property
    def activation( self ) :

        return self.__activation


    def initial_parameter_values( self ) :

        initial_kernel = N.random.standard_normal( self.weights_shape ).astype( T.config.floatX )
        initial_biases = N.zeros( self.output_feature_maps ).astype( T.config.floatX )

        return [ ( 'K', initial_kernel ), ( 'b', initial_biases ) ]


    def graph( self, parameters, inputs ) :

        assert len( parameters ) == 2

        kernel, biases = parameters
        assert len( biases.get_value() ) == self.output_feature_maps

        dimensions = len( self.kernel_shape )
        assert dimensions == 2 or dimensions == 3

        input_shape = T.tensor.shape( inputs )
        extended_shape = ( input_shape[ 0 ], 1 ) + input_shape[ 1 : 1 + dimensions ]
        shaped_inputs = ( T.tensor.reshape( inputs, extended_shape, 2 + dimensions )
                          if self.input_feature_maps == 1
                          else inputs )

        convolution = T.tensor.nnet.conv3d if dimensions == 3 else T.tensor.nnet.conv2d
        convolved_inputs = convolution(
            shaped_inputs,
            kernel,
            filter_shape = self.weights_shape,
            filter_dilation = ( self.stride, ) * dimensions,
            border_mode = 'valid' )

        return self.activation.graph( convolved_inputs + biases )



#---------------------------------------------------------------------------------------------------


class PoolingType( enum.Enum ):

    MAX = 'max'
    MIN = 'min'
    MEAN = 'average_exc_pad'



class PoolingLayer( network.Layer ):
   

    def __init__( self, pooling_type, factor, stride = None ):

        self.__pooling_type = pooling_type
        self.__factor = factor
        self.__stride = stride


    @property
    def stride( self ):

        return self.__stride


    @property
    def factor( self ):

        return self.__factor


    @property
    def pooling_type( self ):

        return self.__pooling_type


    def initial_parameter_values( self ):

        return []


    def graph( self, parameters, inputs ):

        assert parameters == []

        in_3d = len( self.factor ) == 3
        in_2d = len( self.factor ) == 2
        assert in_3d or in_2d
        
        mode = self.pooling_type.value
        pool = T.tensor.signal.pool.pool_3d if in_3d else T.tensor.signal.pool.pool_2d

        return pool( inputs, self.factor, stride=self.stride, mode=mode, ignore_border=True )
        
    

#---------------------------------------------------------------------------------------------------
