#===================================================================================================
# concrete network layer implementations


import numpy as N
import theano as T

import net


#---------------------------------------------------------------------------------------------------

class Softmax( net.Layer ) :


    def initial_parameter_values( self ) :

        return []


    def graph( self, parameters, inputs ) :

        assert( parameters == [] )

        softmax = T.tensor.exp( inputs ) / T.tensor.sum( T.tensor.exp( inputs ) )
        return softmax


#---------------------------------------------------------------------------------------------------

class DenseLayer( net.Layer ) :


    def __init__( self, weights_shape, activation_function, seed = None ) :

        assert( len( weights_shape ) == 2 )
        self.__weights_shape = weights_shape
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

class ConvolutionalLayer( net.Layer ) :


    def __init__(
            self,
            input_feature_maps,
            output_feature_maps,
            kernel_shape,
            stride,
            activation_function,
            seed = None ) :

        self.__input_feature_maps = input_feature_maps
        self.__output_feature_maps = output_feature_maps
        self.__kernel_shape = kernel_shape
        self.__stride = stride
        self.__activation = activation_function


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
    def stride( self ) :

        return self.__stride


    @property
    def activation( self ) :

        return self.__activation


    def initial_parameter_values( self ) :

        weights_shape = ( self.output_feature_maps, self.input_feature_maps ) + self.kernel_shape
        initial_kernel = N.random.standard_normal( weights_shape ).astype( T.config.floatX )
        initial_biases = N.zeros( self.output_feature_maps )

        return [ ( 'K', initial_kernel ), ( 'b', initial_biases ) ]


    def graph( self, parameters, inputs ) :

        assert( len( parameters ) == 2 )

        kernel, biases = parameters
        assert( len( biases ) == self.output_feature_maps )

        dimensions = len( self.kernel_shape )
        assert( dimensions == 2 or dimensions == 3 )

        convolution = T.tensor.nnet.conv3d if dimensions == 3 else T.tensor.nnet.conv2d
        convolved_inputs = convolution( inputs, kernel, biases, filter_dilation = self.stride, border_mode = 'valid' )
        return self.activation.graph( convolved_inputs + biases )



#---------------------------------------------------------------------------------------------------
