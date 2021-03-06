#===================================================================================================
# concrete network layer implementations


import enum

import numpy as N
import theano as T
import theano.printing

import network

FloatType = T.config.floatX


#---------------------------------------------------------------------------------------------------

class ScalarFeatureMapsProbabilityDistribution( network.Layer ):


    @property
    def parameter_names( self ):

        return []


    def initial_parameter_values( self ):

        return []


    def graph( self, parameters, inputs ):

        dimensions = len( list( T.tensor.shape( inputs ) ) )
        patch_dimensions = dimensions - 2
        probability_axis_last = [ 0 ] + [ i + 2 for i in range( patch_dimensions ) ] + [ 1 ]
        return inputs.dimshuffle( probability_axis_last )
    
        


#---------------------------------------------------------------------------------------------------

class Softmax( network.Layer ) :


    @property
    def parameter_names( self ):

        return []


    def initial_parameter_values( self ) :

        return []


    def graph( self, parameters, inputs ) :

        assert( parameters == [] )

        dimensions = len( list( T.tensor.shape( inputs ) ) )
        patch_dimensions = dimensions - 2
        switch_labels_and_batch = [ 1, 0 ] + [ i + 2 for i in range( patch_dimensions ) ]

        log = T.tensor.log
        exp = T.tensor.exp
        max = T.tensor.max
        sum = T.tensor.sum

        p = inputs.dimshuffle( switch_labels_and_batch )
        m = max( p, axis = 0 )
        s = exp( p - m - log( sum( exp( p - m ), axis = 0 ) ) )

        return s.dimshuffle( switch_labels_and_batch )


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


    @property
    def parameter_names( self ):

        return [ 'W', 'b' ]


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
            uniform_weights = False,
            orthogonal_weights = False,
            seed = None,
            add_input_feature_map_axis = False ) :

        assert not ( input_feature_maps > 1 and add_input_feature_map_axis )

        self.__input_feature_maps = input_feature_maps
        self.__output_feature_maps = output_feature_maps
        self.__kernel_shape = kernel_shape
        self.__stride = stride
        self.__activation = activation_function
        self.__add_input_feature_map_axis = add_input_feature_map_axis
        self.__orthogonal_weights = orthogonal_weights
        self.__uniform_weights = uniform_weights


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


    @property
    def parameter_names( self ):

        return [ 'K', 'b' ]


    def initial_parameter_values( self ) :

        shape = self.weights_shape 
        outputs = self.output_feature_maps
        inputs = self.input_feature_maps + N.prod( self.kernel_shape )

        if self.__uniform_weights:
            span = N.sqrt( 2.0 / ( inputs + outputs ) )
            weights = N.random.uniform( -span, +span, shape ).astype( FloatType )
            biases = N.random.uniform( 0, 0.1, outputs ).astype( FloatType )

        else:
            sigma = 1.0 / ( inputs + outputs )
            weights = N.random.normal( 0.0, sigma, shape ).astype( FloatType )
            biases = N.random.normal( 0.0, 0.1, outputs ).astype( FloatType )

        if self.__orthogonal_weights:
            flat_weights = weights.reshape(( shape[0], N.prod( shape[1:] ) ))
            _, _, right_singular = N.linalg.svd( flat_weights, full_matrices = False )
            orthogonal_weights = right_singular.reshape( shape )

            return [ ( 'K', orthogonal_weights ), ( 'b', biases ) ]

        else:
            return [ ( 'K', weights ), ( 'b', biases ) ]


    def graph( self, parameters, inputs ) :

        assert len( parameters ) == 2

        kernel, biases = parameters
        assert len( biases.get_value() ) == self.output_feature_maps

        dimensions = len( self.kernel_shape )
        assert dimensions == 2 or dimensions == 3

        input_shape = T.tensor.shape( inputs )
        input_feature_map_shape = list( input_shape[ 1 : 1 + dimensions ] )
        batches_by_one_feature_map_grid = [ input_shape[ 0 ], 1 ]
        extended_shape = batches_by_one_feature_map_grid + input_feature_map_shape
        shaped_inputs = ( T.tensor.reshape( inputs, extended_shape, 2 + dimensions )
                          if self.input_feature_maps == 1
                          else inputs )

        convolution = T.tensor.nnet.conv3d if dimensions == 3 else T.tensor.nnet.conv2d
        convolved = convolution(
            shaped_inputs,
            kernel,
            filter_shape = self.weights_shape,
            filter_dilation = ( self.stride, ) * dimensions,
            border_mode = 'valid' )

        feature_map_axis_to_end = [ 0 ] + [ 2 + i for i in range( dimensions ) ] + [ 1 ]
        feature_map_axis_to_2nd = [ 0, dimensions + 1 ] + [ 1 + i for i in range( dimensions ) ]
        shuffled_convolved = convolved.dimshuffle( feature_map_axis_to_end )
        shuffled_convolved_with_bias = shuffled_convolved + biases
        convolved_with_bias = shuffled_convolved_with_bias.dimshuffle( feature_map_axis_to_2nd )
        return self.activation.graph( convolved_with_bias )



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


    @property
    def parameter_names( self ):

        raise []


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

class Calculator:


    @staticmethod
    def parameters( kernel_size, layer_sizes, classes, verbose = False ):

        k = kernel_size ** 3
        fs = [ 1 ] + layer_sizes + [ classes ]
        cs = [ fs[i] * fs[i+1] for i in range( len( fs ) - 1 ) ]
        ps = [ c * k for c in cs ]
        total = sum( ps )
        if verbose:
            print( f'output feature maps: {fs}' )
            print( f'combined feature map: {cs}' )
            print( f'parameters per layer: {ps}' )
            print( f'parameters in total: {total}' )
        return total


#---------------------------------------------------------------------------------------------------
