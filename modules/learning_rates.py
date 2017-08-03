#===================================================================================================
# learning rate



#---------------------------------------------------------------------------------------------------

import theano
import numpy

FloatType = theano.config.floatX


#---------------------------------------------------------------------------------------------------


class LearningRate( object ):


    def __init__( self, base_learning_rate ):

        self.__base_learning_rate = base_learning_rate


    @property
    def base_learning_rate( self ):

        return self.__base_learning_rate


#---------------------------------------------------------------------------------------------------


class GlobalLearningRate( LearningRate ):


    def __call__( self, cost ):

        raise NotImplementedError()



class ConstantLearningRate( GlobalLearningRate ):


    def __call__( self, cost_gradients ):

        return self.base_learning_rate, []

    
#---------------------------------------------------------------------------------------------------


class LocalLearningRate( LearningRate ):


    def __call__( self, model_parameters, cost_gradients ):

        raise NotImplementedError()


class RMSPropLearningRate( LocalLearningRate ):


    def __init__( self, base_learning_rate, mean_weighting ):

        self.__mean_weighting = mean_weighting
        super( RMSPropLearningRate, self ).__init__( base_learning_rate )


    @property
    def mean_weighting( self ):

        return self.__mean_weighting


    def __call__( self, parameter, cost_gradient ):

        gamma = self.mean_weighting
        eta = self.base_learning_rate
        epsilon = 0.0001

        initial_value = parameter.get_value()
        is_tensor = isinstance( initial_value, numpy.ndarray ) and ( initial_value.shape != () )
        zero = numpy.zeros( initial_value.shape ).astype( FloatType ) if is_tensor else 0.0

        mean_squared_gradient = theano.shared( name = 'mean_squared_gradient', value = zero )
        squared_gradient = ( gamma * mean_squared_gradient ) + ( 1 - gamma )*( cost_gradient ** 2 )
        rate = eta * theano.tensor.sqrt( squared_gradient + epsilon )

        return rate, ( mean_squared_gradient, squared_gradient )


#---------------------------------------------------------------------------------------------------
