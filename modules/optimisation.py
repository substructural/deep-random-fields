#===================================================================================================
# network optimisation module


import theano.tensor as T


#---------------------------------------------------------------------------------------------------

class CostFunction( object ) :


    def __init__( self, weight_L1, weight_L2 ) :

        self.__weight_L1 = weight_L1
        self.__weight_L2 = weight_L2


    @property
    def weight_L1( self ) :

        return self.__weight_L1


    @property
    def weight_L2( self ) :

        return self.__weight_L2


    def regularise_L1( self, parameters ) :

        L1 = T.sum( [ T.sum( abs( p ) ) for subset in parameters for p in subset ] )
        return self.weight_L1 * L1


    def regularise_L2_square( self, parameters ) :

        L2 = T.sum( [ T.sum( p ** 2 ) for subset in parameters for p in subset ] )
        return self.weight_L2 * L2


#---------------------------------------------------------------------------------------------------

class CategoricalCrossEntropyCost( CostFunction ) :


    def __init__( self, weight_L1, weight_L2 ) :

        super( CategoricalCrossEntropyCost, self ).__init__( weight_L1, weight_L2 )


    def __call__( self, outputs, labels, parameters ) :

        distribution = len( labels.shape ) - 1
        cross_entropy = (-1) * T.sum( labels * T.log( outputs ), axis=distribution )
        mean_cross_entropy = T.mean( cross_entropy )
        penalty_l1 = self.regularise_L1( parameters )
        penalty_l2 = self.regularise_L2_square( parameters )
        return mean_cross_entropy + penalty_l1 + penalty_l2


#---------------------------------------------------------------------------------------------------

class SimpleGradientDescentOptimiser( object ) :


    def __init__( self, learning_rate ) :

        self.__learning_rate = learning_rate


    def __call__( self, parameter, cost ) :

        return parameter - self.learning_rate * T.grad( cost, wrt=parameter )


    @property
    def learning_rate( self ) :

        return self.__learning_rate


#---------------------------------------------------------------------------------------------------
