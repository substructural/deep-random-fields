#===================================================================================================
# cost functions


import theano.tensor as T


#---------------------------------------------------------------------------------------------------

class CostFunction( object ) :


    def __init__( self, weight_L1 = 0.0, weight_L2 = 0.0 ) :

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


    def cost( self, outputs, labels ):

        raise NotImplementedError


    def __call__( self, outputs, labels, parameters ):

        L1 = self.regularise_L1( parameters )
        L2 = self.regularise_L2_square( parameters )
        cost = self.cost( outputs, labels )
        return cost + L1 + L2
    

#---------------------------------------------------------------------------------------------------

class CategoricalCrossEntropyCost( CostFunction ):


    def __init__( self, distribution_axis, weight_L1 = 0.0, weight_L2 = 0.0 ):

        self.__distribution_axis = distribution_axis
        super( CategoricalCrossEntropyCost, self ).__init__( weight_L1, weight_L2 )


    @property
    def distribution_axis( self ):

        return self.__distribution_axis


    def cost( self, outputs, labels ):

        reference = labels.clip( 0.0, 1.0 )
        predicted = outputs.clip( 2**(-126), 1.0 )

        cross_entropy = (-1) * T.sum( reference * T.log2( predicted ), axis=self.distribution_axis )
        mean_cross_entropy = T.mean( cross_entropy )
        return mean_cross_entropy
    

#---------------------------------------------------------------------------------------------------

class WeightedCategoricalCrossEntropyCost( CostFunction ):


    def __init__( self, prior_distribution, distribution_axis, weight_L1 = 0.0, weight_L2 = 0.0 ):

        self.__distribution_axis = distribution_axis
        self.__prior_distribution = prior_distribution
        super( WeightedCategoricalCrossEntropyCost, self ).__init__( weight_L1, weight_L2 )


    @property
    def distribution_axis( self ):

        return self.__distribution_axis


    @property
    def prior_distribution( self ):

        return self.__prior_distribution


    def cost( self, outputs, labels ):

        axis = self.distribution_axis
        label_count = len( self.prior_distribution )
        uniform = 1.0 / label_count
        weights = uniform / self.prior_distribution

        reference = labels.clip( 0.0, 1.0 )
        predicted = outputs.clip( 2**(-126), 1.0 )

        cross_entropy = (-1) * T.sum( weights * reference * T.log2( predicted ), axis=axis )
        mean_cross_entropy = T.mean( cross_entropy )
        return mean_cross_entropy


#---------------------------------------------------------------------------------------------------
