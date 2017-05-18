#===================================================================================================
# activation functions



#---------------------------------------------------------------------------------------------------

import theano.tensor as T


#---------------------------------------------------------------------------------------------------

class LeakyRectifiedLinearUnit( object ) :

    def __init__( self, leakiness ) :

        self.__leakiness = T.TensorConstant( leakiness )


    @property
    def leakiness( self ) :

        return self.__leakiness


    def graph( self, inputs ) :

        return T.switch( ( inputs < 0 ), ( self.leakiness * inputs ), ( inputs ) )
        


#---------------------------------------------------------------------------------------------------
