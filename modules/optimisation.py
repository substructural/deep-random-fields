#===================================================================================================
# network optimisation module


#---------------------------------------------------------------------------------------------------

from datetime import datetime

import theano
import theano.tensor as T
import numpy

import learning_rates
import output


#---------------------------------------------------------------------------------------------------


class Parameters( object ):
    '''
    The meta parameters for the optimisation algorithm.

    '''

    def __init__(
            self,
            learning_rate = 0.1,
            weight_L1 = 0.0,
            weight_L2 = 0.0,
            weight_decay = 0.0,
            cost_sample_size = 3,
            recent_cost_sample_size = 2,
            convergence_threshold = 1e-5,
            maximum_epochs = 16 ):

        self.__learning_rate = learning_rate
        self.__weight_L1 = weight_L1
        self.__weight_L2 = weight_L2
        self.__weight_decay = weight_decay
        self.__cost_sample_size = cost_sample_size 
        self.__recent_cost_sample_size = recent_cost_sample_size
        self.__convergence_threshold = convergence_threshold
        self.__maximum_epochs = maximum_epochs


    @property
    def learning_rate( self ):
        '''
        The coefficient of the error gradient applied during parameter update.

        '''

        return self.__learning_rate


    @property
    def weight_L1( self ) :
        '''
        The coefficient of the L1 norm of the weights, applied during parameter update.

        '''

        return self.__weight_L1


    @property
    def weight_L2( self ) :
        '''
        The coefficient of the L2 norm of the weights, applied during parameter update.

        '''

        return self.__weight_L2


    @property
    def weight_decay( self ):
        '''
        A coefficient for a constant weight reduction term which pulls weights to zero over time.

        '''

        return self.__weight_decay


    @property
    def recent_cost_sample_size( self ):
        '''
        The number of epochs from which to compute a mean cost representative of the current model.

        '''

        return self.__recent_cost_sample_size


    @property
    def cost_sample_size( self ):
        '''
        The number of epochs from which to compute a mean cost to compare the current model against.

        '''

        return self.__cost_sample_size


    @property
    def convergence_threshold( self ):
        '''
        The maximum difference between minimum and maximum costs which constitutes convergence.

        '''

        return self.__convergence_threshold


    @property
    def maximum_epochs( self ):
        '''
        The maximum difference between minimum and maximum costs which constitutes convergence.

        '''

        return self.__maximum_epochs


#---------------------------------------------------------------------------------------------------


class Monitor( object ):


    def on_batch( self, epoch, batch, predicted, reference, positions ):
        ''' 
        Event handler called on completion of a batch during training or validation. 

        '''
        pass


    def on_epoch( self, epoch, model, costs, times ):
        ''' 
        Event handler called on completion of an epoch during training or validation. 

        '''
        pass

        


#---------------------------------------------------------------------------------------------------


class Cost:


    @staticmethod
    def has_converged( costs, parameters ) :

        k = parameters.recent_cost_sample_size

        if len( costs ) > 1 :
            minimum = min( costs[ -k : ] )
            maximum = max( costs[ -k : ] )
            change_over_last_k = abs( maximum - minimum )
            has_converged = change_over_last_k < parameters.convergence_threshold

            return has_converged
        else :
            return False


    @staticmethod
    def has_overfit( costs, parameters ) :

        n = parameters.cost_sample_size
        k = parameters.recent_cost_sample_size

        assert ( 0 < k ) and ( k < n )
        if len( costs ) >= n :
            mean_cost_of_last_n = float( sum( costs[ -n : ] ) ) / n
            mean_cost_of_last_k = float( sum( costs[ -k : ] ) ) / k
            has_overfit = mean_cost_of_last_k > mean_cost_of_last_n

            return has_overfit
        else :
            return False


#---------------------------------------------------------------------------------------------------


class Optimiser( object ) :


    def __init__(
            self,
            parameters,
            cost_function,
            learning_rate = None,
            log = output.Log() ) :

        default_learning_rate = learning_rates.ConstantLearningRate( parameters.learning_rate )
        self.__learning_rate = learning_rate if learning_rate else default_learning_rate
        self.__cost_function = cost_function
        self.__optimisation_parameters = parameters
        self.__log = log


    @property
    def log( self ):

        return self.__log


    @property
    def parameters( self ):

        return self.__optimisation_parameters


    @property
    def cost_function( self ):

        return self.__cost_function


    @property
    def learning_rate_schedule( self ):

        return self.__learning_rate


    def updates( self, model, cost ):

        raise NotImplementedError
    

    def optimisation_step( self, model ):

        cost = self.cost_function( model.outputs, model.labels, model.parameters )
        inputs = [ model.inputs, model.labels ]
        outputs = [ model.outputs, cost ]
        updates = self.updates( model, cost )

        self.log.entry( "compiling optimisation step" )
        return theano.function( inputs, outputs, updates = updates, allow_input_downcast = True )


    def validation_step( self, model ):

        cost = self.cost_function( model.outputs, model.labels, model.parameters )
        inputs = [ model.inputs, model.labels ]
        outputs = [ model.outputs, cost ]

        self.log.entry( "compiling validation step" )
        return theano.function( inputs, outputs, allow_input_downcast = True )


    def iterate(self, step, step_name, epoch, data, monitor, model ):

        self.log.subsection( step_name + " for epoch " + str(epoch) )
        costs = []
        times = []

        batches = len( data )
        epoch_start = datetime.now()

        for batch, ( images, distribution, positions ) in enumerate( data ):
            
            self.log.entry( f"{step_name} epoch: {epoch}, batch: {batch}/{batches}" )
            batch_start = datetime.now()
            predicted_distribution, cost = step( images, distribution )

            monitor.on_batch( epoch, batch, predicted_distribution, distribution, positions )
            batch_duration = ( datetime.now() - batch_start ).total_seconds()
            times.append( batch_duration )
            costs.append( cost )

            self.log.item( f"time: {batch_duration}" )
            self.log.item( f"cost: {cost}\n" )

        monitor.on_epoch( epoch, model, costs, times )
        mean_cost = numpy.sum( costs ) / len( costs )
        epoch_duration = ( datetime.now() - epoch_start ).total_seconds()

        self.log.entry( f"{step_name} epoch {epoch}" )
        self.log.item( f"time: {epoch_duration}" )
        self.log.item( f"cost: {mean_cost}" )

        return mean_cost


    def optimise_until_converged(
            self,
            model,
            optimisation_data,
            validation_data,
            optimisation_monitor,
            validation_monitor,
            initial_epoch = 0 ):

        self.log.section( "constructing graph" )
        optimisation_start = datetime.now()

        self.log.entry( "constructing validation graph" )
        validation_step = self.validation_step( model )

        self.log.entry( "constructing optimisation graph" )
        optimisation_step = self.optimisation_step( model )

        graph_duration = ( datetime.now() - optimisation_start ).total_seconds()
        self.log.entry( f"graph construction time: {graph_duration}" )


        self.log.section( "optimising model" )
        costs = []

        for epoch in range( initial_epoch, self.parameters.maximum_epochs ):

            self.iterate(
                optimisation_step, "optimisation", epoch, optimisation_data, optimisation_monitor, model )

            cost = self.iterate(
                validation_step, "validation", epoch, validation_data, validation_monitor, model )

            costs.append( cost )

            if Cost.has_converged( costs, self.parameters ):
                self.log.entry( "optimisation has converged" )
                break

            if Cost.has_overfit( costs, self.parameters ):
                self.log.entry( "optimisation has overfit" )
                break

        optimisation_duration = ( datetime.now() - optimisation_start ).total_seconds()
        self.log.entry( "optimisation complete" )
        self.log.entry( f"optimisation time: {optimisation_duration}" )
        return model


#---------------------------------------------------------------------------------------------------


class StochasticGradientDescent( Optimiser ) :


    def updates( self, model, cost ):

        parameters = [ p for subset in model.parameters for p in subset ]
        index = list( range( len( parameters ) ) )

        gradient = lambda p, c: T.grad( c, wrt=p, disconnected_inputs='ignore' )
        gradients = [ gradient( p, cost ) for p in parameters ]

        learning_rate = self.learning_rate_schedule

        if isinstance( learning_rate, learning_rates.GlobalLearningRate ):

            learning_rate, learning_rate_updates = learning_rate( cost )
            parameter_update = lambda i:  parameters[i] - learning_rate * gradients[i]
            parameter_updates = [ ( parameters[i], parameter_update(i) ) for i in index ]

            return parameter_updates + learning_rate_updates

        if isinstance( learning_rate, learning_rates.LocalLearningRate ):

            rates_and_updates = [ learning_rate( p, g ) for p, g in zip( parameters, gradients ) ]
            rate_per_parameter = [ r for r, u in rates_and_updates ]
            rate_updates = [ u for r, u in rates_and_updates ]

            parameter_update = lambda i: parameters[i] - rate_per_parameter[i] * gradients[i]
            parameter_updates = [ ( parameters[i], parameter_update(i) ) for i in index ]

            return parameter_updates + rate_updates

        raise Exception( 'unsupported learning rate type ' + str( type( learning_rate ) ) )


#---------------------------------------------------------------------------------------------------
