#====================================================================================================
# base classes for neural network construction


import theano as T
import theano.tensor

import numpy as N
import numpy.random

import data

import pdb

#----------------------------------------------------------------------------------------------------

def null_function( *args ) : return None


#----------------------------------------------------------------------------------------------------


class Layer( object ) :

    ''' A single transformation within the network. '''


    def graph( self, model_parameters, inputs ) :

        raise NotImplementedError


    def initial_parameter_values( self ) :

        raise NotImplementedError



#----------------------------------------------------------------------------------------------------


class Architecture( object ):

    ''' The graph of transformations, represented as layers, defining the structure of the network. '''


    def __init__( self, layers, input_dimensions = 1, output_dimensions = 1 ) :

        assert( layers is not [] )
        self.__layers = layers
        self.__input_dimensions = input_dimensions
        self.__output_dimensions = output_dimensions


    @property
    def input_dimensions( self ) :

        return self.__input_dimensions


    @property
    def output_dimensions( self ) :

        return self.__output_dimensions


    @property
    def layers( self ) :

        return self.__layers


    def initial_parameter_values( self, seed ) :

        N.random.seed( seed )
        return [ layer.initial_parameter_values() for layer in self.layers ]


    def graph( self, model_parameters, inputs ) :

        assert( len( model_parameters ) == len( self.layers ) )

        graph = inputs
        for i, layer in enumerate( self.layers ) :

            graph = layer.graph( model_parameters[ i ], graph )

        return graph




#----------------------------------------------------------------------------------------------------


class Model( object ) :

    ''' A learned parameterisation of a network architecture, trained on labelled data. '''


    def __init__( self, architecture, cost_function, optimiser, learned_parameter_values = None, seed = 42 ) :

        initial_values = (
            learned_parameter_values if learned_parameter_values is not None
            else architecture.initial_parameter_values( seed ) )

        parameters = [ [ T.shared( name = n, value = v ) for n, v in subset ] for subset in initial_values ]

        input_broadcast_pattern = ( False, ) * architecture.input_dimensions
        input_type = T.tensor.TensorType( T.config.floatX, input_broadcast_pattern )
        inputs = input_type( 'X' )

        label_broadcast_pattern = ( False, ) * architecture.output_dimensions
        label_type = T.tensor.TensorType( T.config.floatX, label_broadcast_pattern )
        labels = label_type( 'Y' )

        outputs = architecture.graph( parameters, inputs )
        cost = cost_function( outputs, labels, parameters )
        updates = [ ( parameter, optimiser( parameter, cost ) ) for subset in parameters for parameter in subset ]

        self.__validation_graph = T.function( [ inputs, labels ], [ outputs, cost ], allow_input_downcast = True )
        self.__optimisation_graph = T.function( [ inputs, labels ], [ outputs, cost ], updates = updates, allow_input_downcast = True )
        self.__architecture = architecture
        self.__parameters = parameters


    @property
    def architecture( self ) :

        return self.__architecture


    @property
    def parameters( self ) :

        return self.__parameters


    def current_values( self ) :

        return [ [ w.get_value() for w in layer ] for layer in self.parameters ]


    def validate( self, inputs, labels ) :

        return self.__validation_graph( inputs, labels )


    def optimise( self, inputs, labels ) :

        return self.__optimisation_graph( inputs, labels )



#---------------------------------------------------------------------------------------------------


def has_converged( costs, threshold = 1e-5, k = 4 ) :

    if len( costs ) > 1 :
        minimum = min( costs[ -k : ] )
        maximum = max( costs[ -k : ] )
        change_over_last_k = abs( maximum - minimum )
        return change_over_last_k < threshold
    else :
        return False


def has_overfit( costs, k = 2, n = 4 ) :

    assert( 0 < k and k < n )
    if len( costs ) >= n :
        mean_cost_of_last_n = float( sum( costs[ -n : ] ) ) / n
        mean_cost_of_last_k = float( sum( costs[ -k : ] ) ) / k
        has_overfit = mean_cost_of_last_k > mean_cost_of_last_n
        return has_overfit
    else :
        return False


def train_for_epoch( model, load_training_set, batch_count, on_batch_event = null_function ) :

    training_costs = []

    for batch_index in range( 0, batch_count ) :

        training_inputs, training_labels = load_training_set( batch_index )
        training_output, training_cost = model.optimise( training_inputs, training_labels )

        on_batch_event( batch_index, training_output, training_cost )
        training_costs.append( training_cost )

    return training_costs


def train(
        model,
        load_training_set,
        load_validation_set,
        epoch_count,
        batch_count,
        cost_threshold_for_convergence,
        tail_length_for_convergence = 3,
        tail_length_for_overfitting = 3,
        on_batch_event = null_function,
        on_epoch_event = null_function ) :

    validation_costs = []
    training_costs = []

    for epoch in range( 0, epoch_count ) :

        training_costs_for_epoch = train_for_epoch( model, load_training_set, batch_count, on_batch_event )
        training_costs.append( training_costs_for_epoch )

        validation_inputs, validation_labels = load_validation_set( epoch )
        validation_output, validation_cost = model.validate( validation_inputs, validation_labels )
        validation_costs.append( validation_cost )

        on_epoch_event( epoch, validation_output, validation_cost, training_costs_for_epoch )

        network_has_overfit = has_overfit(
            validation_costs,
            1,
            tail_length_for_overfitting )

        network_has_converged = has_converged(
            validation_costs,
            cost_threshold_for_convergence,
            tail_length_for_convergence )

        if network_has_overfit or network_has_converged:
            break

    return validation_output, validation_costs, training_costs
