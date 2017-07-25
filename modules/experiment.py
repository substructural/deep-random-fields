#===================================================================================================
# experiment framework

import os
import pdb

import numpy
from numpy import logical_not as negation

import data
import labels
import network
import optimisation
import output
import results


#---------------------------------------------------------------------------------------------------

class Parameters( object ) :


    def __init__( self, experiment_id, output_path, epoch_count, cost_threshold, class_count ) :

        self.experiment_id = experiment_id
        self.output_path = output_path
        self.epoch_count = epoch_count
        self.cost_threshold = cost_threshold
        self.class_count = class_count


#---------------------------------------------------------------------------------------------------

class Experiment( object ) :


    def __init__(
            self,
            model,
            label_conversion,
            dataset,
            batch_parameters,
            experiment_parameters,
            results = None,
            log = None ) :

        self.__model = model
        self.__label_conversion = label_conversion
        self.__dataset = dataset
        self.__batch_parameters = batch_parameters
        self.__parameters = experiment_parameters
        self.__results = ( results if results is not None else
                           Results( label_conversion, experiment_parameters ) )

        self.__log = log if log else output.Log()


    @property
    def batch_parameters( self ) :

        return self.__batch_parameters


    @property
    def parameters( self ) :

        return self.__parameters


    @property
    def dataset( self ) :

        return self.__dataset


    @property
    def model( self ) :

        return self.__model


    @property
    def label_conversion( self ) :

        return self.__label_conversion


    @property
    def results( self ) :

        return self.__results


    @property
    def log( self ) :

        return self.__log


    def run( self ) :

        self.log.section( "experiment" )
        
        self.log.subsection( "experiment parameters" )
        self.log.record( self.parameters )

        self.log.subsection( "batch parameters" )
        self.log.record( self.batch_parameters )

        data_accessor = data.Accessor( self.dataset, self.batch_parameters, self.label_conversion )
        training_set_size = len( self.dataset.training_set )
        volumes_per_batch = self.batch_parameters.volume_count
        batch_count = int( training_set_size / volumes_per_batch )

        network.train(
            self.model,
            data_accessor.training_images_and_labels,
            data_accessor.validation_images_and_labels,
            self.parameters.epoch_count,
            batch_count,
            self.parameters.cost_threshold,
            on_batch_event = self.results.on_batch_event,
            on_epoch_event = self.results.on_epoch_event,
            maybe_log = self.log
        )

        self.log.entry( "complete" )


#---------------------------------------------------------------------------------------------------

class TrainingResultsAccumulator( optimisation.Monitor ):

    pass


#---------------------------------------------------------------------------------------------------

class ValidationResultsAccumulator( optimisation.Monitor ):


    def __init__(
            self, data_path, results_id, label_conversion, parameters, log = output.Log() ):

        self.__log = log
        self.__parameters = parameters
        self.__label_conversion = label_conversion
        self.__results_id = results_id
        self.__data_path = data_path

        accumulator_shape = ( 0, ) + parameters.patch_shape
        self.__predicted = numpy.zeros(( accumulator_shape ))
        self.__reference = numpy.zeros(( accumulator_shape ))
        self.__positions = numpy.zeros(( 0, 4 ))

        self.__results = None
    

    def results_for_epoch( self, epoch, class_count ):

        if not self.__results or self.__results.epoch != epoch:
            self.__results = results.SegmentationResults(
                self.__data_path,
                self.__results_id,
                epoch,
                class_count,
                self.__label_conversion,
                self.__log )
        else:
            return self.__results 


    def on_batch( self, epoch, batch, predicted, reference, positions ):

        assert batch is not None

        target_shape = numpy.array( self.__parameters.target_shape )
        patch_shape = numpy.array( self.__parameters.patch_shape )
        patch_count_per_volume = numpy.prod( target_shape // patch_shape )
        class_count = predicted.shape[-1]

        results_for_epoch = self.results_for_epoch( epoch, class_count )

        self.__positions = numpy.concatenate( self.__positions, positions )
        self.__predicted = numpy.concatenate( self.__predicted, predicted )
        self.__reference = numpy.concatenate( self.__reference, reference )

        assert self.__predicted.shape == self.__reference.shape

        accumulated_count = self.__predicted.shape[0]
        completed_count = accumulated_count // patch_count_per_volume
        
        if completed_count > 0:

            for i in range( completed_count ):
                m = patch_count_per_volume * i
                n = patch_count_per_volume * ( i + 1 )
                volume_id = self.__positions[ m ][ 0 ]
                results_for_epoch.append_and_save(
                    volume_id,
                    self.__predicted[ m : n ],
                    self.__reference[ m : n ],
                    target_shape )
            
            self.__predicted = numpy.delete( self.__predicted, completed_count, 0 )
            self.__reference = numpy.delete( self.__reference, completed_count, 0 )


    def on_epoch( self, epoch, mean_cost, model ):

        assert mean_cost is not None
        
        model_parameters = model.save_to_map()
        self.__results.archive.save_model_parameters( model_parameters, epoch = epoch )


#---------------------------------------------------------------------------------------------------
