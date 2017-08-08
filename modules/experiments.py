#===================================================================================================
# experiment framework

'''
A pipeline for optimising a model over a dataset and storing the results.

This module provides the glue which adapts experiment input to the
needs of the optimiser, and then adapts the optimiser output to the
form required for the calculation of results.

'''


#---------------------------------------------------------------------------------------------------

import os
import sys

import ipdb

import numpy
from numpy import logical_not as negation

import data
import labels
import optimisation
import output
import results
import report
import sample


#---------------------------------------------------------------------------------------------------


class ExperimentDefinition( object ):

    
    @property
    def experiment_id( self ):

        raise NotImplementedError()


    @property
    def label_count( self ):

        raise NotImplementedError()
    

    @property
    def sample_parameters( self ):

        raise NotImplementedError()


    def dataset( self, input_path, log ):

        raise NotImplementedError()


    def model( self ):

        raise NotImplementedError() 

    
    def optimiser( self, log ):

        raise NotImplementedError() 




#---------------------------------------------------------------------------------------------------

class SegmentationExperiment( object ):



    def __init__( self, definition, input_path, output_path, log = output.Log( sys.stdout ) ):
        
        self.__log = log
        self.__input_path = input_path
        self.__output_path = output_path
        self.__definition = definition

        self.__model = None
        self.__optimiser = None
        self.__dataset = None


    @property
    def definition( self ):

        return self.__definition


    @property
    def dataset( self ):

        if not self.__dataset:
            self.log.subsection( 'constructing dataset' )
            self.__dataset = self.definition.dataset( self.input_path, self.log )

        return self.__dataset


    @property
    def model( self ):

        if not self.__model:
            self.log.entry( 'constructing model' )
            self.__model = self.definition.model()

        return self.__model


    @property
    def optimiser( self ):

        if not self.__optimiser:
            self.log.entry( 'constructing optimiser' )
            self.__optimiser = self.definition.optimiser( self.log )

        return self.__optimiser
    

    def training_set( self, dataset, random_generator ):

        self.log.entry( 'constructing training set accessor' )
        return sample.RandomAccessor(
            dataset.training_set,
            self.definition.sample_parameters,
            self.image_normalisation_for_optimisation,
            self.label_conversion_for_optimisation,
            random_generator,
            self.log )


    def validation_set( self, dataset ):

        self.log.entry( 'constructing validation set accessor' )
        return sample.SequentialAccessor(
            dataset.validation_set,
            self.definition.sample_parameters,
            self.image_normalisation_for_optimisation,
            self.label_conversion_for_optimisation,
            self.log )


    def validation_result_monitor( self ):

        self.log.entry( 'constructing validation results store' )
        return LabelAccumulationMonitor(
            self.output_path,
            self.definition.experiment_id, 
            self.definition.label_count, 
            self.label_conversion_for_results,
            self.definition.sample_parameters,
            self.log )


    @property
    def image_normalisation_for_optimisation( self ):

        return data.Normalisation.transform_to_zero_mean_and_unit_variance


    @property
    def label_conversion_for_optimisation( self ):

        raise NotImplementedError()


    @property
    def label_conversion_for_results( self ):

        raise NotImplementedError()


    @property
    def input_path( self ):

        return self.__input_path


    @property
    def output_path( self ):

        return self.__output_path


    @property
    def log( self ):

        return self.__log


    def run( self, random_generator ) :

        self.log.section( "initialising experiment" )
        dataset = self.dataset
        sample_parameters = self.definition.sample_parameters

        self.log.subsection( "constructing components" )
        optimiser = self.optimiser
        training_set = self.training_set( dataset, random_generator )
        validation_set = self.validation_set( dataset )
        validation_results_monitor = self.validation_result_monitor()
        model = self.model

        optimiser.optimise_until_converged(
            model,
            training_set,
            validation_set,
            optimisation.Monitor(),
            validation_results_monitor )

        self.log.entry( "optimisation complete" )

        self.log.subsection( "constructing report" )
        experiment_results = validation_results_monitor.results_for_most_recent_epoch
        report.Report.write( experiment_results, dataset, sample_parameters )
        self.log.entry( "report complete" )

        return validation_results_monitor.results_for_most_recent_epoch




#---------------------------------------------------------------------------------------------------


class SegmentationByPerVoxelClassificationExperiment( SegmentationExperiment ):


    @property
    def label_conversion_for_optimisation( self ):

        label_count = self.definition.label_count 
        return labels.dense_patch_indices_to_sparse_patch_distributions( label_count )


    @property
    def label_conversion_for_results( self ):

        assert self.definition.sample_parameters.patch_stride == 1
        return labels.sparse_patch_distribution_to_dense_volume_distribution
    



#---------------------------------------------------------------------------------------------------


class SegmentationByDenseInferenceExperiment( SegmentationExperiment ):


    @property
    def label_conversion_for_optimisation( self ):

        label_count = self.definition.label_count 
        margin = self.definition.sample_parameters.window_margin
        return labels.dense_patch_indices_to_cropped_dense_patch_distributions(
            label_count,
            margin )


    @property
    def label_conversion_for_results( self ):

        return labels.dense_patch_distribution_to_dense_volume_distribution



#---------------------------------------------------------------------------------------------------


class LabelAccumulationMonitor( optimisation.Monitor ):


    def __init__(
            self,
            data_path,
            results_id,
            class_count,
            patch_distributions_to_labelled_volume,
            parameters,
            log = output.Log() ):

        self.__log = log
        self.__parameters = parameters
        self.__patch_distributions_to_labelled_volume = patch_distributions_to_labelled_volume
        self.__results_id = results_id
        self.__data_path = data_path
        self.__class_count = class_count

        margin_loss = 2 * parameters.window_margin
        output_patch_shape = numpy.array( parameters.patch_shape ) - margin_loss
        accumulator_shape = ( 0, ) + tuple( output_patch_shape ) + ( class_count, )
        self.__predicted = numpy.zeros(( accumulator_shape ))
        self.__reference = numpy.zeros(( accumulator_shape ))
        self.__positions = numpy.zeros(( 0, 4 )).astype( 'int64' )

        self.__results = None
    

    @property
    def sample_parameters( self ):
    
        return self.__parameters


    @property
    def results_for_most_recent_epoch( self ):

        assert self.__results is not None
        return self.__results
    

    def results_for_epoch( self, epoch ):

        if self.__results and self.__results.epoch != epoch:
            
            self.__results.delete_from_archive()
            self.__results = None
            

        if not self.__results:
            self.__results = results.SegmentationResults(
                self.__data_path,
                self.__results_id,
                epoch,
                self.__class_count,
                self.__log )
            
        return self.__results 


    def reconstructed_volume( self, distribution_patches ):
        
        target_shape = numpy.array( self.sample_parameters.target_shape )
        margin = self.sample_parameters.window_margin
        distribution_to_labels = self.__patch_distributions_to_labelled_volume
        return distribution_to_labels( distribution_patches, target_shape, margin )


    def on_batch( self, epoch, batch, predicted, reference, positions ):

        assert batch is not None
        assert positions.shape[0] == reference.shape[0]
        assert predicted.shape == reference.shape

        results_for_epoch = self.results_for_epoch( epoch )

        self.__positions = numpy.concatenate(( self.__positions, positions ))
        self.__predicted = numpy.concatenate(( self.__predicted, predicted ))
        self.__reference = numpy.concatenate(( self.__reference, reference ))

        assert self.__predicted.shape == self.__reference.shape

        patch_count_per_volume = self.sample_parameters.patches_per_volume
        accumulated_count = self.__predicted.shape[0]
        completed_count = accumulated_count // patch_count_per_volume
        self.__log.entry( f'this batch  : {predicted.shape[0]}' )
        self.__log.entry( f'accumulated : {accumulated_count}' )
        self.__log.entry( f'per volume  : {patch_count_per_volume}' )
        self.__log.entry( f'completed   : {completed_count}' )
        
        if completed_count > 0:

            for i in range( completed_count ):
                m = patch_count_per_volume * i
                n = patch_count_per_volume * ( i + 1 )
                offset = self.__positions[ m, 1: ]
                volume_id = self.__positions[ m ][ 0 ]
                self.__log.entry( f'reconstructing : {volume_id}' )
                results_for_epoch.append_and_save(
                    volume_id,
                    self.reconstructed_volume( self.__predicted[ m : n ] ),
                    self.reconstructed_volume( self.__reference[ m : n ] ),
                    offset )
            
            block = numpy.s_[ 0 : completed_count * patch_count_per_volume ]
            self.__positions = numpy.delete( self.__positions, block, 0 )
            self.__predicted = numpy.delete( self.__predicted, block, 0 )
            self.__reference = numpy.delete( self.__reference, block, 0 )


    def on_epoch( self, epoch, mean_cost, model ):

        assert mean_cost is not None
        
        model_parameters = model.save_to_map()
        results_for_epoch = self.results_for_epoch( epoch )
        results_for_epoch.archive.save_model_parameters( model_parameters, epoch = epoch )


#---------------------------------------------------------------------------------------------------
