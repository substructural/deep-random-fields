#===================================================================================================
# experiment framework

'''
A pipeline for optimising a model over a dataset and storing the results.

This module provides the glue which adapts experiment input to the
needs of the optimiser, and then adapts the optimiser output to the
form required for the calculation of results.

'''


#---------------------------------------------------------------------------------------------------

import inspect
import os
import sys

import ipdb

import numpy
import numpy.random
from numpy import logical_not as negation

import data
import labels
import network
import optimisation
import output
import results
import report
import sample


#---------------------------------------------------------------------------------------------------


class ExperimentDefinition( object ):


    def __init__( self ):

        self.__id = type( self ).__module__

    
    @property
    def experiment_id( self ):

        return self.__id

    
    @property
    def experiment_name( self ):

        return self.__id.replace( '_', ' ' )


    @property
    def label_count( self ):

        raise NotImplementedError()
    

    @property
    def sample_parameters( self ):

        raise NotImplementedError()


    def dataset( self, input_path, log ):

        raise NotImplementedError()


    def architecture( self ):

        raise NotImplementedError() 

    
    def optimiser( self, dataset, log ):

        raise NotImplementedError() 




#---------------------------------------------------------------------------------------------------

class SegmentationExperiment( object ):



    def __init__(
            self,
            definition,
            input_path,
            output_path,
            initial_epoch = 0,
            model_seed = 42,
            transfer_layers = 0,
            log = output.Log( sys.stdout ) ):
        
        self.__log = log
        self.__input_path = input_path
        self.__output_path = output_path
        self.__definition = definition
        self.__initial_epoch = initial_epoch
        self.__transfer_layers = transfer_layers
        self.__model_seed = model_seed

        self.__model = None
        self.__optimiser = None
        self.__dataset = None


    def archive( self ):

        return results.Archive( self.__output_path, self.definition.experiment_id, self.__log )
        

    @property
    def definition( self ):

        return self.__definition


    @property
    def dataset( self ):

        if not self.__dataset:
            self.__dataset = self.definition.dataset( self.input_path, self.log )

        return self.__dataset


    @property
    def model( self ):

        if not self.__model:

            self.log.entry( 'assembling model parameters' )
            initial_epoch = self.__initial_epoch
            transfer_layers = self.__transfer_layers
            model_seed = self.__model_seed
            architecture = self.definition.architecture()
            experiment_id = self.definition.experiment_id
            archive = results.Archive( self.__output_path, experiment_id, self.__log )

            assert not ( initial_epoch and transfer_layers )

            model_parameters = (
                archive.read_model_parameters( epoch = initial_epoch - 1 ) if initial_epoch > 0 else
                archive.read_model_parameters( object_id = 'transfer' ) if transfer_layers > 0 else
                None )

            self.__model = network.Model(
                architecture,
                existing_model = model_parameters,
                transfer = transfer_layers,
                seed = model_seed,
                log = self.log )

        return self.__model


    @property
    def optimiser( self ):

        if not self.__optimiser:
            self.log.entry( 'constructing optimiser' )
            self.__optimiser = self.definition.optimiser( self.dataset, self.log )

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


    def training_cost_monitor( self ):

        self.log.entry( 'constructing training cost store' )
        return TrainingCostMonitor(
            self.output_path,
            self.definition, 
            initial_epoch = self.__initial_epoch,
            log = self.log )


    def validation_result_monitor( self ):

        self.log.entry( 'constructing validation results store' )
        return LabelAccumulationMonitor(
            self.output_path,
            self.definition.experiment_id, 
            self.definition.label_count, 
            self.label_conversion_for_results,
            self.definition.sample_parameters,
            log = self.log,
            retain_current_results_only = False )


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


    def run( self, seed ) :

        self.log.section( "initialising experiment" )
        self.log.entry( self.definition.experiment_id.replace( '_', ' ' ) )
        dataset = self.dataset
        random_generator = numpy.random.RandomState( seed = seed )

        self.log.subsection( "constructing components" )
        training_set = self.training_set( dataset, random_generator )
        validation_set = self.validation_set( dataset )
        validation_results_monitor = self.validation_result_monitor()
        training_cost_monitor = self.training_cost_monitor()
        optimiser = self.optimiser

        self.log.subsection( "constructing model" )
        model = self.model

        optimiser.optimise_until_converged(
            model,
            training_set,
            validation_set,
            training_cost_monitor,
            validation_results_monitor,
            initial_epoch = self.__initial_epoch,
            constant_layers = self.__transfer_layers )

        self.log.subsection( "constructing report" )
        experiment_results = validation_results_monitor.results_for_most_recent_epoch
        report.Report.write( experiment_results, self )
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


class TrainingCostMonitor( optimisation.Monitor ):

    def __init__( self, output_path, experiment_definition, initial_epoch = 0, log = output.Log() ):

        experiment_id = experiment_definition.experiment_id
        self.__archive = results.Archive( output_path, experiment_id, log )

        if initial_epoch > 0:

            read_single_array_output = self.__archive.read_single_array_output
            self.__costs = list( read_single_array_output( 'costs', epoch = initial_epoch - 1 ) )
            self.__times = list( read_single_array_output( 'times', epoch = initial_epoch - 1 ) )

            assert len( self.__costs ) == initial_epoch
            assert len( self.__times ) == initial_epoch

        else:

            self.__costs = []
            self.__times = []


    def on_epoch( self, epoch, model, costs, times ):
        
        assert len( self.__costs ) == epoch
        assert len( self.__times ) == epoch

        def maybe_padded( xs, existing ):
            target = len( existing[0] ) if existing else len( xs )
            difference = target - len( xs )
            return list( xs ) + [ 0.0 ] * difference

        self.__costs.append( maybe_padded( costs, self.__costs ) )
        self.__times.append( maybe_padded( times, self.__times ) )
        self.__archive.save_array_output( numpy.array( self.__costs ), 'costs', epoch = epoch )
        self.__archive.save_array_output( numpy.array( self.__times ), 'times', epoch = epoch )



#---------------------------------------------------------------------------------------------------


class LabelAccumulationMonitor( optimisation.Monitor ):


    def __init__(
            self,
            data_path,
            results_id,
            class_count,
            patch_distributions_to_labelled_volume,
            parameters,
            log = output.Log(),
            retain_current_results_only = False ):

        self.__log = log
        self.__parameters = parameters
        self.__patch_distributions_to_labelled_volume = patch_distributions_to_labelled_volume
        self.__results_id = results_id
        self.__data_path = data_path
        self.__class_count = class_count
        self.__retain_current_results_only = retain_current_results_only

        accumulator_shape = ( 0, ) + tuple( parameters.output_patch_shape ) + ( class_count, )
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
            
            if self.__retain_current_results_only:
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
        
        if completed_count > 0:

            for i in range( completed_count ):
                m = patch_count_per_volume * i
                n = patch_count_per_volume * ( i + 1 )

                volume_ids = self.__positions[ m:n, 0 ]
                positions = self.__positions[ m:n, 1: ]
                minimum_position = numpy.min( positions, axis = 0 )
                assert numpy.array_equal( minimum_position, positions[0] )
                assert numpy.all( volume_ids == volume_ids[0] )

                results_for_epoch.append_and_save(
                    volume_ids[0],
                    self.reconstructed_volume( self.__predicted[ m : n ] ),
                    self.reconstructed_volume( self.__reference[ m : n ] ),
                    self.__positions[ m ] )
            
            block = numpy.s_[ 0 : completed_count * patch_count_per_volume ]
            self.__positions = numpy.delete( self.__positions, block, 0 )
            self.__predicted = numpy.delete( self.__predicted, block, 0 )
            self.__reference = numpy.delete( self.__reference, block, 0 )


    def on_epoch( self, epoch, model, costs, times ):

        model_parameters = model.save_to_map()
        results_for_epoch = self.results_for_epoch( epoch )
        results_for_epoch.archive.save_model_parameters( model_parameters, epoch = epoch )


#---------------------------------------------------------------------------------------------------
