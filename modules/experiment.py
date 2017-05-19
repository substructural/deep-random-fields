#===================================================================================================
# experiment framework

import os

import numpy

import data
import labels
import network


#---------------------------------------------------------------------------------------------------

class Parameters( object ) :


    def __init__( self, experiment_id, output_path, epoch_count, cost_threshold, class_count ) :

        self.experiment_id = experiment_id
        self.output_path = output_path
        self.class_count = class_count
        self.cost_threshold = cost_threshold
        self.epoch_count = epoch_count


#---------------------------------------------------------------------------------------------------

class Experiment( object ) :


    def __init__(
            self,
            model,
            label_conversion,
            dataset,
            batch_parameters,
            experiment_parameters ) :

        self.__model = model
        self.__label_conversion = label_conversion
        self.__dataset = dataset
        self.__batch_parameters = batch_parameters
        self.__parameters = experiment_parameters


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


    def save_array_output( self, output, output_id ):

        directory = self.parameters.output_path
        filename = self.parameters.experiment_id + "-" + output_id + ".npy"

        if not os.path.exists( directory ):
            os.mkdir( directory )

        numpy.save( directory + "/" + filename, output, allow_pickle=False )


    def on_batch_event( self, batch_index, training_output, training_costs ) :

        print( "\nbatch", batch_index, ":\n", training_costs )


    def on_epoch_event( self, epoch_index, validation_output, validation_costs, training_costs ) :

        index_count = self.parameters.index_count
        volumes = self.label_conversion.labels_for_volumes( validation_output )
        masks = labels.dense_volume_indices_to_dense_volume_masks( volumes, index_count )

        self.save_array_output( volumes, str( epoch_index ) + "-volumes" )
        self.save_array_output( masks, str( epoch_index ) + "-masks" )

        print( "\nepoch", epoch_index, ":\n", validation_costs )
        # TODO : we should calculate dice scores for the masks and output this here


    def run( self ) :

        data_accessor = data.Accessor( self.dataset, self.batch_parameters, self.label_conversion )
        training_set_size = len( self.dataset.training_set )
        volumes_per_batch = self.batch_parameters.volume_count
        batch_count = training_set_size / volumes_per_batch

        network.train(
            self.model,
            data_accessor.training_images_and_labels,
            data_accessor.validation_images_and_labels,
            self.parameters.cost_threshold,
            self.parameters.epoch_count,
            batch_count,
            self.on_batch_event,
            self.on_epoch_event )


#---------------------------------------------------------------------------------------------------
