#===================================================================================================
# experiment framework

import os
import pdb

import numpy
from numpy import logical_not as negation

import data
import labels
import network


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
            results = None ) :

        self.__model = model
        self.__label_conversion = label_conversion
        self.__dataset = dataset
        self.__batch_parameters = batch_parameters
        self.__parameters = experiment_parameters
        self.__results = ( results if results is not None else
                           Results( label_conversion, experiment_parameters ) )


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


    def run( self ) :

        data_accessor = data.Accessor( self.dataset, self.batch_parameters, self.label_conversion )
        training_set_size = len( self.dataset.training_set )
        volumes_per_batch = self.batch_parameters.volume_count
        batch_count = int( training_set_size / volumes_per_batch )

        network.train(
            self.model,
            data_accessor.training_images_and_labels,
            data_accessor.validation_images_and_labels,
            self.parameters.cost_threshold,
            self.parameters.epoch_count,
            batch_count,
            self.results.on_batch_event,
            self.results.on_epoch_event )


#---------------------------------------------------------------------------------------------------

class Results( object ):


    def __init__( self, label_conversion, parameters ):

        self.__parameters = parameters
        self.__label_conversion = label_conversion
        self.__training_costs = []
        self.__validation_costs = []
        self.__dice_scores = []


    @property
    def parameters( self ):

        return self.__parameters


    @property
    def label_conversion( self ):

        return self.__label_conversion


    @property
    def training_costs( self ):

        return self.__training_costs


    @property
    def validation_costs( self ):

        return self.__validation_costs


    @property
    def dice_scores( self ):

        return self.__dice_scores


    def saved_object_file_name( self, object_type, epoch = None ):

        directory = self.parameters.output_path
        epoch_tag = str( epoch ) + "-" if epoch is not None else ""
        filename = self.parameters.experiment_id + "-" + epoch_tag + object_type + ".npy"
        return directory + "/" + filename


    def save_array_output( self, output, object_type, epoch = None ):

        directory = self.parameters.output_path
        if not os.path.exists( directory ):
            os.mkdir( directory )

        filepath = self.saved_object_file_name( object_type, epoch )
        numpy.save( filepath, output, allow_pickle=False )
        return filepath


    def on_batch_event( self, batch_index, training_output, training_costs ) :

        print( "\nbatch {0:03d}: {1:0.5f}".format( batch_index, training_costs ) )


    def on_epoch_event(
            self,
            epoch_index,
            model,
            patch_grid,
            validation_labels,
            validation_output,
            validation_cost,
            training_costs ) :

        print( "\nepoch", epoch_index, ":\n", validation_cost )

        classes = self.parameters.class_count
        volumes = validation_labels.shape[ 0 ]
        test_labels = self.label_conversion.labels_for_volumes( validation_output, patch_grid )
        true_labels = self.label_conversion.labels_for_volumes( validation_labels, patch_grid )
        test_masks = labels.dense_volume_indices_to_dense_volume_masks( test_labels, classes )
        true_masks = labels.dense_volume_indices_to_dense_volume_masks( true_labels, classes )
        mean_dice_scores_per_class = Metrics.mean_dice_score_per_class( test_masks, true_masks )

        self.save_array_output( model, "model", epoch_index )
        self.save_array_output( validation_output, "output", epoch_index )
        self.save_array_output( test_labels, "labels", epoch_index )
        self.save_array_output( test_masks, "masks", epoch_index )
        for c in range( 0, classes ):
            difference_map = Images.difference_of_masks( test_masks[ :, c ], true_masks[ :, c ] )
            self.save_array_output( difference_map, "class-" + str( c ), epoch_index )

        self.training_costs.append( training_costs )
        self.validation_costs.append( validation_cost )
        self.dice_scores.append( mean_dice_scores_per_class )


#---------------------------------------------------------------------------------------------------

class Metrics:


    @staticmethod
    def dice_score( predicted, reference ):

        true_positives = numpy.count_nonzero( predicted & reference )
        false_positives = numpy.count_nonzero( predicted & negation( reference ) )
        false_negatives = numpy.count_nonzero( reference & negation( predicted ) )

        return (
            ( 2.0 * true_positives ) /
            ( 2.0 * true_positives + false_positives + false_negatives ) )


    @staticmethod
    def mean_dice_score_per_class( predicted_masks_per_volume, reference_masks_per_volume ):
        volumes = predicted_masks_per_volume.shape[ 0 ]
        classes = predicted_masks_per_volume.shape[ 1 ]
        dice = lambda v, c : Metrics.dice_score( predicted_masks_per_volume[ v, c ],
                                                 reference_masks_per_volume[ v, c ] )

        return [
            ( 1 / volumes ) * sum( [ dice( v, c ) for v in range( 0, volumes ) ] )
            for c in range( 0, classes ) ]


#---------------------------------------------------------------------------------------------------

class Images:


    @staticmethod
    def difference_of_masks( predicted, reference ):

        r, g, b = 0, 1, 2
        difference_shape = ( 3, ) + predicted.shape

        true_positives  = ( predicted & reference ) == 1
        false_positives = ( predicted & negation( reference ) ) == 1
        false_negatives = ( reference & negation( predicted ) ) == 1

        difference_map = numpy.zeros( difference_shape ).astype( 'uint8' )
        difference_map[ g ][ true_positives  ] = 0xD0
        difference_map[ r ][ false_positives ] = 0xD0
        difference_map[ b ][ false_negatives ] = 0xD0

        permutation_to_rgb_values = [ i for i in range( 1, len( difference_shape ) ) ] + [ 0 ]
        return numpy.transpose( difference_map, permutation_to_rgb_values )


#---------------------------------------------------------------------------------------------------
