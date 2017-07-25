#===================================================================================================
# results


# TODO:
#
#  - implement confusion matrix calculation
#  - update the dice score per class to use only one volume
#  - change the interface to the optimisation monitor
#  - update the experiment to use the new model, optimiser and monitors

#---------------------------------------------------------------------------------------------------

import os

import numpy
from numpy import logical_not as negation

import labels
import output



#---------------------------------------------------------------------------------------------------

class Archive( object ):
    

    def __init__( self, data_path, archive_name, log = output.Log() ):

        self.__data_path = data_path
        self.__archive_name = archive_name
        self.__log = log


    @property
    def log( self ):

        return self.__log


    @property
    def data_path( self ):

        return self.__data_path


    @property
    def archive_name( self ):

        return self.__archive_name

        
    def saved_object_file_name( self, object_type, object_id, epoch = None ):

        epoch_tag = "-" + str( epoch ) if epoch is not None else ""
        object_tag = "-" + str( object_id ) if object_id is not None else ""
        filename = self.archive_name + epoch_tag + "-" + object_type + object_tag
        return self.data_path + "/" + filename


    def save_array_output( self, data, object_type, object_id = None, epoch = None ):

        if not os.path.exists( self.data_path ):
            os.mkdir( self.data_path )

        filepath = self.saved_object_file_name( object_type, object_id, epoch )
        numpy.save( filepath, data, allow_pickle=False )

        self.log.item( "saved " + object_type + " to " + filepath )
        return filepath


    def save_model_parameters( self, parameter_map, object_id = None, epoch = None ):

        if not os.path.exists( self.data_path ):
            os.mkdir( self.data_path )

        filepath = self.saved_object_file_name( 'model', object_id, epoch )
        numpy.savez_compressed( filepath, **parameter_map )

        self.log.item( "saved " + object_type + " to " + filepath )
        return filepath



#---------------------------------------------------------------------------------------------------


class SegmentationResults( object ):


    def __init__(
            self,
            data_path,
            results_id,
            epoch,
            class_count,
            label_conversion,
            log = output.Log() ):

        self.__log = log

        self.__epoch = epoch
        self.__archive = Archive( data_path, results_id, log )

        self.__label_conversion = label_conversion
        self.__class_count = class_count

        self.__dice_scores = numpy.zeros(( 0, class_count ))
        self.__confusion_matrices = numpy.zeros(( 0, class_count, class_count ))

        self.__predicted_labels = []
        self.__predicted_masks = []

        self.__reference_labels = []
        self.__reference_masks = []


    @property
    def epoch( self ):

        return self.__epoch


    @property
    def archive( self ):

        return self.__archive


    @property
    def predicted_label_samples( self ):

        return self.__predicted_labels


    @property
    def reference_label_samples( self ):

        return self.__reference_labels


    @property
    def predicted_mask_samples( self ):

        return self.__predicted_masks


    @property
    def reference_mask_samples( self ):

        return self.__reference_masks


    @property
    def mean_dice_score_per_class( self ):

        return numpy.mean( self.__dice_scores, axis = 1 )


    @property
    def mean_confusion( self ):

        return numpy.mean( self.__confusion_matrices, axis = 0 )
    

    def append_and_save(
            self, volume_id, predicted_distribution, reference_distribution, target_shape ):

        self.append( predicted_distribution, reference_distribution, target_shape )

        self.archive.save_array_output( predicted_distribution, 'predicted', volume_id, self.epoch )
        self.archive.save_array_output( reference_distribution, 'reference', volume_id, self.epoch )


    def append( self, predicted_distribution, reference_distribution, target_shape ):

        distribution_to_labels = self.__label_conversion.labels_for_volumes
        labels_to_masks = labels.dense_volume_indices_to_dense_volume_masks

        predicted_labels = distribution_to_labels( predicted_distribution, target_shape )
        reference_labels = distribution_to_labels( reference_distribution, target_shape )
        self.__predicted_labels.append( Images.sample_images( predicted_labels ) )
        self.__reference_labels.append( Images.sample_images( reference_labels ) )

        predicted_masks = labels_to_masks( predicted_labels, self.__class_count )
        reference_masks = labels_to_masks( reference_labels, self.__class_count )
        self.__predicted_masks.append( Images.sample_images( predicted_masks ) )
        self.__reference_masks.append( Images.sample_images( reference_masks ) )

        dice_scores = Metrics.mean_dice_score_per_class( predicted_masks, reference_masks ) 
        self.__dice_scores = numpy.append( self.__dice_scores, [dice_scores], axis = 0 )

        confusion_matrix = Metrics.confusion_matrix( predicted_masks, reference_masks )
        self.__confusion_matrices = numpy.append(
            self.__confusion_matrices, [confusion_matrix], axis = 0 )


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


    @staticmethod
    def confusion_matrix( predicted_masks, reference_masks ):

        assert len( predicted_masks.shape ) == 4
        assert predicted_masks.shape == reference_masks.shape

        class_count = predicted_masks.shape[0]
        return numpy.array(
            [ [ numpy.count_nonzero( reference_masks[ i ] & predicted_masks[ j ] )
                for i in range( class_count ) ]
              for j in range( class_count ) ] )


#---------------------------------------------------------------------------------------------------

class Images:


    @staticmethod
    def sample_images( volume ):

        positions = ( 0.25, 0.5, 0.75 )
        shape = numpy.array( volume.shape )
        offsets = [ [ shape[i] * positions[j] for j in range(3) ] for i in range(3) ]
        samples = [
            [ volume[ offsets[0, j], :, : ] for j in range(3) ],
            [ volume[ :, offsets[1, j], : ] for j in range(3) ],
            [ volume[ :, :, offsets[2, j] ] for j in range(3) ]]
        return samples


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
