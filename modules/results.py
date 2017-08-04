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

import matplotlib.pyplot

import labels
import output

import ipdb


#---------------------------------------------------------------------------------------------------

class Archive( object ):
    

    def __init__( self, data_path, archive_name, log = output.Log() ):

        self.__data_path = data_path
        self.__archive_name = archive_name
        self.__log = log

        if not os.path.exists( data_path ):
            os.makedirs( data_path )


    @property
    def log( self ):

        return self.__log


    @property
    def data_path( self ):

        return self.__data_path


    @property
    def archive_name( self ):

        return self.__archive_name

        
    def saved_object_file_name( self, object_type, object_id = None, epoch = None ):

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

        object_type = 'model'
        filepath = self.saved_object_file_name( object_type, object_id, epoch )
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
            log ):

        self.__epoch = epoch
        self.__archive = Archive( data_path, results_id, log )

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
    

    def append_and_save( self, volume_id, predicted_distribution, reference_distribution ):

        self.append( predicted_distribution, reference_distribution )

        self.archive.save_array_output( predicted_distribution, 'predicted', volume_id, self.epoch )
        self.archive.save_array_output( reference_distribution, 'reference', volume_id, self.epoch )


    def append( self, predicted_labels, reference_labels ):

        self.__predicted_labels.append( Images.sample_images( predicted_labels ) )
        self.__reference_labels.append( Images.sample_images( reference_labels ) )

        labels_to_masks = labels.dense_volume_indices_to_dense_volume_masks
        predicted_masks = labels_to_masks( predicted_labels, self.__class_count )
        reference_masks = labels_to_masks( reference_labels, self.__class_count )
        self.__predicted_masks.append( Images.sample_images( predicted_masks ) )
        self.__reference_masks.append( Images.sample_images( reference_masks ) )

        dice_scores = Metrics.dice_scores_per_class(
            predicted_masks, reference_masks, self.__class_count )
        self.__dice_scores = numpy.append( self.__dice_scores, [dice_scores], axis = 0 )

        confusion_matrix = Metrics.confusion_matrix( predicted_masks, reference_masks )
        self.__confusion_matrices = numpy.append(
            self.__confusion_matrices, [confusion_matrix], axis = 0 )


#---------------------------------------------------------------------------------------------------

class Metrics:


    @staticmethod
    def dice_score( predicted, reference ):

        intersection_size = numpy.count_nonzero( predicted & reference )
        predicted_size = numpy.count_nonzero( predicted )
        reference_size = numpy.count_nonzero( reference )

        return ( 2.0 * intersection_size ) / ( predicted_size + reference_size )


    @staticmethod
    def dice_scores_per_class( predicted_masks, reference_masks, class_count ):

        dice_scores = [
            Metrics.dice_score( predicted_masks[c], reference_masks[c] )
            for c in range( class_count ) ] 
        return dice_scores


    @staticmethod
    def mean_dice_score_per_class( predicted_masks_per_volume, reference_masks_per_volume ):

        volumes = predicted_masks_per_volume.shape[ 0 ]
        classes = predicted_masks_per_volume.shape[ 1 ]
        dice = lambda v, c : Metrics.dice_score( predicted_masks_per_volume[ v, c ],
                                                 reference_masks_per_volume[ v, c ] )

        dice_per_volume_per_class = numpy.array(
            [ [ dice( v, c ) 
                for c in range( 0, classes ) ]
              for v in range( 0, volumes ) ] )

        sum_per_class = numpy.sum( dice_per_volume_per_class, axis = 0 ) 
        return sum_per_class * ( 1.0 / volumes )


    @staticmethod
    def confusion_matrix( predicted_masks, reference_masks ):

        assert len( predicted_masks ) == len( reference_masks )

        classes = len( predicted_masks )
        for c in range( classes ):
            assert predicted_masks[c].shape == reference_masks[c].shape

        return numpy.array(
            [ [ numpy.count_nonzero( reference_masks[ j ] & predicted_masks[ i ] )
                for i in range( classes ) ]
              for j in range( classes ) ] )


#---------------------------------------------------------------------------------------------------

class Images:


    @staticmethod
    def sample_images( volume ):

        positions = ( 0.25, 0.5, 0.75 )
        shape = numpy.array( volume.shape )
        offsets = numpy.array(
            [ [ int( shape[i] * positions[j] )
                for j in range(3) ]
              for i in range(3)
            ] )
        samples = [
            [ volume[ offsets[0, j], :, : ] for j in range(3) ],
            [ volume[ :, offsets[1, j], : ] for j in range(3) ],
            [ volume[ :, :, offsets[2, j] ] for j in range(3) ]]
        return samples


    @staticmethod
    def difference_of_masks( predicted, reference, include_true_negatives = False ):

        r, g, b = 0, 1, 2
        difference_shape = ( 3, ) + predicted.shape

        true_positives  = ( predicted & reference ) == 1
        false_positives = ( predicted & negation( reference ) ) == 1
        false_negatives = ( reference & negation( predicted ) ) == 1

        difference_map = numpy.zeros( difference_shape ).astype( 'uint8' )
        difference_map[ g ][ true_positives  ] = 0xFF
        difference_map[ r ][ false_positives ] = 0xFF
        difference_map[ b ][ false_negatives ] = 0xFF

        if include_true_negatives:
            true_negatives = negation( predicted | reference )
            difference_map[ :, true_negatives ] = 0xFF

        permutation_to_rgb_values = list( range( 1, len( difference_shape ) ) ) + [ 0 ]
        return numpy.transpose( difference_map, permutation_to_rgb_values )


    @staticmethod
    def overlay( greyscale_image, rgb_overlay):

        minimum = numpy.min( greyscale_image )
        maximum = numpy.max( greyscale_image )
        scale = 1.0 / ( maximum - minimum )
        normalised_greyscale_image = ( greyscale_image - minimum ).astype( 'float32' ) * scale

        rgb_axis = len( rgb_overlay.shape ) - 1
        to_rgb_axis_first = [ rgb_axis ] + list( range( 0, rgb_axis ) )
        to_rgb_axis_last = list( range( 1, rgb_axis + 1 ) ) + [ 0 ]

        rgb_overlay_per_colour = numpy.transpose( rgb_overlay, to_rgb_axis_first )
        rgb_overlay_image_per_colour = rgb_overlay_per_colour * normalised_greyscale_image
        rgb_overlay_image = numpy.transpose( rgb_overlay_image_per_colour, to_rgb_axis_last )

        return rgb_overlay_image.astype( 'uint8' )


    @staticmethod
    def save_image( image, file_path ):

        figure, axes = matplotlib.pyplot.subplots(1, 1)
        axes.imshow( image )
        axes.set_axis_off()

        figure.savefig( file_path, bbox_inches = 'tight', pad_inches = 0.0, transparent = True )



#---------------------------------------------------------------------------------------------------
