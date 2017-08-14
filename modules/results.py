#===================================================================================================
# results


# TODO:
#
#  - implement confusion matrix calculation
#  - update the dice score per class to use only one volume
#  - change the interface to the optimisation monitor
#  - update the experiment to use the new model, optimiser and monitors

#---------------------------------------------------------------------------------------------------

import math
import os
import os.path
import re

import enum
import collections

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
    def archive_path( self ):

        return self.data_path + "/" +  self.archive_name


    @property
    def archive_name( self ):

        return self.__archive_name


    def ensure_path( self ):

        if not os.path.exists( self.archive_path ):
            self.log.item( f"creating {self.archive_path}" )
            os.mkdir( self.archive_path )

        if not os.path.isdir( self.archive_path ):
            raise Exception( f'{self.archive_path} exists, but is not a directory' )

        
    def saved_object_file_name( self, object_type, object_id = None, epoch = None ):

        epoch_tag = "-epoch-" + str( epoch ) if epoch is not None else ""
        object_tag = "-" + str( object_id ) if object_id is not None else ""
        filename =  object_type + epoch_tag + object_tag
        filepath = self.data_path + "/" +  self.archive_name

        return filepath + "/" + filename


    def save_array_output( self, data, object_type, object_id = None, epoch = None ):

        self.ensure_path()

        filepath = self.saved_object_file_name( object_type, object_id, epoch ) + '.npz'
        if isinstance( data, dict ):
            numpy.savez_compressed( filepath, **data )
        else:
            numpy.savez( filepath, data, allow_pickle=False )

        self.log.item( "saved " + object_type + " to " + filepath )
        return filepath


    def read_array_output( self, object_type, object_id = None, epoch = None ):

        filepath = self.saved_object_file_name( object_type, object_id, epoch ) + '.npz'
        array = numpy.load( filepath, allow_pickle=False )

        self.log.item( "read " + object_type + " from " + filepath )
        return array


    def read_single_array_output( self, object_type, object_id = None, epoch = None ):

        with self.read_array_output( object_type, object_id, epoch ) as data:
            return data[ 'arr_0' ]


    def save_model_parameters( self, parameter_map, object_id = None, epoch = None ):

        self.ensure_path()

        object_type = 'model'
        filepath = self.saved_object_file_name( object_type, object_id, epoch )
        numpy.savez_compressed( filepath, **parameter_map )

        self.log.item( "saved " + object_type + " to " + filepath )
        return filepath


    def read_model_parameters( self, object_id = None, epoch = None ):

        filepath = self.saved_object_file_name( 'model', object_id, epoch ) + '.npz'
        with numpy.load( filepath ) as model_archive:
            self.log.item( "reading model from " + filepath )
            return { k : model_archive[ k ] for k in model_archive }


#---------------------------------------------------------------------------------------------------


class SegmentationResults( object ):


    def __init__(
            self,
            data_path,
            results_id,
            epoch,
            class_count,
            log = output.Log() ):

        self.__epoch = epoch
        self.__results_id = results_id
        self.__archive = Archive( data_path, results_id, log )
        self.__files = []

        self.__class_count = class_count

        self.__dice_scores = numpy.zeros(( 0, class_count ))
        self.__confusion_matrices = numpy.zeros(( 0, class_count, class_count ))

        self.__predicted_labels = []
        self.__predicted_masks = []

        self.__reference_labels = []
        self.__reference_masks = []


    def delete_from_archive( self ):

        for f in self.__files:
            os.remove( f )

        self.__files = []


    def predicted_distribution( self, volume_id ):

        with self.archive.read_array_output( 'segmentation', volume_id, self.epoch ) as data:
            predicted = data[ 'predicted' ]
            offset = data[ 'offset' ]
            return predicted, offset


    @property
    def results_id( self ):

        return self.__results_id


    @property
    def epoch( self ):

        return self.__epoch


    @property
    def archive( self ):

        return self.__archive


    @property
    def class_count( self ):

        return self.__class_count


    @property
    def confusion_matrices( self ):

        return self.__confusion_matrices


    @property
    def dice_scores_per_class( self ):

        return self.__dice_scores


    @property
    def statistics_for_mean_dice_score_per_volume( self ):

        mean_dice_scores_per_volume = numpy.mean( self.dice_scores_per_class, axis=1 )
        return Metrics.all_statistic_values_and_indices( mean_dice_scores_per_volume )


    def statistics_for_dice_score_for_class( self, class_index ):

        dice_scores_for_class = self.dice_scores_per_class[ :, class_index ]
        return Metrics.all_statistic_values_and_indices( dice_scores_for_class )


    def restore( self, dataset, sample_parameters, log = output.Log() ):

        log.subsection( 'querying results path' )

        margin = sample_parameters.window_margin
        reconstructed_shape = sample_parameters.reconstructed_shape
        file_names = os.listdir( self.archive.archive_path )

        pattern = os.path.basename(
            self.archive.saved_object_file_name( 'segmentation', '([0-9]+)', self.epoch ) )

        query = re.compile( pattern )
        queries = [ ( name, query.match( name ) ) for name in file_names ]
        sources = [ ( name, int( matched.groups()[0] ) ) for name, matched in queries if matched ]
        sources_sorted_by_id = sorted( sources, key = lambda s : s[1] )

        if not sources:
            raise Exception(
                f'no valid sources found.\n' +
                f'  - pattern: {pattern}\n' +
                f'  - path: {self.archive.data_path}' )

        log.entry( 'found:' )
        for name, volume_id in sources_sorted_by_id:

            log.item( f'{name} ({volume_id})' )

            predicted_distribution, offset = self.predicted_distribution( volume_id )
            volume = dataset.validation_set[ volume_id ].read_volume()
            offset_in_volume = offset[ 1 : ]
            reference_labels = Images.extract(
                volume.labels, offset_in_volume, reconstructed_shape, margin )
            reference_distribution = labels.dense_volume_indices_to_dense_volume_distribution(
                reference_labels, self.class_count )

            self.append( predicted_distribution, reference_distribution, volume_id )
    

    def append_and_save(
            self,
            volume_id,
            predicted_distribution,
            reference_distribution,
            offset_in_input ):

        assert volume_id == offset_in_input[0]
        self.append( predicted_distribution, reference_distribution, volume_id )

        # we do not save the reference as it assumed to exist already as part of the dataset
        data = {
            'predicted' : predicted_distribution,
            'offset'    : offset_in_input }
        name = self.archive.save_array_output( data, 'segmentation', volume_id, self.epoch )
        self.__files.append( name )


    def append( self, predicted_distribution, reference_distribution, volume_id ):

        assert volume_id == len( self.__dice_scores )
        assert volume_id == len( self.__confusion_matrices )
        #assert predicted_distribution.shape == reference_distribution.shape
        if predicted_distribution.shape != reference_distribution.shape:
            ipdb.set_trace()

        distribution_to_labels = labels.dense_volume_distribution_to_dense_volume_indices
        predicted_labels = distribution_to_labels( predicted_distribution )
        reference_labels = distribution_to_labels( reference_distribution )
        
        labels_to_masks = labels.dense_volume_indices_to_dense_volume_masks
        predicted_masks = labels_to_masks( predicted_labels, self.__class_count )
        reference_masks = labels_to_masks( reference_labels, self.__class_count )

        dice_scores = Metrics.dice_scores_per_class(
            predicted_masks, reference_masks, self.__class_count )
        self.__dice_scores = numpy.append( self.__dice_scores, [dice_scores], axis = 0 )

        confusion_matrix = Metrics.confusion_matrix( predicted_masks, reference_masks )
        self.__confusion_matrices = numpy.append(
            self.__confusion_matrices, [confusion_matrix], axis = 0 )


#---------------------------------------------------------------------------------------------------


Statistics = collections.namedtuple( 'Statistics', [ 'mean', 'median', 'minimum', 'maximum' ] )

CostStatistics = collections.namedtuple( 'CostStatistics', [
    'mean', 'median', 'minimum', 'maximum', 'deviation', 'change_from_base' ] )


class Metrics:


    @staticmethod
    def statistic_value_and_closest_index( values, statistic ):

        value = statistic( values )
        index = numpy.argmin( numpy.abs( values - value ) )
        return value, index


    @staticmethod
    def all_statistic_values_and_indices( values ):

        return Statistics(
            Metrics.statistic_value_and_closest_index( values, numpy.mean ),
            Metrics.statistic_value_and_closest_index( values, numpy.median ),
            Metrics.statistic_value_and_closest_index( values, numpy.min ),
            Metrics.statistic_value_and_closest_index( values, numpy.max ) )


    @staticmethod
    def dice_score( predicted, reference ):

        intersection_size = numpy.count_nonzero( predicted & reference )
        predicted_size = numpy.count_nonzero( predicted )
        reference_size = numpy.count_nonzero( reference )

        if predicted_size + reference_size > 0:
            return ( 2.0 * intersection_size ) / ( predicted_size + reference_size )
        else:
            return 0.0


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


    @staticmethod
    def multiclass_to_binary_confusion_matrix_for_class( c, multiclass_confusion ):

        tp = multiclass_confusion[ c, c ]
        fp = numpy.sum( multiclass_confusion[ :, c ] ) - tp
        fn = numpy.sum( multiclass_confusion[ c, : ] ) - tp
        tn = numpy.sum( multiclass_confusion ) - ( tp + fp + fn )

        return numpy.array(
            [ [ tp, fn ],
              [ fp, tn ] ] )


    @staticmethod
    def multiclass_to_binary_confusion_matrix( multiclass_confusion ):

        binary_confusion = Metrics.multiclass_to_binary_confusion_matrix_for_class
        classes = len( multiclass_confusion )
        return numpy.array(
            [ binary_confusion( c, multiclass_confusion ) for c in classes ] )


    @staticmethod
    def costs_over_experiment( costs, phases = 10 ):

        assert len( costs.shape ) == 2
        epochs = costs.shape[0]

        return [ Metrics.costs_over_epoch( costs[ epoch ], phases ) for epoch in range( epochs ) ]


    @staticmethod
    def costs_over_epoch( costs, phases = 10 ):

        assert len( costs.shape ) == 1
        count = len( costs )
        phase_span = count // phases
        phase_span_0 = count - ( phases - 1 ) * phase_span
        phase_extent = lambda i : phase_span_0 + i * phase_span

        baseline = Metrics.costs_over_phase( costs[ 0 : phase_span_0 ] )
        subsequent = [
            Metrics.costs_over_phase(
                costs[ phase_extent( i - 1 ) : phase_extent( i ) ],
                baseline )
            for i in range( 1, phases ) ] 

        return [ baseline ] + subsequent
    
        
    @staticmethod
    def costs_over_phase( costs, baseline = None ):

        mean = numpy.mean( costs )
        median = numpy.median( costs )
        minimum = numpy.min( costs )
        maximum = numpy.max( costs )
        standard_deviation = math.sqrt( numpy.var( costs ) )
        change_from_base = median / baseline.median if baseline else 0.0
        
        return CostStatistics(
            mean, median, minimum, maximum, standard_deviation, change_from_base )



#---------------------------------------------------------------------------------------------------

class Images:


    Samples = collections.namedtuple( 'SampleImages', [ 'mean', 'median', 'minimum', 'maximum' ] )


    class Axes( enum.Enum ):

        axial    = 0
        coronal  = 1
        sagittal = 2


    class SamplePositions( enum.Enum ):

        proximal = 0
        medial   = 1
        distal   = 2


    @staticmethod
    def sample_images( volume, maybe_positions = None ):

        positions = maybe_positions if maybe_positions else ( 0.3, 0.5, 0.7 )
        shape = numpy.array( volume.shape )

        offsets = numpy.array(
            [ [ int( ( shape[i] - 1 ) * positions[j] )
                for j in range(3) ]
              for i in range(3)
            ] )

        samples = numpy.array([
            [ volume[ offsets[0, j], :, : ] for j in range(3) ],
            [ volume[ :, offsets[1, j], : ] for j in range(3) ],
            [ volume[ :, :, offsets[2, j] ] for j in range(3) ]])

        return samples


    @staticmethod
    def sample_difference_of_masks(
            image_data,
            predicted_label_volume,
            reference_label_volume,
            class_count,
            class_index ):

        labels_to_masks = labels.dense_volume_indices_to_dense_volume_masks
        predicted_mask = labels_to_masks( predicted_label_volume, class_count )[ class_index ]
        reference_mask = labels_to_masks( reference_label_volume, class_count )[ class_index ]
        
        difference = Images.difference_of_masks( predicted_mask, reference_mask, True )
        overlay = Images.overlay( image_data, difference )
        return Images.sample_images( overlay )


    @staticmethod
    def sample_difference_of_multiple_masks(
            image_data,
            predicted_label_volume,
            reference_label_volume,
            class_count ):

        labels_to_masks = labels.dense_volume_indices_to_dense_volume_masks
        predicted_masks = labels_to_masks( predicted_label_volume, class_count )
        reference_masks = labels_to_masks( reference_label_volume, class_count )
        
        difference = Images.difference_of_multiple_masks( predicted_masks, reference_masks )
        overlay = Images.overlay( image_data, difference )
        return Images.sample_images( overlay )


    @staticmethod
    def extract( data, offset, extracted_shape, margin = 0 ):

        return data[
            offset[ 0 ] + margin : offset[ 0 ] + margin + extracted_shape[ 0 ],
            offset[ 1 ] + margin : offset[ 1 ] + margin + extracted_shape[ 1 ],
            offset[ 2 ] + margin : offset[ 2 ] + margin + extracted_shape[ 2 ] ]


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
    def difference_of_multiple_masks( predicted_masks, reference_masks ):

        class_count = predicted_masks.shape[ 0 ]
        difference_shape = ( 3, ) + predicted_masks.shape[ 1: ]
        difference_map = numpy.zeros( difference_shape ).astype( 'uint8' )

        for c in range( class_count ):
            true_positives  = ( predicted_masks[ c ] & reference_masks[ c ] ) == 1
            false_positives = ( predicted_masks[ c ] & negation( reference_masks[ c ] ) ) == 1

            negatives = negation( predicted_masks )
            tpc = numpy.count_nonzero( true_positives )
            fpc = numpy.count_nonzero( false_positives )
            nec = numpy.count_nonzero( negatives )

            if c < 3:
                difference_map[ c ][ true_positives  ] = 0xFF
                difference_map[ c ][ false_positives ] = 0x80
            else:
                rgb = ( c + 0 ) % 3
                mix = ( c + 1 ) % 3
                difference_map[ rgb ][ true_positives  ] = 0xFF
                difference_map[ mix ][ true_positives  ] = 0xFF
                difference_map[ rgb ][ false_positives ] = 0x80
                difference_map[ mix ][ false_positives ] = 0x80

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
        axes.imshow( numpy.flip( image, axis = 0 ) )
        axes.set_axis_off()

        figure.savefig( file_path, bbox_inches = 'tight', pad_inches = 0.0, transparent = True )
        figure.clear()
        matplotlib.pyplot.close()



#---------------------------------------------------------------------------------------------------
