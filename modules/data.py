#---------------------------------------------------------------------------------------------------
# data representation



from datetime import datetime

import copy
import numpy as N
import theano as T
import random

from geometry import cuboid, voxel
import output

import pdb

#---------------------------------------------------------------------------------------------------

class Subject( object ) :


    def __init__( self, subject_id, gender, pathology ) :

        self.__subject_id = subject_id
        self.__gender = gender
        self.__pathology = pathology


    @property
    def subject_id( self ) :

        return self.__subject_id


    @property
    def gender( self ) :

        return self.__gender


    @property
    def pathology( self ) :

        return self.__pathology


    def __str__( self ) :

        return (
            "subject-id    : " + str( self.subject_id ) + "\n" +
            "gender        : " + str( self.gender ) + "\n" +
            "pathology     : " + str( self.pathology ) + "\n" )



#---------------------------------------------------------------------------------------------------

class Aquisition( object ) :


    def __init__( self, aquisition_id, subject, subject_age_at_aquisition ):

        self.__aquisition_id = aquisition_id
        self.__subject = subject
        self.__subject_age_at_aquisition = subject_age_at_aquisition


    def read_volume( self ) :

        raise NotImplementedError()


    @property
    def aquisition_id( self ) :

        return self.__aquisition_id


    @property
    def subject( self ) :

        return self.__subject


    @property
    def subject_age_at_aquisition( self ) :

        return self.__subject_age_at_aquisition


    def __str__( self ) :

        return (
            "aquisition-id : " + str( self.aquisition_id ) + "\n" +
            str( self.subject ) +
            "age at scan   : " + str( self.subject_age_at_aquisition ) + "\n" )



#---------------------------------------------------------------------------------------------------

class Volume( object ):


    def __init__( self, image_data, label_data, mask_data ) :

        self.__image_data = image_data
        self.__label_data = label_data
        self.__mask_data = mask_data


    @property
    def images( self ) :

        return self.__image_data


    @property
    def labels( self ) :

        return self.__label_data


    @property
    def masks( self ) :

        return self.__mask_data


    @property
    def dimensions( self ) :

        return self.masks.shape


    @property
    def centre( self ) :

        minima, maxima = self.unmasked_bounds
        d = maxima - minima
        return voxel( minima + ( 0.5 * d ) )


    @property
    def unmasked_bounds( self ) :

        mask_reduced_in_x = N.any( self.masks, ( 0, 1 ) )
        mask_reduced_in_y = N.any( self.masks, ( 0, 2 ) )
        mask_reduced_in_z = N.any( self.masks, ( 1, 2 ) )

        indices_of_unmasked_in_x = N.nonzero( mask_reduced_in_x )[ 0 ]
        indices_of_unmasked_in_y = N.nonzero( mask_reduced_in_y )[ 0 ]
        indices_of_unmasked_in_z = N.nonzero( mask_reduced_in_z )[ 0 ]

        min_x = indices_of_unmasked_in_x[ 0 ]
        min_y = indices_of_unmasked_in_y[ 0 ]
        min_z = indices_of_unmasked_in_z[ 0 ]

        max_x = indices_of_unmasked_in_x[ -1 ]
        max_y = indices_of_unmasked_in_y[ -1 ]
        max_z = indices_of_unmasked_in_z[ -1 ]

        return cuboid( ( min_z, min_y, min_x ), ( max_z, max_y, max_x ) )



#---------------------------------------------------------------------------------------------------

class Dataset( object ) :


    def __init__(
            self,
            aquisitions,
            training_count,
            validation_count,
            testing_count,
            random_seed,
            maybe_log = None ) :

        assert( len( aquisitions ) >= training_count + validation_count + testing_count )
        log = maybe_log if maybe_log else output.Log()
        log.subsection( "partitioning dataset" )

        log.entry( "sorting by subject" )
        subjects = [ s for s in set( [ a.subject.subject_id for a in aquisitions ] ) ]
        aquisitions_for = lambda s : [ a for a in aquisitions if a.subject.subject_id == s ]
        aquisitions_by_subject = { s : aquisitions_for( s ) for s in subjects }

        subsets = [ [], [], [] ]
        targets = [ training_count, validation_count, testing_count ]

        log.entry( "allocating subjects to training, validation and test" )
        random.seed( random_seed )
        for subject in random.sample( subjects, len( subjects ) ) :

            aquisitions_to_add = aquisitions_by_subject[ subject ]
            for subset in range( 0, 3 ) :

                places_remaining = targets[ subset ] - len( subsets[ subset ] )
                if places_remaining >= len( aquisitions_to_add ) :
                    subsets[ subset ] += aquisitions_to_add
                    break

        log.entry( "randomising order" )
        self.__training_set = random.sample( subsets[ 0 ], len( subsets[ 0 ] ) )
        self.__validation_set = random.sample( subsets[ 1 ], len( subsets[ 1 ] ) )
        self.__testing_set = random.sample( subsets[ 2 ], len( subsets[ 2 ] ) )


    @property
    def training_set( self ) :

        return self.__training_set


    @property
    def validation_set( self ) :

        return self.__validation_set


    @property
    def test_set( self ) :

        return self.__testing_set



#---------------------------------------------------------------------------------------------------

class Batch( object ) :


    @staticmethod
    def volumes_for_batch( aquisitions, batch_index, parameters ) :

        start = batch_index * parameters.volume_count
        end = ( batch_index + 1 ) * parameters.volume_count
        volumes = [ aquisition.read_volume() for aquisition in aquisitions[ start : end ] ]
        return volumes


    @staticmethod
    def normalised_bounds_of_unmasked_regions( volumes ) :

        bounds = N.array( [ volume.unmasked_bounds for volume in volumes ] )

        unnormalised_minima = bounds[ :, 0, : ]
        unnormalised_maxima = bounds[ :, 1, : ]
        centres = voxel( 0.5 * ( unnormalised_minima + unnormalised_maxima ) )

        spans = unnormalised_maxima - unnormalised_minima
        maximum_span = N.amax( spans, 0 )

        normalised_minima = voxel( centres - N.ceil( 0.5 * maximum_span ) )
        normalised_maxima = voxel( centres + N.ceil( 0.5 * maximum_span ) )

        return N.transpose( N.array( ( normalised_minima, normalised_maxima ) ), axes=(1, 0, 2) )


    @staticmethod
    def offsets( volume_shape, unmasked_bounds, parameters ) :

        assert( len( parameters.patch_shape ) == 3 )

        outer_bounds = cuboid( ( 0, 0, 0 ), volume_shape )
        inner_bounds = unmasked_bounds if parameters.constrain_to_mask else outer_bounds

        minima = inner_bounds[ 0 ]
        maxima = inner_bounds[ 1 ] - ( parameters.patch_shape - voxel( 1, 1, 1 ) )

        grid = N.mgrid[
            minima[ 0 ] : maxima[ 0 ] + 1 : parameters.patch_stride,
            minima[ 1 ] : maxima[ 1 ] + 1 : parameters.patch_stride,
            minima[ 2 ] : maxima[ 2 ] + 1 : parameters.patch_stride ]

        count = N.product( grid.shape[ 1:5 ] )
        offsets = grid.reshape( 3, count ).T
        return offsets


    @staticmethod
    def patches( volume_data, offsets_per_volume, parameters ) :

        assert( len( parameters.patch_shape ) == 3 )

        patches = N.array(
            [ [ volume_data[ volume_index ][
                offsets[ 0 ] : offsets[ 0 ] + parameters.patch_shape[ 0 ],
                offsets[ 1 ] : offsets[ 1 ] + parameters.patch_shape[ 1 ],
                offsets[ 2 ] : offsets[ 2 ] + parameters.patch_shape[ 2 ] ]
                for offsets in offsets_for_volume ]
              for volume_index, offsets_for_volume in enumerate( offsets_per_volume ) ] )

        return patches


    @staticmethod
    def patch_grid_shape( bounds, parameters ) :

        outer_span = bounds[ 1 ] - bounds[ 0 ] + voxel( 1, 1, 1 )
        inner_span = outer_span - parameters.patch_shape
        grid_shape = 1 + ( inner_span / parameters.patch_stride ).astype( 'int32' )
        return grid_shape


    def __init__( self, aquisitions, batch_index, parameters, maybe_log = None ) :

        log = maybe_log if maybe_log else output.Log()
        log.entry( "constructing batch" )
        
        log.item( "acquiring volumes for batch" )
        volumes = Batch.volumes_for_batch( aquisitions, batch_index, parameters )

        log.item( "computing bounds" )
        bounds = Batch.normalised_bounds_of_unmasked_regions( volumes )

        log.item( "computing offsets" )
        offsets_per_volume = [
            Batch.offsets( v.images.shape, bounds[ i ], parameters ) for i, v in enumerate( volumes ) ]
        self.__patch_offsets = offsets_per_volume
        self.__patch_grid_shape = Batch.patch_grid_shape( bounds[ 0 ], parameters )

        log.item( "extracting image patches" )
        image_data = [ volume.images for volume in volumes ]
        self.__image_patches = Batch.patches( image_data, offsets_per_volume, parameters )

        log.item( "extracting label patches" )
        label_data = [ volume.labels for volume in volumes ]
        self.__label_patches = Batch.patches( label_data, offsets_per_volume, parameters )

        log.item( "extracting mask patches" )
        mask_data = [ volume.masks for volume in volumes ]
        self.__mask_patches = Batch.patches( mask_data, offsets_per_volume, parameters )


    @property
    def patch_offsets( self ) :

        return self.__patch_offsets


    @property
    def patch_grid_shape_for_batch( self ):

        return self.__patch_grid_shape


    @property
    def image_patches( self ) :

        return self.__image_patches


    @property
    def label_patches( self ) :

        return self.__label_patches


    @property
    def masks_patches( self ) :

        return self.__mask_patches



#---------------------------------------------------------------------------------------------------

class Accessor( object ) :


    def __init__(
            self,
            dataset,
            training_batch_parameters,
            label_conversion ) :

        self.__label_conversion = label_conversion
        self.__dataset = dataset
        self.__training_batch_parameters = training_batch_parameters
        self.__validation_batch_parameters = \
            training_batch_parameters.with_volume_count( len( dataset.validation_set ) )


    def images_and_labels( self, data_subset, batch_parameters, batch_index ) :

        batch = Batch( data_subset, batch_index, batch_parameters )
        label_distribution = self.__label_conversion.distribution_for_patches( batch.label_patches )
        return ( batch.image_patches, label_distribution, batch.patch_grid_shape )


    def training_images_and_labels( self, batch_index ) :

        return self.images_and_labels(
            self.__dataset.training_set, self.__training_batch_parameters, batch_index )


    def validation_images_and_labels( self, batch_index ) :

        return self.images_and_labels(
            self.__dataset.validation_set, self.__validation_batch_parameters, batch_index )




#---------------------------------------------------------------------------------------------------

class Parameters( object ) :


    def __init__(
            self,
            volume_count,
            patch_shape = ( 1, 1 ),
            patch_stride = 1,
            window_margin = 0,
            constrain_to_mask = True ):

        self.volume_count = volume_count
        self.patch_shape = patch_shape
        self.patch_stride = patch_stride
        self.constrain_to_mask = constrain_to_mask
        self.window_margin = 0


    def __str__( self ) :

        return ( "parameters {\n" +
                 "volume_count      : " + str( self.volume_count ) + "\n" +
                 "patch_shape       : " + str( self.patch_shape ) + "\n" +
                 "patch_stride      : " + str( self.patch_stride ) + "\n" +
                 "constrain_to_mask : " + str( self.constrain_to_mask ) + "\n" +
                 "window_margin     : " + str( self.window_margin ) + "\n}" )

    def with_volume_count( self, batch_volume_count ) :

        other = copy.copy( self )
        other.batch_volume_count = batch_volume_count
        return other


    def with_patch_shape( self, patch_shape ) :

        other = copy.copy( self )
        other.patch_shape = patch_shape
        return other


    def with_patch_stride( self, patch_stride ) :

        other = copy.copy( self )
        other.patch_stride = patch_stride
        return other


    def with_constrain_to_mask( self, constrain_to_mask ) :

        other = copy.copy( self )
        other.constrain_to_mask = constrain_to_mask
        return other


    def with_window_margin( self, window_margin ) :

        other = copy.copy( self )
        other.window_margin = window_margin
        return other



#---------------------------------------------------------------------------------------------------
