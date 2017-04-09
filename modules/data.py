#---------------------------------------------------------------------------------------------------
# data representation


from datetime import datetime

import copy
import numpy as N
import theano as T
import random

from geometry import cuboid, voxel

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
        return geometry.voxel( minima + ( 0.5 * d ) )


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


    def __init__( self, aquisitions, training_count, validation_count, testing_count, random_seed ) : 

        assert( len( aquisitions ) >= training_count + validation_count + testing_count )

        subjects = [ s for s in set( [ a.subject.subject_id for a in aquisitions ] ) ]
        aquisitions_for = lambda s : [ a for a in aquisitions if a.subject.subject_id == s ] 
        aquisitions_by_subject = { s : aquisitions_for( s ) for s in subjects }

        subsets = [ [], [], [] ]
        targets = [ training_count, validation_count, testing_count ]

        random.seed( random_seed )
        for subject in random.sample( subjects, len( subjects ) ) :

            aquisitions_to_add = aquisitions_by_subject[ subject ]
            for subset in range( 0, 3 ) :

                places_remaining = targets[ subset ] - len( subsets[ subset ] )
                if places_remaining >= len( aquisitions_to_add ) :
                    subsets[ subset ] += aquisitions_to_add
                    break

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
    def offsets( volume, parameters ) :

        assert( len( parameters.patch_shape ) == 3 )

        outer_bounds = volume.images.shape
        inner_bounds = volume.unmasked_bounds if parameters.constrain_to_mask else outer_bounds
        minimum = inner_bounds[ 0 ]
        maximum = inner_bounds[ 1 ] - ( parameters.patch_shape - voxel( 1, 1, 1 ) ) 
        
        grid = N.mgrid[
            minimum[ 0 ] : maximum[ 0 ] + 1 : parameters.patch_stride,
            minimum[ 1 ] : maximum[ 1 ] + 1 : parameters.patch_stride,
            minimum[ 2 ] : maximum[ 2 ] + 1 : parameters.patch_stride ]

        count = N.product( grid.shape[ 1:5 ] )
        offsets = grid.reshape( 3, count ).T
        return offsets
        

    @staticmethod
    def patches( volume_data, offsets_per_volume, parameters ) :

        assert( len( parameters.patch_shape ) == 3 )
        
        patches = N.array( [
            volume_data[ volume_index ][ 
                offsets[ 0 ] : offsets[ 0 ] + parameters.patch_shape[ 0 ],
                offsets[ 1 ] : offsets[ 1 ] + parameters.patch_shape[ 1 ],
                offsets[ 2 ] : offsets[ 2 ] + parameters.patch_shape[ 2 ] ]
            for volume_index, offsets in offsets_per_volume ] )

        return patches


    def __init__( self, aquisitions, batch_index, parameters, seed = None ) : 

        volumes = Batch.volumes_for_batch( aquisitions, batch_index, parameters )
        offsets_per_volume = [
            ( i, offsets )
            for i in range( 0, len( volumes ) )
            for offsets in Batch.offsets( volumes[ i ], parameters ) ]

        if seed is not None :
            random.seed( seed )
            random.shuffle( offsets_per_volume )

        image_data = [ volume.images for volume in volumes ] 
        self.__image_patches = Batch.patches( image_data, offsets_per_volume, parameters )

        label_data = [ volume.labels for volume in volumes ] 
        self.__label_patches = Batch.patches( label_data, offsets_per_volume, parameters )

        mask_data = [ volume.masks for volume in volumes ] 
        self.__mask_patches = Batch.patches( mask_data, offsets_per_volume, parameters )


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

class Loader( object ) :


    @staticmethod
    def dense_to_patch_based_labels( dense_labels ) :

        shape = dense_labels.shape
        all_instances = slice( 0, shape[ 0 ], 1 )
        offset_to_centre = tuple( int( d / 2 ) for d in shape[ 1 : ] )
        patch_labels = dense_labels[ ( all_instances, ) + offset_to_centre ]
        return patch_labels
    

    @staticmethod
    def dense_labels_in_centered_window( dense_labels, margin ) :

        shape = dense_labels.shape
        all_instances = slice( 0, shape[ 0 ], 1 )
        window = tuple( slice( margin, d - margin, 1 ) for d in shape[ 1: ] )

        labels_in_window = dense_labels[ ( all_instances, ) + window ]
        return labels_in_window
    

    @staticmethod
    def labels_to_distribution( labels, label_count ) :

        batch_size = labels.shape[ 0 ]
        data_shape = labels.shape[ 1 : ]
        distribution_shape = ( label_count, batch_size, ) + data_shape 
        distribution = N.zeros( distribution_shape ).astype( T.config.floatX )

        for i in range( 0, label_count ) :
            distribution[ i ][ labels == i ] = 1.0

        permutation = [ i for i in range( 1, len( distribution_shape ) ) ] + [ 0 ]
        return N.transpose( distribution, permutation )
    

    @staticmethod
    def per_patch_distribution_over_labels( batch, label_count ) :
        
        labels = Loader.dense_to_patch_based_labels( batch.label_patches )
        label_distribution = Loader.labels_to_distribution( labels, label_count )
        return ( batch.image_patches, label_distribution )


    @staticmethod
    def dense_distribution_over_labels( batch, window_margin, label_count ) :
        
        labels = Loader.dense_labels_in_centered_window( batch.label_patches, window_margin )
        label_distribution = Loader.labels_to_distribution( labels, label_count )
        return ( batch.image_patches, label_distribution )


    def __init__( self, dataset, training_set_parameters, label_count, window_margin = 0 ) :

        self.__label_count = label_count
        self.__dataset = dataset
        self.__training_set_parameters = training_set_parameters
        self.__validation_set_parameters = (
            training_set_parameters.with_patch_stride( 1 ).with_volume_count( len( dataset.validation_set ) ) )


    @property
    def label_count( self ) :

        return self.__label_count


    @property
    def dataset( self ) :

        return self.__dataset


    def training_batch( self, index ) :

        return Batch( self.dataset.training_set, index, self.__training_set_parameters )


    def validation_set( self ) :

        return Batch( self.dataset.validation_set, 0, self.__validation_set_parameters )
    

    def load_training_batch_with_per_patch_distribution_over_labels( self, batch_index ) :

        return Loader.per_patch_distribution_over_labels( self.training_batch( batch_index ), self.label_count )


    def load_training_batch_with_dense_distribution_over_labels( self, batch_index ) :

        return Loader.dense_distribution_over_labels(
            self.training_batch( batch_index ), self.parameters.window_margin, self.label_count )


    def load_validation_set_with_per_patch_distribution_over_labels( self ):

        return Loader.per_patch_distribution_over_labels( self.validation_set(), self.label_count )


    def load_validation_set_with_dense_distribution_over_labels( self ) :

        return Loader.dense_distribution_over_labels(
            self.validation_set(), self.parameters.window_margin, self.label_count )


    def load_validation_set_with_dense_labels( self ) :

        validation_set = self.validation_set()
        labels = Loader.labels_in_centered_window( validation_set.label_patches, self.parameters.window_margin )
        return ( validation_set.image_patches, labels )



#---------------------------------------------------------------------------------------------------

class Parameters( object ) :


    def __init__(
            self,
            volume_count,
            patch_shape = ( 1, 1 ),
            patch_stride = 1,
            window_margin = 0,
            constrain_to_mask = True,
            use_dense_labels = True ) :

        self.volume_count = volume_count
        self.patch_shape = patch_shape
        self.patch_stride = patch_stride
        self.constrain_to_mask = constrain_to_mask
        self.window_margin = 0


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
