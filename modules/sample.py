#===================================================================================================
# volumetic sampling module



#---------------------------------------------------------------------------------------------------

import copy
import math
import numpy
import numpy.random

from geometry import cuboid, voxel
import output

import pdb


#---------------------------------------------------------------------------------------------------

class Parameters( object ) :


    def __init__(
            self,
            volume_count = 1,
            patch_count = 1,
            patch_shape = ( 1, 1, 1 ),
            patch_stride = 1,
            target_bounds = ( 1, 1, 1 ),
            window_margin = 0,
            seed = 42,
            constrain_to_mask = True ):

        self.volume_count = volume_count
        self.patch_count = patch_count
        self.patch_shape = patch_shape
        self.patch_stride = patch_stride
        self.constrain_to_mask = constrain_to_mask
        self.targe_bounds = target_bounds
        self.window_margin = 0
        self.seed = seed


    def __str__( self ) :

        return ( "parameters {\n" +
                 "volume_count      : " + str( self.volume_count ) + "\n" +
                 "patch_shape       : " + str( self.patch_shape ) + "\n" +
                 "patch_stride      : " + str( self.patch_stride ) + "\n" +
                 "constrain_to_mask : " + str( self.constrain_to_mask ) + "\n" +
                 "window_margin     : " + str( self.window_margin ) + "\n}" )


    def with_volume_count( self, volume_count ) :

        other = copy.copy( self )
        other.volume_count = volume_count
        return other


    def with_patch_count( self, patch_count ) :

        other = copy.copy( self )
        other.patch_count = patch_count
        return other


    def with_patch_shape( self, patch_shape ) :

        other = copy.copy( self )
        other.patch_shape = patch_shape
        return other


    def with_patch_stride( self, patch_stride ) :

        other = copy.copy( self )
        other.patch_stride = patch_stride
        return other


    def with_target_bounds( self, target_bounds ) :

        other = copy.copy( self )
        other.target_bounds = target_bounds
        return other


    def with_constrain_to_mask( self, constrain_to_mask ) :

        other = copy.copy( self )
        other.constrain_to_mask = constrain_to_mask
        return other


    def with_window_margin( self, window_margin ) :

        other = copy.copy( self )
        other.window_margin = window_margin
        return other
 

    def with_seed( self, seed ) :

        other = copy.copy( self )
        other.seed = seed
        return other


#---------------------------------------------------------------------------------------------------

class PatchSet( object ):


    @staticmethod
    def normalised_bounds_of_unmasked_regions( volumes ) :

        bounds = numpy.array( [ volume.unmasked_bounds for volume in volumes ] )

        unnormalised_minima = bounds[ :, 0, : ]
        unnormalised_maxima = bounds[ :, 1, : ]
        centres = voxel( 0.5 * ( unnormalised_minima + unnormalised_maxima ) )

        spans = unnormalised_maxima - unnormalised_minima
        maximum_span = numpy.amax( spans, 0 )

        normalised_minima = voxel( centres - numpy.ceil( 0.5 * maximum_span ) )
        normalised_maxima = voxel( centres + numpy.ceil( 0.5 * maximum_span ) )

        return numpy.transpose(
            numpy.array( ( normalised_minima, normalised_maxima ) ),
            axes=(1, 0, 2) )


    @staticmethod
    def offsets( bounds, patch_shape, patch_stride ) :

        assert len( patch_shape ) == 3

        minima = bounds[ 0 ]
        maxima = bounds[ 1 ] - ( patch_shape - voxel( 1, 1, 1 ) )

        grid = numpy.mgrid[
            minima[ 0 ] : maxima[ 0 ] + 1 : patch_stride,
            minima[ 1 ] : maxima[ 1 ] + 1 : patch_stride,
            minima[ 2 ] : maxima[ 2 ] + 1 : patch_stride ]

        count = numpy.product( grid.shape[ 1:5 ] )
        offsets = grid.reshape( 3, count ).T
        return offsets


    @staticmethod
    def extract( volume_data, offsets_per_volume, patch_shape ) :

        assert len( patch_shape ) == 3

        patches = numpy.array(
            [ [ volume_data[ volume_index ][
                offsets[ 0 ] : offsets[ 0 ] + patch_shape[ 0 ],
                offsets[ 1 ] : offsets[ 1 ] + patch_shape[ 1 ],
                offsets[ 2 ] : offsets[ 2 ] + patch_shape[ 2 ] ]
                for offsets in offsets_for_volume ]
              for volume_index, offsets_for_volume in enumerate( offsets_per_volume ) ] )

        return patches


    def __init__( self, volumes, offsets_per_volume, patch_shape, log = output.Log() ) :

        log.entry( "constructing batch" )
        self.__patch_offsets = offsets_per_volume

        log.item( "extracting image patches" )
        image_data = [ volume.images for volume in volumes ]
        self.__image_patches = PatchSet.extract( image_data, offsets_per_volume, patch_shape )

        log.item( "extracting label patches" )
        label_data = [ volume.labels for volume in volumes ]
        self.__label_patches = PatchSet.extract( label_data, offsets_per_volume, patch_shape )

        log.item( "extracting mask patches" )
        mask_data = [ volume.masks for volume in volumes ]
        self.__mask_patches = PatchSet.extract( mask_data, offsets_per_volume, patch_shape )
    

    @property
    def patch_offsets( self ) :

        return self.__patch_offsets


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

class RandomPatchSet( PatchSet ):
    '''
    A patch set constructed by randomly sampling the possible patches in a subset of volumes.
    
    The sampling algorithm proceeds as follows:

      * consider the full set of patches across all volumes laid out in a grid, with 
        rows corresponding to volumes, and columns to patches within a volume
    
      * now randomise the order of patches in each row, using a different seed for each
        (for this we simply use the global seed plus the index of the row)
    
      * we now partition both the rows and the columns of the grid as follows:

          - group consecutive rows into subsets according to the parameter volume count
            these are the volumes which will be processed together in each batch
    
          - group consecutive columns according to the parameter patch_count divided by
            the volume_count

      * the subsets formed from the interscetion of a row and column group will consist of 
        approximately patch_count patches drawn from volume_count volumes. 

      * the exact number may be less than patch_count if:
        
          - the number of volumes is not divisible by volume_count, in which case
            patch sets drawn from final subset of rows will have fewer elements

          - the number of patches in a volume is not divisible by patch_count / volume_count
            in which case subsets drawn from the final subset of columns will have 
            fewer elements

        Patch sets drawn from subsets which are neither in the last rows or columns will
        always have exactly patch_count elements
    
      * the order of iteration is by column, and then by row within that column, i.e. we
        iterate through each patch set in the first column, and then in the second, etc.
        This ensures that the set of volumes is sampled as evenly as possible.  In the code
        we use the term iteration to mean a single pass through the set of volumes.

    '''

    @staticmethod
    def volumes_for_batch( aquisitions, batch, batches_per_iteration, volumes_per_batch ) :

        batch_in_iteration = batch % batches_per_iteration
        start = batch_in_iteration * volumes_per_batch

        maybe_end = ( batch_in_iteration + 1 ) * volumes_per_batch
        end = maybe_end if maybe_end <= len( aquisitions ) else len( aquisitions )

        volumes = [ aquisition.read_volume() for aquisition in aquisitions[ start : end ] ]
        return volumes


    @staticmethod
    def random_subset_of_offsets(
            offsets,
            batch,
            batches_per_iteration,
            patches_per_volume_per_batch,
            random_generator ):

        patches_in_volume = len( offsets )
        number_of_batches_drawn_from_this_volume = math.floor( batch / batches_per_iteration )

        start = number_of_batches_drawn_from_this_volume * patches_per_volume_per_batch
        maybe_end = ( number_of_batches_drawn_from_this_volume + 1 ) * patches_per_volume_per_batch
        end = maybe_end if maybe_end <= patches_in_volume else patches_in_volume

        random_offsets = random_generator.permutation( offsets )
        return random_offsets[ start : end ]
    

    def __init__( self, aquisitions, batch, parameters, random_generator, log = output.Log() ):

        volume_count = len( aquisitions )
        volumes_per_batch = parameters.volume_count
        batches_per_iteration = math.ceil( volume_count / float( volumes_per_batch ) )
        patches_per_volume_per_batch = math.floor( parameters.patch_count / volumes_per_batch )

        volumes = RandomPatchSet.volumes_for_batch(
            aquisitions,
            batch,
            batches_per_iteration,
            volumes_per_batch )

        bounds = PatchSet.normalised_bounds_of_unmasked_regions( volumes )

        random_offsets_per_volume = [
            RandomPatchSet.random_subset_of_offsets(
                PatchSet.offsets(
                    bounds[ i ],
                    parameters.patch_shape,
                    parameters.patch_stride ),
                batch,
                batches_per_iteration,
                patches_per_volume_per_batch,
                random_generator )
            for i, volume in enumerate( volumes ) ]

        super( RandomPatchSet, self ).__init__(
            volumes,
            random_offsets_per_volume,
            parameters.patch_shape,
            log )


#---------------------------------------------------------------------------------------------------

class ContiguousPatchSet( PatchSet ):
    '''
    A set of patches sampled from contiguous regions in one or more volumes.

    The sampled regions are densely sampled up to the specified patch
    stride, and the order over patches is preserved.  The significance
    of order preservation is that given a stride equal to the patch
    size, simply reshaping the resulting patches will restore the
    original data.

    '''

    @staticmethod
    def volume_and_patch_index_for_batch( batch, parameters ):

        target_bounds = numpy.array( parameters.target_bounds )
        patch_grid_shape = numpy.ceil( target_bounds / parameters.patch_stride )
        patches_per_volume = numpy.prod( patch_grid_shape )
        patches_per_batch = parameters.patch_count

        preceding_patch_count = patches_per_batch * batch
        volume = math.floor( preceding_patch_count / patches_per_volume )
        patch_index_in_volume = preceding_patch_count % patches_per_volume
        return ( volume, patch_index_in_volume )


    @staticmethod
    def target_bounds( volume, parameters ):

        bounds = volume.unmasked_bounds if parameters.constrain_to_mask else volume.bounds 
        centre = numpy.floor( bounds / 2 )
        lower_span = numpy.floor( parameters.target_bounds / 2 )
        upper_span = parameters.target_bounds - ( lower_span + 1 )
        target = numpy.array(( centre - lower_span, centre + upper_span ))

        def correction( axis ):
            
            offset_to_minimum = target[0][axis] - bounds[0][axis]
            offset_to_maximum = bounds[1][axis] - target[1][axis]

            if offset_to_maximum + offset_to_minimum < 0:
                raise Exception( "cannot fit target", target, "to bounds", bounds )

            elif offset_to_minimum < 0:
                corrected_minimum = bounds[0][axis]
                corrected_maximum = target[1][axis] - offset_to_minimum
                return ( corrected_minimum, corrected_maximum )

            elif offset_to_maximum < 0:
                corrected_minimum = target[0][axis] - offset_to_maximum
                corrected_maximum = bounds[1][axis]
                return ( corrected_minimum, corrected_maximum )

            else:
                return ( target[0][axis], target[1][axis] )

        corrected = numpy.array([ correction( axis ) for axis in range( 0, 3 ) ]).T
        return corrected
    

    @staticmethod
    def subset_of_offsets( offsets, volume, start, end ):

        start_volume, start_patch_for_batch = start
        start_patch = start_patch_for_batch if volume == start_volume else 0
        
        end_volume, end_patch_for_batch = end
        end_patch = end_patch_for_batch if volume == end_volume else len( offsets )

        return offsets[ start_patch : end_patch ]
        

    def __init__( self, aquisitions, batch, parameters, log = output.Log() ):

        start = ContiguousPatchSet.volume_and_patch_index_for_batch( batch, parameters )
        end = ContiguousPatchSet.volume_and_patch_index_for_batch( batch + 1, parameters )

        volumes = [
            aquisition.read_volume() for aquisition in aquisitions[ start[0] : end[0] ] ]

        offsets_per_volume = [
            ContiguousPatchSet.subset_of_offsets( 
                PatchSet.offsets(
                    ContiguousPatchSet.target_bounds( volumes[ i ], parameters ),
                    parameters.patch_shape,
                    parameters.patch_stride ),
                i,
                start,
                end )
            for i in range( start, end ) ]

        super( ContiguousPatchSet, self ).__init__(
            volumes,
            offsets_per_volume,
            parameters.patch_shape,
            log )


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
