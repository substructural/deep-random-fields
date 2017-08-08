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
            target_shape = ( 1, 1, 1 ),
            window_margin = 0,
            seed = 42,
            constrain_to_mask = True ):

        self.volume_count = volume_count
        self.patch_count = patch_count
        self.patch_shape = patch_shape
        self.patch_stride = patch_stride
        self.constrain_to_mask = constrain_to_mask
        self.target_shape = target_shape
        self.window_margin = 0
        self.seed = seed


    def __str__( self ) :

        return ( "parameters {\n" +
                 "volume_count      : " + str( self.volume_count ) + "\n" +
                 "target_shape      : " + str( self.target_shape ) + "\n" +
                 "patch_shape       : " + str( self.patch_shape ) + "\n" +
                 "patch_stride      : " + str( self.patch_stride ) + "\n" +
                 "constrain_to_mask : " + str( self.constrain_to_mask ) + "\n" +
                 "window_margin     : " + str( self.window_margin ) + "\n}" )


    @property
    def reconstructed_shape( self ):

        target_shape = numpy.array( self.target_shape )
        margin_loss = numpy.array( self.window_margin ) * 2
        return target_shape - margin_loss


    @property
    def output_patch_shape( self ):

        margin_loss = numpy.array( self.window_margin ) * 2
        input_patch_shape = numpy.array( self.patch_shape )
        output_patch_shape = input_patch_shape - margin_loss

        patch_stride = numpy.ones( ( len( input_patch_shape ), ) ) * self.patch_stride
        assert numpy.array_equal( output_patch_shape, patch_stride )

        return output_patch_shape


    @property
    def patches_per_volume( self ):

        target_shape = numpy.array( self.target_shape )
        patch_shape = numpy.array( self.patch_shape )
        patches_per_axis = ( ( target_shape - patch_shape ) // self.patch_stride ) + 1
        return numpy.prod( patches_per_axis )


    def with_volume_count( self, volume_count ) :

        other = copy.copy( self )
        other.volume_count = volume_count
        return other


    def with_target_shape( self, target_shape ) :

        other = copy.copy( self )
        other.target_shape = target_shape
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
    def target_bounds( outer_bounds, inner_bounds, target_shape ):

        inner_shape = inner_bounds[1] - inner_bounds[0] + 1
        inner_span = numpy.floor( inner_shape / 2 ).astype( 'int64' )
        centre = inner_bounds[0] + inner_span

        lower_span = numpy.floor( target_shape / 2 ).astype( 'int64' )
        upper_span = target_shape - ( lower_span + 1 )
        target_bounds = numpy.array(( centre - lower_span, centre + upper_span ))

        def correction( axis ):

            offset_to_minimum = target_bounds[0][axis] - outer_bounds[0][axis]
            offset_to_maximum = outer_bounds[1][axis] - target_bounds[1][axis]

            if offset_to_maximum + offset_to_minimum < 0:
                raise Exception( "cannot fit target", target_shape, "to bounds", outer_bounds )

            elif offset_to_minimum < 0:
                shift_up = abs( offset_to_minimum )
                corrected_minimum = outer_bounds[0][axis]
                corrected_maximum = target_bounds[1][axis] + shift_up
                return ( corrected_minimum, corrected_maximum )

            elif offset_to_maximum < 0:
                shift_down = abs( offset_to_maximum )
                corrected_minimum = target_bounds[0][axis] - shift_down
                corrected_maximum = outer_bounds[1][axis]
                return ( corrected_minimum, corrected_maximum )

            else:
                return target_bounds[ :, axis]

        corrected = numpy.array([ correction( axis ) for axis in range( 0, 3 ) ]).T
        return corrected


    @staticmethod
    def target_region_offset_from_patch_offsets( offsets_per_volume ):

        return [ numpy.min( offsets, axis = 0 ) for offsets in offsets_per_volume ]


    @staticmethod
    def offsets_in_volume( bounds, patch_shape, patch_stride ) :

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
    def extract( volume_data, volume_offset, patch_offsets, patch_shape ) :

        assert len( patch_shape ) == 3

        patches = numpy.array([
            volume_data[ v - volume_offset ][
                z : z + patch_shape[ 0 ],
                y : y + patch_shape[ 1 ],
                x : x + patch_shape[ 2 ] ]
            for v, z, y, x in patch_offsets  ] )

        return patches


    def __init__(
            self,
            volumes,
            volume_offset,
            patch_offsets_per_volume,
            patch_shape,
            log = output.Log() ) :

        log.entry( "constructing batch" )

        log.item( "updating patch offsets" )
        patch_offsets = numpy.array(
            [ ( v + volume_offset, z, y, x )
              for v, offsets in enumerate( patch_offsets_per_volume )
              for z, y, x in offsets ] ).astype( 'int64' )
        self.__patch_offsets = patch_offsets

        log.item( "extracting image patches" )
        image_data = [ volume.images for volume in volumes ]
        self.__image_patches = PatchSet.extract(
            image_data, volume_offset, patch_offsets, patch_shape )

        log.item( "extracting label patches" )
        label_data = [ volume.labels for volume in volumes ]
        self.__label_patches = PatchSet.extract(
            label_data, volume_offset, patch_offsets, patch_shape )

        log.item( "extracting mask patches" )
        mask_data = [ volume.masks for volume in volumes ]
        self.__mask_patches = PatchSet.extract(
            mask_data, volume_offset, patch_offsets, patch_shape )


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

      * the subsets formed from the intersection of a row and column group will consist of 
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
        return volumes, start


    @staticmethod
    def random_subset_of_offsets(
            offsets_in_volume,
            batch,
            batches_per_iteration,
            patches_per_volume_per_batch,
            random_generator ):

        patches_in_volume = len( offsets_in_volume )
        number_of_batches_drawn_from_this_volume = math.floor( batch / batches_per_iteration )

        start = number_of_batches_drawn_from_this_volume * patches_per_volume_per_batch
        maybe_end = ( number_of_batches_drawn_from_this_volume + 1 ) * patches_per_volume_per_batch
        end = maybe_end if maybe_end <= patches_in_volume else patches_in_volume

        random_offsets_in_volume = random_generator.permutation( offsets_in_volume )
        return random_offsets_in_volume[ start : end ]


    def __init__( self, aquisitions, batch, parameters, random_generator, log = output.Log() ):

        volume_count = len( aquisitions )
        volumes_per_batch = parameters.volume_count
        batches_per_iteration = math.ceil( volume_count / float( volumes_per_batch ) )
        patches_per_volume_per_batch = math.floor( parameters.patch_count / volumes_per_batch )

        volumes, volume_offset = RandomPatchSet.volumes_for_batch(
            aquisitions,
            batch,
            batches_per_iteration,
            volumes_per_batch )

        random_offsets_per_volume = [
            RandomPatchSet.random_subset_of_offsets(
                PatchSet.offsets_in_volume(
                    PatchSet.target_bounds(
                        volume.bounds,
                        volume.unmasked_bounds,
                        numpy.array( parameters.target_shape ) ),
                    parameters.patch_shape,
                    parameters.patch_stride ),
                batch,
                batches_per_iteration,
                patches_per_volume_per_batch,
                random_generator )
            for i, volume in enumerate( volumes ) ]

        super( RandomPatchSet, self ).__init__(
            volumes,
            volume_offset,
            random_offsets_per_volume,
            parameters.patch_shape,
            log )


#---------------------------------------------------------------------------------------------------

class SequentialPatchSet( PatchSet ):
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

        patch_shape = numpy.array( parameters.patch_shape )
        target_shape = numpy.array( parameters.target_shape )
        stride_space = target_shape - patch_shape
        stride_space_is_negative = numpy.prod( stride_space ) < 0

        if stride_space_is_negative:

            raise Exception( 'patch shape exceeds free space in volume' )

        else:

            patch_grid_shape = numpy.floor( stride_space / parameters.patch_stride ) + 1
            patches_per_volume = numpy.prod( patch_grid_shape ).astype( 'int64' )
            patches_per_batch = parameters.patch_count

            preceding_patch_count = patches_per_batch * batch
            volume = math.floor( preceding_patch_count / patches_per_volume )
            patch_index_in_volume = preceding_patch_count % patches_per_volume

            return ( volume, patch_index_in_volume )


    @staticmethod
    def subset_of_offsets( offsets, volume, start, end ):

        start_volume, start_patch_for_batch = start
        start_patch = start_patch_for_batch if volume == start_volume else 0

        end_volume, end_patch_for_batch = end
        end_patch = end_patch_for_batch if volume == end_volume else len( offsets )

        return offsets[ start_patch : end_patch ]


    def __init__( self, aquisitions, batch, parameters, log = output.Log() ):

        start = SequentialPatchSet.volume_and_patch_index_for_batch( batch, parameters )
        start_volume = start[ 0 ]
        start_in_batch = ( 0, start[1] )

        end = SequentialPatchSet.volume_and_patch_index_for_batch( batch + 1, parameters )
        #end_patch = end[ 1 ]
        last_volume = end[ 0 ] #if end_patch == 0 else end[ 0 ] + 1
        end_in_batch = ( last_volume - start_volume, end[1] )

        volumes = [
            aquisition.read_volume() for aquisition in aquisitions[ start_volume : last_volume + 1 ] ]

        offsets_per_volume = [
            PatchSet.offsets_in_volume(
                PatchSet.target_bounds(
                    volume.bounds,
                    volume.unmasked_bounds,
                    numpy.array( parameters.target_shape ) ),
                parameters.patch_shape,
                parameters.patch_stride )
            for volume in volumes ]

        offsets_per_volume_in_this_batch = [
            SequentialPatchSet.subset_of_offsets(
                offsets_per_volume[ i ],
                i,
                start_in_batch,
                end_in_batch )
            for i in range( len( volumes ) ) ]

        

        print(  '> sequential patch set' )
        print( f'| - batch {batch} start:{start} start_in_batch:{start_in_batch}' )
        print( f'| - batch {batch} end:{end} end_in_batch:{end_in_batch}' )
        print( f'| - offsets per volume: {len(offsets_per_volume)} {len(offsets_per_volume[0])}' )
        print( f'| - offsets per volume in batch: {len(offsets_per_volume_in_this_batch)} {len(offsets_per_volume_in_this_batch[0])}' )

        super( SequentialPatchSet, self ).__init__(
            volumes,
            start_volume,
            offsets_per_volume_in_this_batch,
            parameters.patch_shape,
            log )


#---------------------------------------------------------------------------------------------------


class Accessor( object ) :


    def __init__(
            self,
            aquisitions,
            sample_parameters,
            image_normalisation,
            label_conversion,
            log = output.Log() ) :

        self.__image_nomalisation = image_normalisation
        self.__label_conversion = label_conversion
        self.__sample_parameters = sample_parameters
        self.__aquisitions = aquisitions
        self.__log = log

        volume_count = len( self.aquisitions )
        patches_per_batch = self.sample_parameters.patch_count

        total_patch_count = volume_count * sample_parameters.patches_per_volume
        batch_count = int( math.ceil( float( total_patch_count ) / patches_per_batch ) )
        self.__length = batch_count


    @property
    def aquisitions( self ):

        return self.__aquisitions
        

    @property
    def sample_parameters( self ):

        return self.__sample_parameters
        

    @property
    def log( self ):

        return self.__log


    def label_conversion( self, labels ):

        return self.__label_conversion( labels ) 


    def image_normalisation( self, images ):
        
        return self.__image_nomalisation( images )
        

    def patch_set( self, batch ):

        raise NotImplementedError()


    def __len__( self ):

        assert self.__length >= 0
        return int( self.__length )
    

    def __iter__( self ):

        class Iterator( object ):

            def __init__( self, accessor ):
                self.batch = 0
                self.accessor = accessor

            def __next__( self ):
                if self.batch < len( self.accessor ):
                    patches = self.accessor.patch_set( self.batch )
                    images = self.accessor.image_normalisation( patches.image_patches )
                    labels = self.accessor.label_conversion( patches.label_patches )
                    offsets = patches.patch_offsets
                    print(  '> accessor' )
                    print( f'| - batch {self.batch + 1} of {len(self.accessor)}' )
                    print( f'| - offsets: {len(offsets)}' )
                    print( f'| - images: {len(images)}' )
                    print( f'| - labels: {len(labels)}' )
                    self.batch += 1
                    return ( images, labels, offsets )
                else:
                    raise StopIteration()

        return Iterator( self )


#---------------------------------------------------------------------------------------------------


class SequentialAccessor( Accessor ):


    def patch_set( self, batch ):

        return SequentialPatchSet(
            self.aquisitions,
            batch,
            self.sample_parameters,
            self.log )


#---------------------------------------------------------------------------------------------------


class RandomAccessor( Accessor ):


    def __init__(
            self,
            aquisitions,
            sample_parameters,
            image_normalisation,
            label_conversion,
            random_generator,
            log = output.Log() ) :

        self.__random_generator = random_generator
        super( RandomAccessor, self ).__init__(
            aquisitions,
            sample_parameters,
            image_normalisation,
            label_conversion,
            log )


    @property
    def random_generator( self ):

        return self.__random_generator
    

    def patch_set( self, batch ):

        return RandomPatchSet(
            self.aquisitions,
            batch,
            self.sample_parameters,
            self.random_generator,
            self.log )


#---------------------------------------------------------------------------------------------------
