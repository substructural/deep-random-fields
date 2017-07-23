#===================================================================================================
# class label manipulations

'''
Module for conversion between different value and spatial formats for image labels.

There are two aspects to the format of a set of class labels:

  * the value representation:

      * indices -- an integer identifying the label in a canonical order
      * distributions -- a sequence of probabilities for each label
      * masks -- a sequence of binary values, equal to 1 at the index, and 0 elsewhere

  * the spatial representation:

      * dense volume -- the labels for each voxel in the underlying volume
      * dense patches -- the labels for each voxel in a patch
      * sparse patches -- the label for the voxel at the centre of a patch

Most but not all combinations of these two aspects are applicable at some point in the training
or evaluation of a network:

                    +-------------------+-------------------+-------------------+
                    | masks             | indices           | distributions     |
  +-----------------+-------------------+-------------------+-------------------+
  |                 |                   |                   |                   |
  | dense / volume  | segmentation      | input/evaluation  | probability maps  |
  |                 |                   |                   |                   |
  +-----------------+-------------------+-------------------+-------------------|
  |                 |                   |                   |                   |
  | dense / patch   |                   | input/training    | output/dense      |
  |                 |                   |                   |                   |
  +-----------------+-------------------+-------------------+-------------------|
  |                 |                   |                   |                   |
  | sparse / patch  |                   |                   | output/sparse     |
  |                 |                   |                   |                   |
  +-----------------+-------------------+-------------------+-------------------+

Note that three combinatinons, dense patch masks, sparse patch indices and sparse patch
masks, are not required for any stage in training or evaluation, and will thus be ignored
here.  We present the sequence of transformations from one format to the next as required
for training a network whose final classification layers are non-convolutional below:

                    +-------------------+-------------------+-------------------+
                    | masks             | indices           | distributions     |
  +-----------------+-------------------+-------------------+-------------------+
  |                 |                   |                   |                   |
  | dense / volume  |         6 <--------------- 0/5 <----------------- 4       |
  |                 |                   |         |         |           ^       |
  +-----------------+-------------------+---------|---------+-----------|-------|
  |                 |                   |         V         |           |       |
  | dense / patch   |                   |         1 ----------------> 2 |       |
  |                 |                   |                   |         | |       |
  +-----------------+-------------------+-------------------+---------|-|-------|
  |                 |                   |                   |         V |       |
  | sparse / patch  |                   |                   |          3        |
  |                 |                   |                   |                   |
  +-----------------+-------------------+-------------------+-------------------+

The sequence for sense (fully convolutional) network training is similar, but avoids
the need to construct sparse per patch distributions:

                    +-------------------+-------------------+-------------------+
                    | masks             | indices           | distributions     |
  +-----------------+-------------------+-------------------+-------------------+
  |                 |                   |                   |                   |
  | dense / volume  |         5 <--------------- 0/4 <---------------- 3        |
  |                 |                   |         |         |          ^        |
  +-----------------+-------------------+---------|---------+----------|--------|
  |                 |                   |         V         |          |        |
  | dense / patch   |                   |         1 -----------------> 2        |
  |                 |                   |                   |                   |
  +-----------------+-------------------+-------------------+-------------------|
  |                 |                   |                   |                   |
  | sparse / patch  |                   |                   |                   |
  |                 |                   |                   |                   |
  +-----------------+-------------------+-------------------+-------------------+

In total this gives us the following transformations:

  * dense volume indices to dense patch indices
  * dense patch indices to dense patch distributions
  * dense patch distributions to dense volume distributions
  * dense patch distributions to sparse patch distributions
  * sparse patch distributions to dense volume distributions
  * dense volume distributions to dense volume masks
  * dense volume masks to dense volume indices

Of these, the first, from dense volume indices to dense patch indices, is already implemented
in the data.Batch class, as part of its sampling process (as the same transformation is applied
to the image data and valid region masks).  The remainder are implemented here.

'''


#---------------------------------------------------------------------------------------------------

from geometry import voxel

from functools import reduce

import operator
import numpy
import theano

import pdb


#---------------------------------------------------------------------------------------------------


def dense_patch_indices_to_dense_patch_distribution( indices, index_count ) :

    volume_count = indices.shape[ 0 ]
    patch_count = indices.shape[ 1 ]
    patch_shape = indices.shape[ 2 : ]

    distribution_shape = ( index_count, volume_count, patch_count, ) + patch_shape
    distribution = numpy.zeros( distribution_shape ).astype( theano.config.floatX )

    for i in range( 0, index_count ) :
        distribution[ i ][ indices == i ] = 1.0

    permutation = [ i for i in range( 1, len( distribution_shape ) ) ] + [ 0 ]
    return numpy.transpose( distribution, permutation )



def dense_patch_distribution_to_dense_volume_distribution( dense_patches, volume_shape ) :

    volume_count = dense_patches.shape[ 0 ]
    patch_count = dense_patches.shape[ 1 ]
    patch_shape = dense_patches.shape[ 2 : -1 ]
    index_count = dense_patches.shape[ -1 ]

    assert( len( patch_shape ) == 3 )
    assert( len( volume_shape ) == 3 )

    grid_dimensions = numpy.array( volume_shape ) / numpy.array( patch_shape )
    grid_shape = tuple( int( grid_dimensions[ i ] ) for i in range( 0, len( grid_dimensions ) ) )
    grid_of_patches_shape = ( volume_count, ) + grid_shape + patch_shape + ( index_count, )
    grid_of_patches = dense_patches.reshape( grid_of_patches_shape )

    permutation = ( 0, 1, 4, 2, 5, 3, 6, 7 )
    dense_volume_shape = ( volume_count, ) + volume_shape + ( index_count, )
    dense_volume = numpy.transpose( grid_of_patches, permutation ).reshape( dense_volume_shape )

    return dense_volume



def dense_volume_distribution_to_dense_volume_indices( dense_volume_distribution ) :

    distribution_axis = len( dense_volume_distribution.shape ) - 1
    return numpy.argmax( dense_volume_distribution, axis = distribution_axis )



def dense_volume_indices_to_dense_volume_masks( dense_volume_indices, index_count ) :

    volume_shape = dense_volume_indices.shape
    volume_dimensions = len( volume_shape )

    masked = numpy.zeros( ( index_count, ) + volume_shape ).astype( 'int8' )
    for i in range( 0, index_count ) :
        masked[ i ][ dense_volume_indices == i ] = 1

    group_by_mask = ( 1, 0 ) + tuple( i + 1 for i in range( 1, volume_dimensions ) )
    return numpy.transpose( masked, group_by_mask )



#---------------------------------------------------------------------------------------------------


def dense_patch_distribution_to_sparse_patch_distribution( dense_patch_distribution ) :

    dimensions = dense_patch_distribution.shape
    volume_count = dimensions[ 0 ]
    patch_count = dimensions[ 1 ]
    patch_shape = dimensions[ 2 : -1 ]
    label_count = dimensions[ -1 ]

    offset = tuple( int( d  / 2 ) for d in patch_shape )
    sparse = numpy.array( [
        [ [ dense_patch_distribution[ v, p ][ offset ]
            for p in range( 0, patch_count ) ]
        for v in range( 0, volume_count ) ] ] )

    return sparse



def sparse_patch_distribution_to_dense_volume_distribution( sparse_patch_distribution, volume_shape ) :

    dimensions = sparse_patch_distribution.shape
    assert( len( dimensions ) == 3 )
    assert( len( volume_shape ) == 3 )

    volume_count = dimensions[ 0 ]
    patch_count = dimensions[ 1 ]
    label_count = dimensions[ 2 ]
    voxel_count = reduce( operator.mul, volume_shape )
    assert( patch_count == voxel_count )

    k, j, i = volume_shape
    dense_shape = ( volume_count, k, j, i, label_count, )
    return sparse_patch_distribution.reshape( dense_shape )


#---------------------------------------------------------------------------------------------------

class DenseLabelConversions( object ):


    def __init__( self, label_count ):

        self.__label_count = label_count


    def distributions_for_patches( self, dense_patch_indices ):

        return dense_patch_indices_to_dense_patch_distribution(
            dense_patch_indices,
            self.__label_count )


    def labels_for_volumes( self, dense_patch_distribution, patch_grid_shape ):

        patch_count = dense_patch_distribution.shape[ 1 ]
        assert patch_count == numpy.prod( patch_grid_shape )
        assert len( dense_patch_distribution.shape ) == 6
        assert len( patch_grid_shape ) == 3

        patch_shape = dense_patch_distribution.shape[ 2:5 ]
        volume_shape = patch_grid_shape * patch_shape

        dense_volume_distribution = dense_patch_distribution_to_dense_volume_distribution(
            dense_patch_distribution,
            volume_shape )

        return dense_volume_distribution_to_dense_volume_indices( dense_volume_distribution )


#---------------------------------------------------------------------------------------------------

class SparseLabelConversions( object ):


    def __init__( self, label_count ):

        self.__label_count = label_count


    def distributions_for_patches( self, dense_patch_indices ):

        dense_distributions = dense_patch_indices_to_dense_patch_distribution(
            dense_patch_indices,
            self.__label_count )

        return dense_patch_distribution_to_sparse_patch_distribution( dense_distributions )


    def labels_for_volumes( self, sparse_patch_distribution, patch_grid_shape ):

        patch_count = sparse_patch_distribution.shape[ 1 ]
        assert patch_count == numpy.prod( patch_grid_shape )
        assert len( patch_grid_shape ) == 3

        dense_volume_distribution = sparse_patch_distribution_to_dense_volume_distribution(
            sparse_patch_distribution,
            patch_grid_shape )

        return dense_volume_distribution_to_dense_volume_indices( dense_volume_distribution )


#---------------------------------------------------------------------------------------------------
