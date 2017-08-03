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

import ipdb

FLOAT_X = theano.config.floatX 



#---------------------------------------------------------------------------------------------------
# primitive dense patch functions


def dense_patch_indices_to_dense_patch_distribution( indices, index_count ) :

    patch_count = indices.shape[ 0 ]
    patch_shape = indices.shape[ 1 : ]

    distribution_shape = ( index_count, patch_count ) + patch_shape
    distribution = numpy.zeros( distribution_shape ).astype( FLOAT_X )

    for i in range( 0, index_count ) :
        distribution[ i ][ indices == i ] = 1.0

    permutation = [ i for i in range( 1, len( distribution_shape ) ) ] + [ 0 ]
    return numpy.transpose( distribution, permutation )



def dense_patch_distribution_to_dense_volume_distribution( dense_patches, volume_shape, margin ):

    patch_count = dense_patches.shape[ 0 ]
    patch_shape = numpy.array( dense_patches.shape[ 1 : -1 ] )
    index_count = dense_patches.shape[ -1 ]
    assert len( patch_shape ) == 3

    margin_loss = 2 * margin
    reconstructed_shape = numpy.array( volume_shape ) - margin_loss
    expected_voxel_count = numpy.prod( reconstructed_shape )
    voxel_count = patch_count * numpy.prod( patch_shape )
    assert voxel_count == expected_voxel_count

    grid_dimensions = reconstructed_shape // patch_shape
    grid_shape = tuple( int( grid_dimensions[ i ] ) for i in range( 0, len( grid_dimensions ) ) )
    grid_of_patches_shape = grid_shape + tuple( patch_shape ) + ( index_count, )
    grid_of_patches = dense_patches.reshape( grid_of_patches_shape )

    permutation = ( 0, 3, 1, 4, 2, 5, 6 )
    dense_volume_shape = tuple( reconstructed_shape ) + ( index_count, )
    dense_volume = numpy.transpose( grid_of_patches, permutation ).reshape( dense_volume_shape )

    return dense_volume



def dense_volume_distribution_to_dense_volume_indices( dense_volume_distribution ) :

    distribution_axis = len( dense_volume_distribution.shape ) - 1
    return numpy.argmax( dense_volume_distribution, axis = distribution_axis )



def dense_volume_indices_to_dense_volume_masks( dense_volume_indices, index_count ) :

    volume_shape = dense_volume_indices.shape

    masked = numpy.zeros( ( index_count, ) + volume_shape ).astype( 'int8' )
    for i in range( 0, index_count ) :
        masked[ i ][ dense_volume_indices == i ] = 1

    return masked



#---------------------------------------------------------------------------------------------------
# compound dense patch functions


def dense_patch_distributions_to_dense_volume_indices(
        dense_patch_distribution,
        target_shape,
        margin ):

    assert len( dense_patch_distribution.shape ) == 5

    dense_volume_distribution = dense_patch_distribution_to_dense_volume_distribution(
        dense_patch_distribution,
        target_shape,
        margin )

    return dense_volume_distribution_to_dense_volume_indices( dense_volume_distribution )



def dense_patch_indices_to_cropped_dense_patch_distributions( label_count, margin ):

    def distributions_for_patches( dense_patch_indices ):

        outer = dense_patch_indices.shape
        inner = tuple( slice( margin, span - margin ) for span in outer[1:] )
        batch = slice( 0, outer[0] )
        cropped = ( batch, ) + inner

        selected_patch_indices = dense_patch_indices[cropped] if margin else dense_patch_indices
        return dense_patch_indices_to_dense_patch_distribution(
            selected_patch_indices,
            label_count )

    return distributions_for_patches



#---------------------------------------------------------------------------------------------------
# primitive sparse patch functions


def dense_patch_distribution_to_sparse_patch_distribution( dense_patch_distribution ) :

    dimensions = dense_patch_distribution.shape
    patch_count = dimensions[ 0 ]
    patch_shape = dimensions[ 1 : -1 ]

    offset = tuple( int( d  / 2 ) for d in patch_shape )
    sparse = numpy.array( 
        [ dense_patch_distribution[ p ][ offset ]
          for p in range( 0, patch_count ) ])

    return sparse



def sparse_patch_distribution_to_dense_volume_distribution( sparse_patch_distribution, target_shape ) :

    dimensions = sparse_patch_distribution.shape
    assert( len( dimensions ) == 2 )
    assert( len( target_shape ) == 3 )

    patch_count = dimensions[ 0 ]
    label_count = dimensions[ 1 ]
    voxel_count = reduce( operator.mul, target_shape )
    assert( patch_count == voxel_count )

    k, j, i = target_shape
    dense_shape = ( k, j, i, label_count )
    return sparse_patch_distribution.reshape( dense_shape )



#---------------------------------------------------------------------------------------------------
# compound sparse patch functions


def sparse_patch_distribution_to_dense_volume_indices( sparse_patch_distribution, target_shape ):

    patch_count = sparse_patch_distribution.shape[ 1 ]
    assert patch_count == numpy.prod( target_shape )
    assert len( target_shape ) == 3

    dense_volume_distribution = sparse_patch_distribution_to_dense_volume_distribution(
        sparse_patch_distribution,
        target_shape )

    return dense_volume_distribution_to_dense_volume_indices( dense_volume_distribution )



def dense_patch_indices_to_sparse_patch_distributions( label_count ):

    def distributions_for_patches( dense_patch_indices ):

        dense_distributions = dense_patch_indices_to_dense_patch_distribution(
            dense_patch_indices,
            label_count )

        return dense_patch_distribution_to_sparse_patch_distribution( dense_distributions )

    return distributions_for_patches


#---------------------------------------------------------------------------------------------------
