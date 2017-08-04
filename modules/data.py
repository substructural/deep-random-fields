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
    def bounds( self ):

        return cuboid( ( 0, 0, 0 ), self.dimensions )


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


class Normalisation:


    @staticmethod
    def transform_to_zero_mean_and_unit_variance(
            images,
            lower_percentile = 5,
            upper_percentile = 95 ):

        image_min = N.percentile( images, lower_percentile )
        image_max = N.percentile( images, upper_percentile )

        mask = N.ones( images.shape ) 
        mask[ images <= image_min ] = 0
        mask[ images >= image_max ] = 0

        count = N.sum( mask )
        mean = N.sum( images * mask ) / count
        variance = N.sum( ( ( images - mean ) * mask ) ** 2 ) / count

        return ( images - mean ) / variance
    


#---------------------------------------------------------------------------------------------------
