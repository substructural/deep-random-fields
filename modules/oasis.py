#====================================================================================================
# oasis dataset module

import os
import glob
import random
import re
import sys

import nibabel
import matplotlib.pyplot as plot
import math
import numpy

import data 
import geometry
import output

import pdb

from math import floor

#----------------------------------------------------------------------------------------------------


class OasisAquisition( data.Aquisition ):


    @staticmethod
    def parse_aquisition_header( file_path ):

        descriptor = open( file_path )
        entry_pattern = re.compile( '([^:]+):[ \t]+(.*)' )
        entries = [ entry_pattern.match( line ) for line in descriptor ]
        key_value_pairs = { entry.group( 1 ) : entry.group( 2 ) for entry in entries if entry }
        return key_value_pairs


    @staticmethod
    def read_data_in_analyse_format( file_pattern ) :

        file_list = glob.glob( file_pattern )
        assert( len( file_list ) == 1 )

        meta_data = nibabel.analyze.load( file_list[ 0 ] )
        raw_data = meta_data.get_data()
        flipped_data = numpy.flipud( raw_data )
        rotated_data = numpy.transpose( flipped_data, ( 2, 1, 0, 3 ) )
        processed_data = rotated_data
        x, y, z, _ = processed_data.shape
        return processed_data.reshape( ( x, y, z ) )
        return raw_data
    

    def __init__( self, dataset_path, aquisition_id ):

        matched = re.search( 'OAS1_([0-9]+)_MR([0-9]+)', aquisition_id )
        assert( matched )
        subject_id = matched.group( 1 )
        aquisition_index = matched.group( 2 )

        aquisition_path = dataset_path + "/" + aquisition_id
        metadata = OasisAquisition.parse_aquisition_header( aquisition_path + '/' + aquisition_id + '.txt' )
        subject = data.Subject( subject_id, metadata[ 'M/F' ], metadata[ 'CDR' ] )
        super( OasisAquisition, self ).__init__( aquisition_id, subject, int( metadata[ 'AGE' ] ) )

        self.__metadata = metadata
        self.__aquisition_path = aquisition_path
    

    @property
    def aquisition_path( self ):

        return self.__aquisition_path


    def read_volume( self ):

        file_name_pattern = self.aquisition_id + "_mpr_n?_anon_111_t88_masked_gfc"

        image_file_pattern = self.aquisition_path + "/PROCESSED/MPRAGE/T88_111/" + file_name_pattern + ".img"
        image_data = OasisAquisition.read_data_in_analyse_format( image_file_pattern )

        label_pattern = self.aquisition_path + "/FSL_SEG/" + file_name_pattern + "_fseg.img"
        label_data = OasisAquisition.read_data_in_analyse_format( label_pattern )

        mask_data = image_data > 0 

        return data.Volume( image_data, label_data, mask_data )


#----------------------------------------------------------------------------------------------------


class OasisDataSet( data.Dataset ):


    def __init__( self, root_path, training_count, validation_count, testing_count, random_seed, maybe_log = None ):

        log = maybe_log if maybe_log else output.Log()
        log.subsection( "loading OASIS dataset" )
        
        log.entry( "scanning for aquisitions" )
        aquisition_ids = os.listdir( root_path )
        aquisitions_found = len( aquisition_ids )
        assert( aquisitions_found > 0 )

        log.entry( "loading aquisition headers" )
        aquisitions_requested = training_count + validation_count + testing_count 
        aquisitions_to_read = min( aquisitions_found, aquisitions_requested )        
        aquisitions = [ OasisAquisition( root_path, id ) for id in aquisition_ids ]

        super( OasisDataSet, self ).__init__(
           aquisitions, training_count, validation_count, testing_count, random_seed, log )


#----------------------------------------------------------------------------------------------------



if __name__ == '__main__' :

    file_path = sys.argv[ 1 ] 
    volume_count = int( sys.argv[ 2 ] )
    dataset = OasisDataSet( file_path, volume_count, 0, 0, 42 )

    for aquisition in dataset.training_set :

        print( str( aquisition ) + "\n" )


    volume = dataset.training_set[ 0 ].read_volume()
    centre = volume.centre
    centre_image = volume.images[ centre[ 0 ], :, : ]
    centre_label = volume.labels[ centre[ 0 ], :, : ]
    centre_mask = volume.masks[ centre[ 0 ], :, : ]
    
    figure1 = plot.figure()
    figure1.add_subplot( 2, 2, 1 )
    plot.imshow( centre_image )
    figure1.add_subplot( 2, 2, 2 )
    plot.imshow( centre_label )
    figure1.add_subplot( 2, 2, 3 )
    plot.imshow( centre_mask )
    #plot.show()

    parameters = data.Parameters( 
        volume_count = volume_count,
        patch_shape = geometry.voxel( 1, 32, 32 ),
        patch_stride = 4 )
    
    batch = data.Batch( dataset.training_set, 0, parameters )

    figure2 = plot.figure()
    for i in range( 1, 12 + 1 ) :
        image_shape = batch.image_patches.shape[ 3:5 ]
        figure2.add_subplot( 3, 4, i )
        plot.imshow( batch.image_patches[ 0, 16000 + i, 0, :, : ].reshape( image_shape ) )

    plot.show()

