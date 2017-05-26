#===================================================================================================
# experiment tests



#---------------------------------------------------------------------------------------------------

import os
import tempfile
import unittest

import numpy

import experiment
import labels


#---------------------------------------------------------------------------------------------------

class ResultsTests( unittest.TestCase ):


    def test_save_array_output( self ):

        I = J = K = range( 0, 100 )
        original_data = numpy.array( [ [ [ k*100 + j*10 + i for i in I ] for j in J ] for k in K ] )

        with tempfile.TemporaryDirectory() as base_path:
            temp_path = base_path + "/new-subdirectory"
            conversion = labels.DenseLabelConversions( 3 )
            parameters = experiment.Parameters( "42", temp_path, 1, 0.01, 3 )
            results = experiment.Results( conversion, parameters )
            tag = "save-array-test"

            results.save_array_output( original_data, tag )
            reconstituted_data = numpy.load( results.saved_object_file_name( tag ) )
            self.assertTrue( numpy.array_equal( reconstituted_data, original_data ) )

            results.save_array_output( original_data, tag, 42 )
            reconstituted_data_42 = numpy.load( results.saved_object_file_name( tag, 42 ) )
            self.assertTrue( numpy.array_equal( reconstituted_data_42, original_data ) )


    def test_on_epoch_event( self ):

        classes = 2
        I = J = range( 0, 20 )
        K = V = range( 0, 1 )
        patch_grid_shape = numpy.array( ( 1, 20, 20 ) )
        validation_labels = numpy.array( [
            [ [ [ ( 1.0, 0.0 ) if i < 10 else ( 0.0, 1.0 ) for i in I ] for j in J ] for k in K ]
            for v in V ] )
        validation_output = numpy.array( [
            [ [ [ ( 0.8, 0.2 ) if i < 12 else ( 0.2, 0.8 ) for i in I ] for j in J ] for k in K ]
            for v in V ] )
        validation_cost = 0.2
        training_costs = [ 0.5, 0.3, 0.2, 0.15, 0.125 ]
        dice_scores = [ ( 2.0 * 10 ) / ( 2.0 * 10 + 2 + 0 ), ( 2.0 * 8 ) / ( 2.0 * 8 + 0 + 2 ) ]
        model = [ 6, 9, 42 ]

        with tempfile.TemporaryDirectory() as experiment_path:

            epoch = 0
            conversion = labels.SparseLabelConversions( classes )
            parameters = experiment.Parameters( "42", experiment_path, 1, 0.01, classes )
            results = experiment.Results( conversion, parameters )

            results.on_epoch_event(
                epoch,
                model,
                patch_grid_shape,
                validation_labels.reshape( 1, 1 * 20 * 20, classes ),
                validation_output.reshape( 1, 1 * 20 * 20, classes ),
                validation_cost,
                training_costs )

            self.assertTrue( os.path.exists( results.saved_object_file_name( "model", epoch ) ) )
            self.assertTrue( os.path.exists( results.saved_object_file_name( "output", epoch ) ) )
            self.assertTrue( os.path.exists( results.saved_object_file_name( "labels", epoch ) ) )
            self.assertTrue( os.path.exists( results.saved_object_file_name( "masks", epoch ) ) )
            for c in range( 0, classes ):
                self.assertTrue( os.path.exists( results.saved_object_file_name( "class-"+str( c ), epoch ) ) )

            self.assertEqual( results.training_costs, [ training_costs ] )
            self.assertEqual( results.validation_costs, [ validation_cost ] )
            self.assertEqual( results.dice_scores, [ dice_scores ] )



#---------------------------------------------------------------------------------------------------

if __name__ == '__main__' :

    unittest.main()

#---------------------------------------------------------------------------------------------------
