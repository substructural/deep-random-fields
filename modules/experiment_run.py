import importlib
import inspect
import sys

import experiments
import output
import report

#---------------------------------------------------------------------------------------------------


experiment_name = sys.argv[ 1 ] 
operation = sys.argv[ 2 ]
oasis_input_path = sys.argv[ 3 ]
output_base_path = sys.argv[ 4 ]

experiment = importlib.import_module( experiment_name )
experiment_instance = experiments.SegmentationByDenseInferenceExperiment(
    experiment.Definition(),
    oasis_input_path,
    output_base_path,
    output.Log( sys.stdout ) )

if operation == 'train':

    experiment_instance.run( seed = 42 ) 

if operation == 'report':

    epoch = int( sys.argv[5] )
    report.Report.generate( epoch, experiment_instance  )


#---------------------------------------------------------------------------------------------------
