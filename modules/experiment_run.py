import importlib
import inspect
import optparse
import os
import sys

import experiment
import output
import report

#---------------------------------------------------------------------------------------------------

usage = 'usage: %prog [options] command experiment_1 [ ... experiment_n ]'
command_line = optparse.OptionParser( usage )
command_line.add_option( '-i', '--input', action = 'store' )
command_line.add_option( '-o', '--output', action = 'store' )
command_line.add_option( '-e', '--epoch', action = 'store', type = 'int' )

#---------------------------------------------------------------------------------------------------

def exit_on_error( condition, message ):

    if condition:
        print( f'\n{message}\n' )
        command_line.print_help() 
        exit( 1 )

#---------------------------------------------------------------------------------------------------

options, arguments = command_line.parse_args()

exit_on_error( len( arguments ) < 1, 'you must specify a command to run' )
exit_on_error( len( arguments ) < 2, 'at least one experiment must be specified' )

exit_on_error( not options.input, 'the input path argument is required' )
exit_on_error( not options.input, 'the input path argument is required' )


command = arguments[0]
experiments_to_run = arguments[1:]

exit_on_error( command == 'report' and not options.epoch ,
               'the epoch argument is required for the report command' )

#---------------------------------------------------------------------------------------------------

for experiment_to_run in experiments_to_run:

    try:
        experiment_name = os.path.basename( experiment_to_run.replace( '.py', '' ) )
        experiment_module = importlib.import_module( experiment_name )
        experiment_instance = experiment.SegmentationByDenseInferenceExperiment(
            experiment_module.Definition(),
            options.input,
            options.output,
            output.Log( sys.stdout ) )

        if command == 'train':

            experiment_instance.run( seed = 42 ) 

        if command == 'report':

            epoch = options.epoch
            report.Report.generate( epoch, experiment_instance  )

        if command == 'print':

            source = inspect.getsource( type( experiment_instance ) )
            print( source )

    except Exception as e:

        print( f'\n\ncommand {command} failed for {experiment_name}\n\nerror:\n\n{e}\n\n' )


#---------------------------------------------------------------------------------------------------
