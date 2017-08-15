import importlib
import inspect
import optparse
import os
import sys
import traceback

import ipdb

import experiment
import output
import report

#---------------------------------------------------------------------------------------------------

usage = 'usage: %prog [options] command experiment_1 [ ... experiment_n ]'
command_line = optparse.OptionParser( usage )
command_line.add_option( '-i', '--input', action = 'store' )
command_line.add_option( '-o', '--output', action = 'store' )
command_line.add_option( '-e', '--epoch', action = 'store', type = 'int' )
command_line.add_option( '-t', '--transfer-layers', action = 'store', type = 'int', default = 0 )
command_line.add_option( '-s', '--seed', action = 'store', type = 'int' )
command_line.add_option( '-m', '--model-seed', action = 'store', type = 'int' )
command_line.add_option( '-d', '--debug', action = 'store_true', default = False )

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

exit_on_error( command == 'report' and options.epoch is None,
               'the epoch argument is required for the report command' )

#---------------------------------------------------------------------------------------------------

for experiment_to_run in experiments_to_run:

    try:
        experiment_name = os.path.basename( experiment_to_run.replace( '.py', '' ) )
        experiment_module = importlib.import_module( experiment_name )
        log = output.Log( sys.stdout )

        initial_epoch = options.epoch if options.epoch else 0
        model_seed = options.model_seed if options.model_seed else 42
        experiment_seed = options.seed if options.seed else 54
        experiment_instance = experiment.SegmentationByDenseInferenceExperiment(
            experiment_module.Definition(),
            options.input,
            options.output,
            initial_epoch = initial_epoch,
            transfer_layers = options.transfer_layers,
            model_seed = model_seed,
            log = log )

        if command == 'training':

            experiment_instance.run( seed = experiment_seed ) 

        if command == 'report':

            epoch = options.epoch
            report.Report.generate( epoch, experiment_instance  )

        if command == 'source':

            source = inspect.getsource( type( experiment_instance.definition ) )
            print( source )

    except Exception as e:

        exception_type, exception_value, backtrace = sys.exc_info()
        print( f'\n\ncommand {command} failed for {experiment_name}' )
        print( f'\n\nerror:\n\n{e}\n' )
        traceback.print_tb( backtrace )
        if options.debug:
            print( '\nentering debugger...\n\n' )
            ipdb.set_trace()


#---------------------------------------------------------------------------------------------------
