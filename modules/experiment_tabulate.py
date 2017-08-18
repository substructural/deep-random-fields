import importlib
import inspect
import math
import optparse
import os
import sys
import traceback

import numpy

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
exit_on_error( options.epoch is None, 'the epoch argument is required' )


metric = arguments[0]
experiments_to_run = arguments[1:]

#---------------------------------------------------------------------------------------------------

def instance( experiment_pattern ):

    experiment_name = os.path.basename( experiment_pattern.replace( '.py', '' ) )
    experiment_module = importlib.import_module( experiment_name )
    log = output.Log( sys.stdout )

    initial_epoch = options.epoch if options.epoch else 0
    model_seed = options.model_seed if options.model_seed else 42
    return experiment.SegmentationByDenseInferenceExperiment(
        experiment_module.Definition(),
        options.input,
        options.output,
        initial_epoch = initial_epoch,
        transfer_layers = options.transfer_layers,
        model_seed = model_seed,
        log = log )
    

def metrics_for_instance( i, epoch, metric ):

    archive = i.archive()
    data = archive.read_array_output( 'results', epoch = epoch )
    return data[ metric ]
    

def results_for_instances( patterns, epoch, metric ):
    
    instances = [ instance( pattern ) for pattern in patterns ]
    metrics = [ metrics_for_instance( i, epoch, metric ) for i in instances ]
    means = [ numpy.mean( m ) for m in metrics ]
    errors = [ numpy.std( m ) / math.sqrt( len( m ) ) for m in metrics ]

    header = f'{metric}\n\nexperiment id & mean & std error \\'
    rows = '\n'.join(
        [ f'{instances[i].definition.experiment_id} & {means[i]} & \pm {errors[i]} \\'
          for i in range( len( metrics ) ) ] )
    return header + '\n' + rows


def all_results_for_instances( patterns, epoch ):

    metrics = [ 'mean_dice' ] + [ f'dice_for_class_{i}' for i in range( 4 ) ]
    tables = [ results_for_instances( patterns, epoch, m ) for m in metrics ]
    return '\n\n'.join( tables )


#---------------------------------------------------------------------------------------------------

try:

    if metric == 'all':
        print( all_results_for_instances( experiments_to_run, options.epoch ) )

    else:
        print( results_for_instances( experiments_to_run, options.epoch, metric ) )
    

except Exception as e:

    exception_type, exception_value, backtrace = sys.exc_info()
    print( f'\n\ncommand {command} failed for {experiment_name}' )
    print( f'\n\nerror:\n\n{e}\n' )
    traceback.print_tb( backtrace )
    if options.debug:
        print( '\nentering debugger...\n\n' )
        ipdb.set_trace()


#---------------------------------------------------------------------------------------------------
