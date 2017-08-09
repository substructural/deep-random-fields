#===================================================================================================
# report



#---------------------------------------------------------------------------------------------------

import inspect
import os
import os.path

import labels
import output
from results import Images, SegmentationResults

import ipdb


#---------------------------------------------------------------------------------------------------


def html( experiment_name,
          content,
          page_foreground = '#000000',
          page_background = '#ffffff',
          image_table_foreground = '#ffffff',
          image_table_background = '#000000' ):

    return f'''

<html>
<style>

    html
    {{
        width: 80%;
        margin-left: auto;
        margin-right: auto;
    }}

    body 
    {{
        color: {page_foreground};
        background-color: {page_background};
    }}

    table.metrictable
    {{
        width: 80%;
        margin: 50px auto;
        text-align: left;
    }}

    img
    {{
        margin: 25px 25px;
        text-align: center;
        vertical-align: center;
    }}

    table.imagetable
    {{
        width: 80%;
        margin: 50px auto;
        color: {image_table_foreground};
        background-color: {image_table_background};
        text-align: center;
    }}

</style>

<body>
<h1>{experiment_name}</h1>

{content}

</body>
</html>

'''


#---------------------------------------------------------------------------------------------------


def source_section( source ):

    return f'''
    
<hr>
<h2>definition</h2>

<pre>
{source}
</pre>

'''


#---------------------------------------------------------------------------------------------------


def metric_section(
        metric_name,
        statistics,
        sample_images ): 

    value_table = statistics_table( statistics )

    mean_image_table    = image_table( 'mean '    + metric_name, sample_images.mean )
    median_image_table  = image_table( 'median '  + metric_name, sample_images.median )
    minimum_image_table = image_table( 'minimum ' + metric_name, sample_images.minimum )
    maximum_image_table = image_table( 'maximum ' + metric_name, sample_images.maximum )

    return f'''

<hr>
<h2>{metric_name}</h2>

{value_table}

{mean_image_table}
{median_image_table}
{minimum_image_table}
{maximum_image_table}

'''


#---------------------------------------------------------------------------------------------------


def statistics_table( statistics ):

    return f'''

<h3>overview</h3>
<table class="metrictable">

    <tr> <th> metric  </th> <th> value     </th> </tr>

    <tr> <td> mean    </td> <td> {statistics.mean[0]}    </td> </tr>
    <tr> <td> median  </td> <td> {statistics.median[0]}  </td> </tr>
    <tr> <td> minimum </td> <td> {statistics.minimum[0]} </td> </tr>
    <tr> <td> maximum </td> <td> {statistics.maximum[0]} </td> </tr>

</table>

'''


#---------------------------------------------------------------------------------------------------


def image_table( table_name, sample_images ):

    body = ( 
        image_row( 'axial', sample_images[ 0 ] ) + 
        image_row( 'coronal', sample_images[ 1 ] ) + 
        image_row( 'sagittal', sample_images[ 2 ] ) )

    return f'''

<h3>{table_name}</h3>
<table class="imagetable">

{body}

</table>

'''


def image_row( label_for_row, image_paths ):

    image_cells = ''.join( [ image_cell( p ) for p in image_paths ] )

    return f'''

    <tr>
        <td>{label_for_row}</td>
        {image_cells}    
    </tr>

'''


def image_cell( image_path ):

    return f'''
        <td><image src="{image_path}"></td> '''


#---------------------------------------------------------------------------------------------------


class SourceData( object ):


    @staticmethod
    def representative_volumes_for_metrics( metrics, dataset ):

        indices = set( index for metric in metrics for statistic, index in metric )
        volumes = { i: dataset.validation_set[ i ].read_volume() for i in indices }

        return volumes


    @staticmethod
    def representative_distributions_and_offsets_for_metrics( metrics, results ):

        indices = set( index for metric in metrics for statistic, index in metric )
        distribution_and_ofsets = {
            i: results.predicted_distribution( i ) for i in indices }

        distributions = { i : d for i, (d, o) in distribution_and_ofsets.items() }
        offsets = { i : o for i, (d, o) in distribution_and_ofsets.items() }

        return distributions, offsets


    @staticmethod
    def image_data_from_volumes( volumes, offsets, target_shape ):
        
        return { i : Images.extract( volumes[ i ].images, offsets[ i ], target_shape )
                 for i in volumes }


    @staticmethod
    def reference_labels_from_volumes( volumes, offsets, target_shape ):

        return { i : Images.extract( volumes[ i ].labels, offsets[ i ], target_shape )
                 for i in volumes }


    @staticmethod
    def predicted_labels_from_distributions( distributions ):

        return { i : labels.dense_volume_distribution_to_dense_volume_indices( d )
                 for i, d in distributions.items() }
    


#---------------------------------------------------------------------------------------------------

class Report( object ):


    @staticmethod
    def generate( epoch, experiment ):

        definition = experiment.definition
        results = SegmentationResults(
            experiment.output_path, definition.experiment_id, epoch, definition.label_count )

        results.restore( experiment.dataset, definition.sample_parameters, experiment.log )
        Report.write( results, experiment )


    @staticmethod
    def write( results, experiment ):

        log = experiment.log
        log.subsection( 'writing report' )
        
        class_count = results.class_count
        dataset = experiment.dataset
        sample_parameters = experiment.definition.sample_parameters
        reconstructed_shape = sample_parameters.reconstructed_shape

        log.entry( 'collating metrics' )
        dice = results.statistics_for_mean_dice_score_per_volume
        dice_per_class = [
            results.statistics_for_dice_score_for_class( i )
            for i in range( class_count ) ]
        metrics = [ dice ] + dice_per_class

        log.entry( 'loading data' )
        log.item( 'volumes' )
        volumes = SourceData.representative_volumes_for_metrics( metrics, dataset )
        log.item( 'distributions' )
        distributions, offsets = SourceData.representative_distributions_and_offsets_for_metrics(
            metrics,
            results )

        log.entry( 'extracting data')
        log.item( 'images' )
        image_data = SourceData.image_data_from_volumes( volumes, offsets, reconstructed_shape )
        log.item( 'reference distribution' )
        reference = SourceData.reference_labels_from_volumes( volumes, offsets, reconstructed_shape )
        log.item( 'predicted distribution' )
        predicted = SourceData.predicted_labels_from_distributions( distributions )

        log.entry( 'generating source section' )
        source_code = inspect.getsource( type( experiment.definition ) ) 
        source = source_section( source_code )

        log.entry( 'generating overview section' )
        archive = results.archive
        overview = Report.section_for_overview( dice, image_data, predicted, reference, results )

        log.entry( 'generating per class sections' )
        section_per_class = [
            Report.section_for_class(
                c, dice_per_class[ c ], image_data, predicted, reference, results )
            for c in range( class_count ) ]

        report_name = results.results_id.replace( '_', ' ' )
        file_content = html( report_name, source + overview + '\n'.join( section_per_class ) )
        file_name = archive.saved_object_file_name( 'report' ) + '.html' 

        log.entry( f'writing report to {file_name}' )
        with open( file_name, 'w' ) as output_file:
            output_file.write( file_content )

        log.entry( 'done' )
        return file_name


    @staticmethod
    def section_for_overview( statistics, image_data, predicted, reference, results ):
         
        name = f'dice over all classes'
        method = Images.sample_difference_of_multiple_masks
        return Report.section(
            name, statistics, image_data, predicted, reference, method, results )

    
    @staticmethod
    def section_for_class( c, statistics_for_c, image_data, predicted, reference, results ):
         
        name = f'dice for class {c}'
        method = lambda i, p, r, n : Images.sample_difference_of_masks( i, p, r, n, c )
        return Report.section(
            name, statistics_for_c, image_data, predicted, reference, method, results )

    
    @staticmethod
    def section( name, statistics, image_data, predicted, reference, sample_method, results ):
        
        names = statistics._fields

        images_per_statistic = [
            sample_method(
                image_data[ volume_index ],
                predicted[ volume_index ],
                reference[ volume_index ],
                results.class_count )
            for value, volume_index in statistics ]

        image_file_names_per_statistic = [
            Report.save_sample_images_for_statistic(
                images,
                f'{name}-{names[ statistic ]}',
                results.archive )
            for statistic, images in enumerate( images_per_statistic ) ]

        samples_indexed_by_statistic_name = Images.Samples( **{
            names[ statistic ] : image_file_names
            for statistic, image_file_names in enumerate( image_file_names_per_statistic ) } )
        
        return metric_section( name, statistics, samples_indexed_by_statistic_name )
        

    @staticmethod
    def save_sample_images_for_statistic( sample_images_for_statistic, section_name, archive ):

        axis_count = len( Images.Axes )
        position_count = len( Images.SamplePositions )
        assert sample_images_for_statistic.shape[ 0 : 2 ] == ( axis_count, position_count )
    
        prefix = 'report-' + section_name.replace( ' ', '_' )
        file_names = (
            [ [ archive.saved_object_file_name( prefix, f'{axis.name}-{position.name}.png' )
                for position in Images.SamplePositions ]
              for axis in Images.Axes ] )

        relative_file_names = (
            [ [ os.path.basename( file_name ) for file_name in row ] for row in file_names ] )

        for i in range( axis_count ):
            for j in range( position_count ):
                if os.path.exists( file_names[ i ][ j ] ):
                    os.remove( file_names[ i ][ j ] )
                Images.save_image( sample_images_for_statistic[ i ][ j ], file_names[ i ][ j ] )

        return relative_file_names


#---------------------------------------------------------------------------------------------------
