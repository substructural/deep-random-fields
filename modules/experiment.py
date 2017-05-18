#===================================================================================================
# experiment framework


import theano
import numpy



#---------------------------------------------------------------------------------------------------

class Parameters( object ) :


    def __init__( self, epoch_count, cost_threshold, class_count ) :

        self.class_count = class_count
        self.cost_threshold = cost_threshold
        self.epoch_count = epoch_count




#---------------------------------------------------------------------------------------------------

class Experiment( object ) :


    def __init__(
            self,
            dataset,
            architecture,
            cost_function,
            optimiser,
            training_batch_parameters,
            validation_batch_parameters,
            experiment_parameters,
            seed = None ) :

        self.__dataset = dataset
        self.__model = network.Model( architecture, cost_function, optimiser )
        self.__training_batch_parameters = training_batch_parameters
        self.__validation_batch_parameters = validation_batch_parameters
        self.__parameters = experiment_parameters


    @property
    def parameters( self ) :

        return self.__parameters


    @property
    def dataset( self ) :

        return self.__dataset


    @property
    def model( self ) :

        return self.__model


    def training_images_and_labels( self, batch_index ) :

        batch = data.Batch( self.dataset.training_set, index, self.__training_batch_parameters )
        return ( batch.image_patches, self.format_labels( batch.label_patches ) )


    def validation_images_and_labels( self, batch_index ) :

        batch = data.Batch( self.dataset.validation_set, 0, self.__validation_batch_parameters )
        return ( batch.image_patches, self.format_labels( batch.label_patches ) )


    def format_labels( self, dense_patch_indices ):

        raise NotImplementedError


    def on_batch_event( self, batch_index, training_output, training_costs ) :

        raise NotImplementedError


    def on_epoch_event( self, epoch_index, validation_output, validation_costs, training_costs ) :

        raise NotImplementedError


    def run( self ) :

        training_set_size = len( self.dataset.training_set )
        volumes_per_batch = self.__training_batch_parameters.volume_count
        batch_count = training_set_size / volumes_per_batch

        network.train(
            self.model,
            self.training_images_and_labels,
            self.validation_images_and_labels,
            self.parameter.cost_threshold,
            self.parameters.epoch_count,
            batch_count,
            self.on_batch_event,
            self.on_epoch_event )


#---------------------------------------------------------------------------------------------------




#---------------------------------------------------------------------------------------------------

class SparsePatchClassificationExperiment( Experiment ) :


    def __init__(
            self,
            dataset,
            architecture,
            cost_function,
            optimiser,
            training_batch_parameters,
            validation_batch_parameters,
            experiment_parameters,
            seed = None ) :

        super().__init__(
            dataset,
            architecture,
            cost_function,
            optimiser,
            training_batch_parameters,
            validation_batch_parameters,
            experiment_parameters,
            seed )


    def format_labels( self, dense_patch_indices ) :

        distribution = labels.dense_patch_indices_to_dense_patch_distribution( dense_patch_indices )
        per_patch_distribution = labels.dense_patch_distribution_to_sparse_patch_distribution( distribution )
        return per_patch_distribution


    def on_batch_event( self, batch_index, training_output, training_costs ) :

        # TODO: record these, maybe graph them, do something more interesting
        print( "batch {} : {}".format( batch_index, training_costs[ -1 ] ) )


    def on_epoch_event( self, epoch_index, validation_output, validation_costs, training_costs ) :

        # TODO: record these, maybe graph them, do something more interesting
        print( "epoch {} : {}".format( epoch_index, validation_costs[ -1 ] ) )


#---------------------------------------------------------------------------------------------------





#---------------------------------------------------------------------------------------------------

class DensePatchClassificationExperiment( Experiment ) :


    def __init__(
            self,
            dataset,
            architecture,
            cost_function,
            optimiser,
            training_batch_parameters,
            validation_batch_parameters,
            experiment_parameters,
            seed = None ) :

        super().__init__(
            dataset,
            architecture,
            cost_function,
            optimiser,
            training_batch_parameters,
            validation_batch_parameters,
            experiment_parameters,
            seed )

        final_feature_map_size = validation_batch_parameters.stride
        first_feature_map_size = validation_batch_parameters.patch_shape[ 0 ]
        self.__margin = final_feature_map_size - first_feature_map_size


    def format_labels( self, dense_patch_indices ) :

        distribution = labels.dense_patch_indices_to_dense_patch_distribution( dense_patch_indices )
        shape = distribution.shape

        all_instances = slice( 0, shape[ 0 ], 1 )
        all_classes = slice( 0, shape[ -1 ], 1 )
        window = tuple( slice( self.__margin, d - self.__margin, 1 ) for d in shape[ 1:-1 ] )

        distribution_in_window = distribution[ ( all_instances, ) + window + ( all_classes, ) ]
        return labels_in_window


    def on_batch_event( self, batch_index, training_output, training_costs ) :

        # TODO: record these, maybe graph them, do something more interesting
        print( "batch {} : {}".format( batch_index, training_costs[ -1 ] ) )


    def on_epoch_event( self, epoch_index, validation_output, validation_costs, training_costs ) :

        # TODO: record these, maybe graph them, do something more interesting
        print( "epoch {} : {}".format( epoch_index, validation_costs[ -1 ] ) )


#---------------------------------------------------------------------------------------------------
