"""Tuner module
"""
import os
 
import tensorflow as tf
from keras.utils.vis_utils import plot_model
import tensorflow_transform as tft 
import kerastuner as kt

def get_hyperparameters() -> kt.HyperParameters:
    """Returns hyperparameters for building Keras model."""
    hparams = kt.HyperParameters()
    # Defines search space.
#     hparams.Choice('learning_rate', [1e-1, 1e-2, 1e-3], default=1e-2)
    hparams.Float("learning_rate",
            min_value=1e-4, max_value=10,
            step=1)
    return hparams

def tuner_fn(fn_args):
    """Build the tuner using the KerasTuner API.
    Args:
    fn_args: Holds args used to tune models as name/value pairs.

    Returns:
    A namedtuple contains the following:
      - tuner: A BaseTuner that will be used for tuning.
      - fit_kwargs: Args to pass to tuner's run_trial function for fitting the
                    model , e.g., the training and validation dataset. Required
                    args depend on the above tuner's implementation.
    """
    # Memuat training dan validation dataset yang telah di-preprocessing
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)
    train_set = input_fn(fn_args.train_files[0], tf_transform_output, num_epochs=5)
    val_set = input_fn(fn_args.eval_files[0], tf_transform_output, num_epochs=5)

    # Mendefinisikan strategi hyperparameter tuning
    tuner = kt.Hyperband(model_builder(get_hyperparameters(), fn_args),
            objective='val_accuracy', max_epochs=10,
            factor=3,
            directory=fn_args.working_dir,
            project_name='kt_hyperband')
#     tuner = kt.RandomSearch(
#         model_builder.model_builder,
#         max_trials=6,
#         hyperparameters=model_builder.get_hyperparameters(),
#         allow_new_entries=False,
#         objective=keras_tuner.Objective('val_accuracy', 'max'),
#         directory=fn_args.working_dir,
#         project_name='imdb_sentiment_classification')

    return TunerFnResult(
    tuner=tuner,
    fit_kwargs={ "callbacks":[stop_early], 
            'x': train_set, 'validation_data': val_set,
            'steps_per_epoch': fn_args.train_steps, 'validation_steps': fn_args.eval_steps
    })