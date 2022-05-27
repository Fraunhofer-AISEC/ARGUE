import tensorflow as tf
from argparse import ArgumentParser

from libs.DataHandler import CovType
from libs.ExperimentWrapper import ExperimentWrapper

from libs.constants import add_standard_arguments, ALARM_SMALL, ALARM_BIG

# Reduce the hunger of TF when we're training on a GPU
try:
    tf.config.experimental.set_memory_growth(tf.config.list_physical_devices("GPU")[0], True)
except IndexError:
    tf.config.run_functions_eagerly(True)
    pass  # No GPUs available

# Configuration
this_parse = ArgumentParser(description="Train ARGUE on CovType")
add_standard_arguments(this_parse)
this_args = this_parse.parse_args()

experiment_config = [
    CovType(
        random_state=this_args.random_seed, y_normal=[1, 2, 3, 4], y_anomalous=[5, 6, 7],
        n_train_anomalies=this_args.n_train_anomalies, p_pollution=this_args.p_contamination
    )
]

DIM_TARGET = [90, 75, 60, 45, 25, 15]
DIM_ALARM = ALARM_BIG
BATCH_SIZE = 4096
A4RGUE_CLUSTERS = 2

if __name__ == '__main__':

    this_experiment = ExperimentWrapper(
        save_prefix="CovType", data_setup=experiment_config,
        random_seed=this_args.random_seed, out_path=this_args.model_path,
        p_contamination=this_args.p_contamination
    )

    this_experiment.do_everything(
        dim_target=DIM_TARGET, dim_alarm=DIM_ALARM,
        learning_rate=this_args.learning_rate, batch_size=BATCH_SIZE, n_epochs=this_args.n_epochs,
        out_path=this_args.result_path, evaluation_split=this_args.data_split,
        train_devnet=this_args.n_train_anomalies > 0
    )
