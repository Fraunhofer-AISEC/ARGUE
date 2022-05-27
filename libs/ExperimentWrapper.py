import random
import re

from matplotlib import pyplot as plt

import numpy as np
import pandas as pd
import tensorflow as tf

from typing import Union, List, Tuple, Dict, NoReturn
from pathlib import Path
from copy import deepcopy

from libs.ARGUE import ARGUE
from libs.DataHandler import DataLabels
from libs.Cluster import SplitAAE
from libs.Metrics import evaluate_roc, roc_to_pandas
from libs.architecture.target import MultiAutoencoder, Autoencoder, AdversarialAutoencoder, AdversarialClustering
from libs.constants import BASE_PATH

from sklearn.metrics import roc_curve, roc_auc_score


class ExperimentWrapper:
    def __init__(
            self, data_setup: List[DataLabels], p_contamination: float = 0.0,
            random_seed: int = None, is_override=False,
            save_prefix: str = '', out_path: Path = BASE_PATH, auto_subfolder: bool = True,
    ):
        """
        Wrapper class to have a common scheme for the experiments
        :param data_setup: data configuration for every experiment
        :param p_contamination: fraction of contamination, i.e. anomaly samples in the training data
        :param save_prefix: prefix for saved NN models
        :param random_seed: seed to fix the randomness
        :param is_override: override output if it already exists
        :param out_path: output base path for the models, usually the base path
        :param auto_subfolder: create a subfolder on the out_path with the random seed and the contamination level
        """

        # Save the parameter grid
        self.data_setup = data_setup  # This is mutable to allow target splitting
        self.p_contamination = p_contamination

        # Configuration
        self.is_override = is_override

        # Folder paths
        self.out_path = out_path
        if auto_subfolder:
            self.out_path /= f"{p_contamination}_{random_seed}"
        # If necessary, create the output folder
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self.save_prefix = f"{save_prefix}"
        if random_seed is not None:
            self.save_prefix += f"_{random_seed}"

        # Fix randomness
        self.random_seed = random_seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        # Alright, we can't make the NN deterministic on a GPU [1]. Probably makes more sense to keep the sample
        # selection deterministic, but repeat all NN-related aspects.
        # [1] https://keras.io/getting-started/faq/#how-can-i-obtain-reproducible-results-using-keras-during-development
        # tf.random.set_seed(random_seed)

    def train_mae(
            self,
            network_params: dict = None, compile_params: dict = None, fit_params: dict = None,
            use_given_splits: bool = True
    ) -> NoReturn:
        """
        Train the multi-autoencoder
        :param network_params: additional arguments passed to the network class
        :param compile_params: additional arguments passed to the network class
        :param fit_params: additional arguments passed to the network class
        :param use_given_splits: use the clusters given in the data, else: approximate them
        :return:
        """

        # Default to empty dictionaries
        if not network_params: network_params = {}
        if not compile_params: compile_params = {}
        if not fit_params: fit_params = {}

        for cur_data in deepcopy(self.data_setup):
            # Announce what we're doing
            this_prefix = self.parse_name(cur_data)
            print(f"Now training multi-autoencoder {this_prefix}")

            # Model paths
            aae_cluster_path = self.get_model_path(
                base_path=self.out_path, file_name=this_prefix,
                sub_folder="AAECluster"
            )
            mae_path = self.get_model_path(
                base_path=self.out_path, file_name=this_prefix,
                sub_folder="MAE" if use_given_splits else "MAE-0"
            )

            # Check if the model already exists
            if mae_path.exists() and not self.is_override:
                print("This MAE was already trained. Aborting.")
                continue

            # Load data
            this_data = cur_data.to_data(
                for_experts=use_given_splits, equal_size=True,
            )

            # Split the target if desired
            if not use_given_splits:
                this_split = SplitAAE(
                    target_ae=tf.keras.models.load_model(aae_cluster_path),
                    data_source=this_data,
                )
                # Modify the data source
                this_data = this_split.split_data(this_data)

            # Build the MAE
            this_model = MultiAutoencoder(n_experts=len(this_data.train_target), **network_params)
            this_model.compile(**compile_params)
            this_model.fit(
                x=[cur_val[0] for cur_val in this_data.train_target.values()],
                y=[cur_val[1] for cur_val in this_data.train_target.values()],
                validation_data=(
                    tuple(cur_val[0] for cur_val in this_data.val_target.values()),
                    tuple(cur_val[1] for cur_val in this_data.val_target.values()),
                ),
                **fit_params
            )
            this_model.save(mae_path)

    def train_argue(
            self, architecture_params: dict = None, compile_params: dict = None,
            fit_params: dict = None, use_given_splits: bool = True
    ) -> NoReturn:
        """
        Train the alarm network, e.g. the anomaly classifier
        :param architecture_params: additional arguments passed to the overall ARGUE architecture
        :param compile_params: additional arguments passed to the compile() function
        :param fit_params: additional arguments passed to the network class
        :param use_given_splits: use the clusters given in the data, else: approximate them
        :return:
        """

        if not architecture_params: architecture_params = {}
        if not compile_params: compile_params = {}
        if not fit_params: fit_params = {}

        for cur_data in deepcopy(self.data_setup):
            # Announce what we're doing
            component_prefix = self.parse_name(cur_data, is_supervised=False)
            # ARGUE can be unsupervised and semi-supervised
            argue_prefix = self.parse_name(cur_data, is_supervised=True)
            print(f"Now training ARGUE {argue_prefix}")

            # Model paths
            argue_path = self.get_model_path(
                base_path=self.out_path, file_name=argue_prefix,
                sub_folder="ARGUE" if use_given_splits else "ARGUE-0"
            )
            aae_cluster_path = self.get_model_path(
                base_path=self.out_path, file_name=component_prefix,
                sub_folder="AAECluster"
            )
            mae_path = self.get_model_path(
                base_path=self.out_path, file_name=component_prefix,
                sub_folder="MAE" if use_given_splits else "MAE-0"
            )

            # Check if the model already exists
            if (argue_path.exists() or argue_path.with_suffix(".tf.index").exists()) and not self.is_override:
                print("This ARGUE model was already trained. Aborting.")
                continue

            # Load data
            this_data = cur_data.to_data(
                for_experts=use_given_splits, equal_size=True,
            )

            # Split the target if desired
            if not use_given_splits:
                this_split = SplitAAE(
                    target_ae=tf.keras.models.load_model(aae_cluster_path),
                    data_source=this_data,
                )
                # Modify the data source
                this_data = this_split.split_data(this_data)

            # Load the MAE
            this_mae = tf.keras.models.load_model(mae_path)

            # Build ARGUE
            this_model = ARGUE(m_target=this_mae, **architecture_params)
            this_model.compile(**compile_params)

            # Use the right data for fitting
            train_y = [this_data.train_alarm[1], this_data.train_attention]
            val_xy = (this_data.val_alarm[0], [this_data.val_alarm[1], this_data.val_attention])
            this_model.fit(
                x=this_data.train_alarm[0], y=train_y, validation_data=val_xy, **fit_params
            )

            this_data = cur_data.to_data(
                for_experts=use_given_splits, equal_size=True,
            )

            # Predict
            pred_y = this_model.predict(this_data.val_alarm[0])

            # Calculate the AUC and AP
            this_auc = roc_auc_score(y_true=this_data.val_alarm[1], y_score=pred_y)

            print(f"\n ***\n AUC: {this_auc} \n ***\n")

            # Save
            this_model.save(argue_path)

    def evaluate_argue(
            self, data_split: str,
            architecture_params: dict = None, target_params: dict = None,
            evaluate_baseline: Dict[str, dict] = None, out_path: Path = None,
            evaluate_a4rgue: bool = True
    ) -> NoReturn:
        """
        Evaluate the performance of ARGUE
        :param data_split: data split, e.g. val or test
        :param architecture_params: ARGUE architecture settings
        :param target_params: target network's architecture settings
        :param evaluate_baseline: also evaluate the given baseline methods, expects a dict of {"baseline": {config}}
        :param out_path: special output path for the results
        :return:
        """

        if architecture_params is None: architecture_params = {}
        if target_params is None: target_params = {}
        if out_path is None: out_path = self.out_path

        for i_data, cur_data in enumerate(deepcopy(self.data_setup)):
            # Start with a new session
            plt.clf()
            tf.keras.backend.clear_session()

            # We'll output the metrics and the x,y coordinates for the ROC
            df_metric = pd.DataFrame(columns=["AUC", "AP"])
            df_roc = pd.DataFrame()

            # Announce what we're doing
            component_prefix = self.parse_name(cur_data, is_supervised=False)
            argue_prefix = self.parse_name(cur_data, is_supervised=True)
            print(f"Now evaluating {argue_prefix}")

            # Get the output path
            # Check if the respective model exists
            csv_path = self.get_model_path(
                base_path=out_path, file_name=argue_prefix,
                file_suffix=".csv"
            )

            # Evaluate baseline methods
            if evaluate_baseline:
                for baseline_name, baseline_config in evaluate_baseline.items():
                    print(f"Evaluating {baseline_name}")
                    tf.keras.backend.clear_session()
                    baseline_metric, baseline_roc = self.evaluate_baseline_on(
                        data_split=data_split, baseline=baseline_name, input_config=cur_data, **baseline_config
                    )
                    df_metric.loc[baseline_name, :] = baseline_metric
                    df_roc = pd.concat([df_roc, baseline_roc], axis=1, ignore_index=False)

            # Load the data
            this_data = cur_data.to_data(test_type=data_split)

            # Model paths
            # Sorry for having everything twice, the number of experiments explodes with each iteration of the paper
            argue_path = self.get_model_path(
                base_path=self.out_path, file_name=argue_prefix,
                sub_folder="ARGUE"
            )
            argue0_path = self.get_model_path(
                base_path=self.out_path, file_name=argue_prefix,
                sub_folder="ARGUE-0"
            )

            # Open ARGUE model
            this_argue = tf.keras.models.load_model(argue_path)
            df_gate = self._evaluate_gating_decision(this_argue, this_data.test_alarm)
            pred_y = this_argue.predict(this_data.test_alarm[0])
            pred_y = pred_y if not isinstance(pred_y, list) else pred_y[0]
            # Calculate metrics
            df_metric.loc["ARGUE", :] = evaluate_roc(pred_scores=pred_y, test_alarm=this_data.test_alarm)

            # Same for A4RGUE
            if evaluate_a4rgue:
                this_argue0 = tf.keras.models.load_model(argue0_path)
                pred_y0 = this_argue0.predict(this_data.test_alarm[0])
                pred_y0 = pred_y0 if not isinstance(pred_y0, list) else pred_y0[0]
                df_metric.loc["A4RGUE", :] = evaluate_roc(pred_scores=pred_y0, test_alarm=this_data.test_alarm)

            # Plot ROC
            fpr_argue, tpr_argue, thresholds_argue = roc_curve(
                y_true=this_data.test_alarm[1], y_score=pred_y
            )
            plt.plot(fpr_argue, tpr_argue, label="ARGUE")
            df_roc = pd.concat([
                df_roc,
                roc_to_pandas(fpr=fpr_argue, tpr=tpr_argue, suffix="ARGUE")
            ], axis=1, ignore_index=False)

            # Save the resulting DFs
            df_metric.to_csv(csv_path.with_suffix(".metric.csv"))
            df_roc.to_csv(csv_path.with_suffix(".roc.csv"))
            df_gate.to_csv(csv_path.with_suffix(".gate.csv"))

            # Plot the ROC
            plt.plot([0, 1], [0, 1], label="Random Classifier")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.legend()
            # plt.show()
            # For some reason, mpl tries to use the LaTeX processor when adding "GANomaly" - this might fail
            try:
                plt.savefig(csv_path.with_suffix(".roc.png"))
            except RuntimeError:
                # LaTeX used, but not available - this should not interfere with the rest of the evaluation
                pass

    def _evaluate_gating_decision(self, m_argue, alarm_data):

        # Filter by normal and anomalous data
        x_norm = alarm_data[0][(alarm_data[1] == 0).flatten(), :]
        x_anom = alarm_data[0][(alarm_data[1] == 1).flatten(), :]

        # Get the activations on them
        act_norm = m_argue.m_mae.m_enc_act(x_norm, training=False)
        act_anom = m_argue.m_mae.m_enc_act(x_anom, training=False)

        # Get the gating decision
        gate_norm = m_argue.m_gating.predict(act_norm)
        gate_anom = m_argue.m_gating.predict(act_anom)

        # Take the mean
        mean_norm = np.mean(gate_norm, axis=0)
        mean_anom = np.mean(gate_anom, axis=0)

        # Accumulate to one dataframe
        gating_out = pd.DataFrame(
            {"normal": mean_norm, "anomalous": mean_anom}
        )

        return gating_out.transpose()

    # -- Baselines --
    @staticmethod
    def _get_baseline_info(baseline: str) -> Tuple[str, bool]:
        """
        Get the right file suffix for the respective baseline
        :param baseline: baseline name
        :return: file suffix and if the method is supervised
        """
        if baseline == "DAGMM":
            file_suffix = ""
            is_supervised = False
        elif baseline in ["A3", "DevNet", "DeepSAD"]:
            file_suffix = ".tf"
            # It's just about the naming - they will get the train anomalies
            is_supervised = False
        elif baseline in ["AE", "AAE", "AAECluster", "DeepSVDD-AE", "DeepSVDD", "GANomaly", ""
                                                                                            "", "fAnoGAN", "MEx_CVAEC"]:
            file_suffix = ".tf"
            is_supervised = False
        elif baseline == "IF":
            file_suffix = ".joblib"
            is_supervised = False
        else:
            raise NotImplementedError(f"{baseline} is not a known baseline method")

        return file_suffix, is_supervised

    def train_baseline(
            self, baseline: str,
            compile_params: dict = None, fit_params: dict = None, **model_params
    ) -> NoReturn:
        """
        Train some baseline methods
        :param baseline: which baseline method to evaluate
        :param compile_params: arguments for the (optional) compile function
        :param fit_params: arguments for the (optional) fit function
        :param model_params: extra arguments for the baseline method constructor
        :return:
        """

        # Check if baseline method exists
        file_suffix, is_supervised = self._get_baseline_info(baseline)

        # Default to empty dictionaries
        if compile_params is None: compile_params = {}
        if fit_params is None: fit_params = {}

        for cur_data in self.data_setup:
            # Unsupervised methods don't know the training anomalies
            this_prefix = self.parse_name(cur_data, is_supervised=is_supervised)
            print(f"Now training baseline method '{baseline}' for {this_prefix}")

            # Check if the respective model exists
            out_path = self.get_model_path(
                base_path=self.out_path, file_name=this_prefix,
                file_suffix=file_suffix, sub_folder=baseline
            )
            if not self.is_override and (
                    out_path.exists()
                    or out_path.with_suffix(".overall.h5").exists()
                    or out_path.with_suffix(".tf.index").exists()
            ):
                print("This baseline method was already trained. Use is_override=True to override it.")
                continue

            # Create the parent folder if not existing
            if not out_path.parent.exists():
                out_path.parent.mkdir(parents=True, exist_ok=False)

            # Open the data
            this_data = cur_data.to_data(for_experts=False)

            # Fit the baseline method
            try:
                if baseline == 'DAGMM':
                    from baselines.dagmm_v2 import DAGMM

                    # Revert to TF1 for compatibility
                    tf.compat.v1.disable_v2_behavior()

                    baseline_model = DAGMM(random_seed=self.random_seed, **model_params)
                    baseline_model.fit(
                        this_data.train_target[0].reshape(this_data.train_target[0].shape[0], -1), **fit_params
                    )

                    baseline_model.save(out_path)

                    # We need to restore the TF2 behaviour afterwards
                    tf.compat.v1.reset_default_graph()
                    tf.compat.v1.enable_v2_behavior()

                elif baseline == "fAnoGAN":
                    from baselines.anogan.AnoGAN import fAnoGAN

                    # Load the AE
                    this_ae = tf.keras.models.load_model(out_path.parent.parent / "AAE" / out_path.name)
                    # Create the baseline
                    baseline_model = fAnoGAN(base_ae=this_ae, **model_params)

                    # Call once to initialise the weights
                    baseline_model(this_data.val_alarm[0])
                    baseline_model.compile(**compile_params)
                    baseline_model.fit(
                        x=this_data.train_target[0],
                        **fit_params
                    )

                    # Save the baseline
                    baseline_model.save_weights(out_path)

                elif baseline == "MEx_CVAEC":
                    from baselines.mex_cvaec.MEx_CVAEC import MEx_CVAEC

                    # Create the baseline
                    baseline_model = MEx_CVAEC(**model_params)

                    # Call once to initialise the weights
                    baseline_model(this_data.val_alarm[0])
                    baseline_model.compile(**compile_params)
                    baseline_model.fit(
                        x=this_data.train_target[0],
                        y=this_data.train_target[1],
                        **fit_params
                    )

                    # Save the baseline
                    baseline_model.save_weights(out_path)

                elif baseline == "IF":
                    from sklearn.ensemble import IsolationForest
                    from joblib import dump

                    baseline_model = IsolationForest(random_state=self.random_seed, n_jobs=3, **model_params)
                    baseline_model.fit(
                        this_data.train_target[0].reshape(this_data.train_target[0].shape[0], -1), **fit_params
                    )

                    dump(baseline_model, out_path)

                elif baseline in ["AE", "AAE", "AAECluster", "DeepSVDD-AE"]:
                    # Use the right architecture
                    if baseline == "AE":
                        this_net = Autoencoder(**model_params)
                    elif baseline == "AAE":
                        this_net = AdversarialAutoencoder(**model_params)
                    elif baseline == "AAECluster":
                        this_net = AdversarialClustering(**model_params)
                    elif baseline == "DeepSVDD-AE":
                        from baselines.deep_svdd.DeepSVDD import DeepSVDDAE
                        this_net = DeepSVDDAE(**model_params)
                    else:
                        raise NotImplementedError("Unknown AE architecture")

                    if "optimizer" in compile_params:
                        raise NotImplementedError("Please omit the 'optimizer' keyword. So far only Adam is supported. Specify the LR directly.")
                    this_net.compile(**compile_params)

                    this_net.fit(
                        x=this_data.train_target[0], y=this_data.train_target[1],
                        validation_data=this_data.val_target,
                        **fit_params
                    )

                    this_net.save(out_path)

                elif baseline in ["DeepSVDD"]:
                    from baselines.deep_svdd.DeepSVDD import DeepSVDD

                    # Load the DeepSVDD-AE
                    this_ae = tf.keras.models.load_model(out_path.parent.parent / "DeepSVDD-AE" / out_path.name)

                    # Create the baseline
                    if baseline == "DeepSVDD":
                        baseline_model = DeepSVDD(pretrained_ae=this_ae, **model_params)
                    else:
                        raise NotImplementedError

                    # Call once to initialise the weights
                    baseline_model(this_data.val_alarm[0])
                    baseline_model.calculate_c(this_data.train_target[0])
                    baseline_model.compile(**compile_params)
                    baseline_model.fit(
                        x=this_data.train_target[0],
                        **fit_params
                    )

                    # Save the baseline
                    baseline_model.save_weights(out_path)

                elif baseline == "DeepSAD":
                    from baselines.deep_svdd.DeepSVDD import DeepSAD

                    # Load the DeepSVDD-AE
                    ae_prefix = self.parse_name(cur_data, is_supervised=False)
                    # Check if the respective model exists
                    ae_path = self.get_model_path(
                        base_path=self.out_path, file_name=ae_prefix,
                        file_suffix=file_suffix, sub_folder="DeepSVDD-AE"
                    )
                    this_ae = tf.keras.models.load_model(ae_path)

                    # Create the baseline
                    baseline_model = DeepSAD(pretrained_ae=this_ae, **model_params)

                    # Call once to initialise the weights
                    baseline_model(this_data.val_alarm[0])
                    baseline_model.calculate_c(this_data.train_target[0])
                    baseline_model.compile(**compile_params)
                    # DeepSAD expects the labels to be either +1 (normal) or -1 (anomalous)
                    y_sad = (-2 * this_data.train_alarm[1].astype(np.int8) + 1).reshape((-1,)).astype(np.float32)
                    baseline_model.fit(
                        x=this_data.train_alarm[0],
                        y=y_sad,
                        **fit_params
                    )

                    # Save the baseline
                    baseline_model.save_weights(out_path)

                elif baseline == "A3":
                    from baselines.a3.A3v2 import A3

                    # Load the target autoencoder
                    this_ae = tf.keras.models.load_model(out_path.parent.parent / "AE" / out_path.name)

                    # Create A3
                    baseline_model = A3(m_target=this_ae, **model_params)
                    baseline_model.compile(**compile_params)

                    # Fit and save
                    baseline_model.fit(
                        x=this_data.train_alarm[0], y=this_data.train_alarm[1],
                        validation_data=this_data.val_alarm, **fit_params
                    )

                    baseline_model.save_weights(out_path)

                elif baseline == "GANomaly":
                    from baselines.tf2_ganomaly.model import GANomaly, opt, batch_resize

                    # Check if the input data contains images as we need to scale them
                    if len(this_data.data_shape) > 1:
                        x_train = batch_resize(this_data.train_target[0], (32, 32))[..., None]
                        x_val = batch_resize(this_data.val_alarm[0], (32, 32))[..., None]
                        opt.isize = 32
                    else:
                        x_train = this_data.train_target[0]
                        x_val = this_data.val_alarm[0]
                        opt.isize = x_train.shape[-1]

                    # Use their option object
                    opt.encdims = model_params["enc_dims"]

                    # Convert data
                    # Although GANomaly is unsupervised, it needs training labels: just use zeros
                    train_dataset = tf.data.Dataset.from_tensor_slices(
                        (x_train, np.zeros((this_data.train_target[0].shape[0], )))
                    )
                    val_dataset = tf.data.Dataset.from_tensor_slices(
                        (x_val, this_data.val_alarm[1])
                    )
                    train_dataset = train_dataset.shuffle(opt.shuffle_buffer_size).batch(opt.batch_size, drop_remainder=True)
                    val_dataset = val_dataset.batch(opt.batch_size, drop_remainder=False)

                    # Construct and train
                    baseline_model = GANomaly(opt, train_dataset=train_dataset)
                    baseline_model.fit(opt.niter)

                    # Save
                    baseline_model.save(out_path)

                elif baseline in ["DevNet", "DevNet+AAE"]:
                    from baselines.devnet_v2.devnet_kdd19 import fit_devnet

                    # Revert to TF1 for compatibility
                    tf.compat.v1.disable_v2_behavior()
                    try:
                        baseline_model = fit_devnet(
                            random_state=self.random_seed,
                            x=this_data.train_alarm[0].reshape(this_data.train_alarm[0].shape[0], -1),
                            y=this_data.train_alarm[1].reshape(this_data.train_alarm[1].shape[0], -1),
                        )
                        baseline_model.save_weights(str(out_path))
                    except ValueError:
                        print("Error fitting DevNet. Are there any known anomalies available?")

                    # We need to restore the TF2 behaviour afterwards
                    tf.compat.v1.reset_default_graph()
                    tf.compat.v1.enable_v2_behavior()

                elif baseline == "REPEN":
                    from baselines.repen.REPEN import REPEN

                    baseline_model = REPEN(**model_params)
                    # Call once to build the model
                    baseline_model(this_data.val_alarm[0].reshape(this_data.val_alarm[0].shape[0], -1))
                    baseline_model.compile(**compile_params)

                    # REPEN is an outlier detection method - it expects to find some anomalies in the training data
                    # If no pollution was applied, give REPEN the advantage of working on the test data
                    repen_data = this_data.train_alarm if self.p_contamination else this_data.test_alarm

                    # Fit and save
                    baseline_model.fit(
                        x=repen_data[0].reshape(repen_data[0].shape[0], -1),
                        y=repen_data[1].reshape(repen_data[1].shape[0], -1),
                        **fit_params
                    )

                    baseline_model.save_weights(out_path)

            except Exception as e:
                # DAGMM sometimes has problems on IDS - and DevNet only works in semi-supervised environments
                print(f"Could not fit {baseline}: {e}. Aborting.")
                return

    def evaluate_baseline_on(
            self, data_split: str, baseline: str, input_config, **model_params
    ) -> Tuple[list, pd.DataFrame]:
        """
        Evaluate a baseline method on a given data config
        :param data_split: data split, e.g. val or test
        :param baseline: which baseline method to evaluate
        :param input_config: configuration the baseline is evaluated on (takes test data)
        :param model_params: extra arguments for the baseline method constructor
        :return: DataFrame containing the metrics & DataFrame containing the ROC x,y data
        """

        # Check if baseline method exists
        file_suffix, is_supervised = self._get_baseline_info(baseline)

        this_prefix = self.parse_name(input_config, is_supervised=is_supervised)

        # Handle the file origins
        in_path = self.get_model_path(
            base_path=self.out_path, file_name=this_prefix,
            file_suffix=file_suffix, sub_folder=baseline
        )

        # Open the baseline and predict
        this_data = input_config.to_data(test_type=data_split, for_experts=False)
        pred_y = None
        try:
            if baseline == 'DAGMM':
                from baselines.dagmm_v2 import DAGMM

                baseline_model = DAGMM(random_seed=self.random_seed, **model_params)
                baseline_model.restore(in_path)

                pred_y = baseline_model.predict(
                    this_data.test_alarm[0].reshape((this_data.test_alarm[0].shape[0], -1))
                )

            elif baseline == "fAnoGAN":
                from baselines.anogan.AnoGAN import fAnoGAN

                # Load the AE
                this_ae = tf.keras.models.load_model(in_path.parent.parent / "AAE" / in_path.name)
                # Create the baseline
                baseline_model = fAnoGAN(base_ae=this_ae, **model_params)
                # Load the weights
                baseline_model(this_data.val_alarm[0])
                baseline_model.load_weights(in_path)

                # Debug the generator
                # gen_out = baseline_model.m_gen(tf.random.normal((5, baseline_model.gen_dim))).numpy()

                # Get the variables for the anomaly score
                x_gen, f_in, f_gen = baseline_model(this_data.test_alarm[0])
                pred_y = baseline_model.anomaly_score(
                    x_in=this_data.test_alarm[0], x_gen=x_gen,
                    f_in=f_in, f_gen=f_gen
                )

            elif baseline == "MEx_CVAEC":
                from baselines.mex_cvaec.MEx_CVAEC import MEx_CVAEC

                # Create the baseline
                baseline_model = MEx_CVAEC(**model_params)
                # Load the weights
                baseline_model(this_data.val_alarm[0])
                baseline_model.load_weights(in_path)

                # Get the variables for the anomaly score
                pred_y = baseline_model.get_anomaly_score(this_data.test_alarm[0])

            elif baseline == "IF":
                if model_params:
                    raise AttributeError("AE does not accept parameters while loading. Change while training.")

                from joblib import load
                baseline_model = load(in_path)

                pred_y = baseline_model.decision_function(
                    this_data.test_alarm[0].reshape((this_data.test_alarm[0].shape[0], -1))
                )
                # We need to invert the results as "The lower, the more abnormal."
                # See also https://github.com/scikit-learn/scikit-learn/blob/master/benchmarks/bench_isolation_forest.py
                pred_y *= -1

            elif baseline in ["AE", "AAE", "AAECluster"]:

                baseline_model = tf.keras.models.load_model(in_path)

                pred_y = baseline_model.m_dec.predict(
                    baseline_model.m_enc(this_data.test_alarm[0])
                )

                # We'll return the MSE as score
                pred_y = np.square(pred_y - this_data.test_alarm[0])
                # We might have 2D inputs: collapse to one dimension
                pred_y = np.reshape(pred_y, (pred_y.shape[0], -1))
                pred_y = np.mean(pred_y, axis=1)

            elif baseline in ["DeepSAD", "DeepSVDD"]:
                from baselines.deep_svdd.DeepSVDD import DeepSVDD, DeepSAD

                # Load the DeepSVDD-AE
                ae_prefix = self.parse_name(input_config, is_supervised=False)
                # Check if the respective model exists
                ae_path = self.get_model_path(
                    base_path=self.out_path, file_name=ae_prefix,
                    file_suffix=file_suffix, sub_folder="DeepSVDD-AE"
                )
                this_ae = tf.keras.models.load_model(ae_path)

                # Create the baseline
                if baseline == "DeepSAD":
                    baseline_model = DeepSAD(pretrained_ae=this_ae, **model_params)
                elif baseline == "DeepSVDD":
                    baseline_model = DeepSVDD(pretrained_ae=this_ae, **model_params)
                else:
                    raise NotImplementedError

                # Call once to initialise the weights
                baseline_model(this_data.val_alarm[0])
                baseline_model.load_weights(in_path)
                # c isn't saved, thus we need to recalculate is based on the training data
                baseline_model.calculate_c(this_data.train_target[0])

                # Predict
                pred_y = baseline_model.score(this_data.test_alarm[0])

            elif baseline == "A3":
                from baselines.a3.A3v2 import A3

                # Load the target autoencoder
                this_ae = tf.keras.models.load_model(in_path.parent.parent / "AE" / in_path.name)

                # Load A3
                baseline_model = A3(m_target=this_ae, **model_params)
                # baseline_model(this_data.val_alarm[0])
                baseline_model.load_weights(in_path)

                pred_y = baseline_model.predict(x=this_data.test_alarm[0])

            elif baseline == "GANomaly":
                from baselines.tf2_ganomaly.model import GANomaly, opt, batch_resize

                # Check if the input data contains images as we need to scale them
                if len(this_data.data_shape) > 1:
                    x_train = batch_resize(this_data.train_alarm[0], (32, 32))[..., None]
                    x_test = batch_resize(this_data.test_alarm[0], (32, 32))[..., None]
                    opt.isize = 32
                else:
                    x_train = this_data.train_alarm[0]
                    x_test = this_data.test_alarm[0]
                    opt.isize = x_train.shape[-1]

                # Use their option object
                opt.encdims = model_params["enc_dims"]

                # Convert data
                train_dataset = tf.data.Dataset.from_tensor_slices(
                    (x_train, this_data.train_alarm[1])
                )
                test_dataset = tf.data.Dataset.from_tensor_slices(
                    (x_test, this_data.test_alarm[1])
                )
                train_dataset = train_dataset.shuffle(opt.shuffle_buffer_size).batch(opt.batch_size, drop_remainder=True)
                test_dataset = test_dataset.batch(opt.batch_size, drop_remainder=False)

                # Construct and predict
                baseline_model = GANomaly(opt, train_dataset=train_dataset, test_dataset=test_dataset)
                baseline_model.load(in_path)

                pred_y = baseline_model._evaluate(test_dataset)[0]

            elif baseline in ["DevNet", "DevNet+AAE"]:
                from baselines.devnet_v2.devnet_kdd19 import predict_devnet

                pred_y = predict_devnet(
                    model_name=str(in_path),
                    x=this_data.test_alarm[0].reshape(this_data.test_alarm[0].shape[0], -1)
                )

            elif baseline == "REPEN":
                from baselines.repen.REPEN import REPEN

                # Load REPEN
                baseline_model = REPEN(**model_params)
                # Call once to build the model
                baseline_model(this_data.val_alarm[0].reshape(this_data.val_alarm[0].shape[0], -1))
                baseline_model.load_weights(in_path)

                pred_y = baseline_model.predict(
                    x=this_data.test_alarm[0].reshape(this_data.test_alarm[0].shape[0], -1)
                )
            else:
                raise NotImplementedError("Unknown baseline method")

        except FileNotFoundError:
            print(f"No model for {baseline} found. Aborting.")
            return [None, None], pd.DataFrame()

        # Plot ROC
        fpr, tpr, thresholds = roc_curve(
            y_true=this_data.test_alarm[1], y_score=pred_y
        )
        plt.plot(fpr, tpr, label=baseline)

        # Generate the output DF
        df_metric = evaluate_roc(pred_scores=pred_y, test_alarm=this_data.test_alarm)
        df_roc = roc_to_pandas(fpr=fpr, tpr=tpr, suffix=baseline)

        return df_metric, df_roc

    def do_everything(
            self, dim_target: Union[List[int], None], dim_alarm: List[int],
            learning_rate: float, batch_size: int, n_epochs: int,
            out_path: Path, evaluation_split: str = "val",
            dagmm_conf: dict = None, train_dagmm: bool = True,
            train_devnet: bool = True, train_repen: bool = True,
            train_a4rgue: bool = True, a4rgue_clusters: int = 5
    ) -> NoReturn:
        """
        Train & evaluate ARGUE and all relevant baseline methods
        :param dim_target: dimensions of the autoencoder's encoder (decoder is symmetric to this)
        :param dim_alarm: dimensions of the alarm network
        :param learning_rate: training learning rate for Adam
        :param batch_size: training batch size
        :param n_epochs: training epochs
        :param out_path: output path for the evaluation
        :param evaluation_split: data split to evaluate the methods on
        :param dagmm_conf: special configuration for DAGMM
        :param train_dagmm: train DAGMM
        :param train_devnet: train DevNet
        :param train_repen: train REPEN
        :param train_a4rgue: train A4RGUE
        :param a4rgue_clusters: number of clusters for A4RGUE
        :return:
        """

        if dagmm_conf is None:
            # This is their setting for KDD - we better stick to their architecture as this 1-dimensional layer seems
            # to make a difference
            dagmm_conf = {
                "comp_hiddens": [12, 4, 1], "comp_activation": tf.nn.tanh,
                "est_hiddens": [10, 2], "est_dropout_ratio": 0.5, "est_activation": tf.nn.tanh,
                "learning_rate": learning_rate, "epoch_size": n_epochs, "minibatch_size": batch_size
            }

        # We need the AAE to cluster the data
        self.train_baseline(
            baseline="AAE",
            layer_dims=dim_target,
            compile_params={"learning_rate": learning_rate},
            fit_params={"epochs": n_epochs, "batch_size": batch_size, "verbose": 2},
        )
        if train_a4rgue:
            self.train_baseline(
                baseline="AAECluster",
                layer_dims=dim_target, code_dim_override=2, n_clusters=a4rgue_clusters,
                compile_params={"learning_rate": learning_rate},
                fit_params={"epochs": n_epochs, "batch_size": batch_size, "verbose": 2},
            )


        # Split the targets
        self.train_mae(
            network_params={"layer_dims": dim_target},
            compile_params={"learning_rate": learning_rate},
            fit_params={"epochs": n_epochs, "batch_size": batch_size, "verbose": 2},
            use_given_splits=True
        )
        if train_a4rgue:
            try:
                self.train_mae(
                    network_params={"layer_dims": dim_target},
                    compile_params={"learning_rate": learning_rate},
                    fit_params={"epochs": n_epochs, "batch_size": batch_size, "verbose": 2},
                    use_given_splits=False
                )
            except ValueError:
                # Sometimes the AAE does not find enough clusters
                train_a4rgue = False

        # Train the alarm network
        self.train_argue(
            architecture_params={"layer_dims": dim_alarm},
            compile_params={"learning_rate": learning_rate},
            fit_params={"epochs": n_epochs, "batch_size": batch_size, "verbose": 2},
            use_given_splits=True
        )
        if train_a4rgue:
            self.train_argue(
                architecture_params={"layer_dims": dim_alarm},
                compile_params={"learning_rate": learning_rate},
                fit_params={"epochs": n_epochs, "batch_size": batch_size, "verbose": 2},
                use_given_splits=False
            )

        # Train the baselines
        self.train_baseline(
            baseline="IF"
        )
        self.train_baseline(
            baseline="AE",
            layer_dims=dim_target,
            compile_params={"learning_rate": learning_rate},
            fit_params={"epochs": n_epochs, "batch_size": batch_size, "verbose": 2},
        )
        self.train_baseline(
            baseline="GANomaly",
            enc_dims=dim_target
        )
        if train_dagmm:
            try:
                self.train_baseline(
                    baseline="DAGMM", **dagmm_conf
                )
            except:
                train_dagmm = False
        self.train_baseline(
            baseline="A3",
            layer_dims=dim_alarm,
            compile_params={"learning_rate": learning_rate},
            fit_params={"epochs": n_epochs, "batch_size": batch_size, "verbose": 2}
        )
        self.train_baseline(
            baseline="DeepSVDD-AE",
            layer_dims=dim_target,
            compile_params={"learning_rate": learning_rate},
            fit_params={"epochs": n_epochs, "batch_size": batch_size, "verbose": 2},
        )
        self.train_baseline(
            baseline="DeepSVDD",
            compile_params={"learning_rate": learning_rate},
            fit_params={"epochs": n_epochs, "batch_size": batch_size, "verbose": 2},
        )
        self.train_baseline(
            baseline="DeepSAD",
            compile_params={"learning_rate": learning_rate},
            fit_params={"epochs": n_epochs, "batch_size": batch_size, "verbose": 2},
        )
        self.train_baseline(
            baseline="MEx_CVAEC",
            layer_dims=dim_target,
            compile_params={"learning_rate": learning_rate},
            fit_params={"epochs": n_epochs, "batch_size": batch_size, "verbose": 2},
        )
        self.train_baseline(
            baseline="fAnoGAN",
            compile_params={"learning_rate": learning_rate},
            fit_params={"epochs": n_epochs, "batch_size": batch_size, "verbose": 2},
        )
        # REPEN is very slow - stick to the 30 epochs as done in their code
        if train_repen:
            self.train_baseline(
                baseline="REPEN", random_seed=self.random_seed,
                compile_params={"learning_rate": learning_rate},
                fit_params={"epochs": min(30, n_epochs), "batch_size": batch_size, "verbose": 2},
            )
        if train_devnet:
            self.train_baseline(
                baseline="DevNet"
            )

        # Get the results
        baseline_methods = {
            "IF": {},
            "AE": {},
            "DeepSVDD": {},
            "DeepSAD": {},
            "MEx_CVAEC": {"layer_dims": dim_target},
            "fAnoGAN": {},
            "GANomaly": {"enc_dims": dim_target},
            "A3": {"layer_dims": dim_alarm},
        }
        if train_dagmm:
            baseline_methods["DAGMM"] = dagmm_conf
        if train_devnet:
            baseline_methods["DevNet"] = {}
        if train_repen:
            baseline_methods["REPEN"] = {"random_seed": self.random_seed}
        self.evaluate_argue(
            evaluation_split, out_path=out_path, architecture_params={"layer_dims": dim_alarm},
            evaluate_baseline=baseline_methods, evaluate_a4rgue=train_a4rgue
        )

    # -- Helpers --
    @staticmethod
    def parse_name(
            in_conf: dict, prefix: str = None, suffix: str = None,
            is_supervised: bool = False
    ) -> str:
        """
        Convert configuration to a nicer file name
        :param in_conf: dictionary
        :param prefix: a string that will be prepended to the name
        :param suffix: a string that will be appended to the name
        :param is_supervised: is the method supervised? if so use different keywords for the file name
        :return: string describing the dictionary
        """
        # Semi-supervised methods are trained on known anomalies
        keep_keywords = ("y_normal", "y_anomalous", "n_train_anomalies", "p_pollution") if is_supervised \
            else ("y_normal", "p_pollution")

        # Convert to member dict if it's not a dict
        out_dict = in_conf if isinstance(in_conf, dict) else vars(in_conf).copy()

        # Remove all keywords but the desired ones
        out_dict = {
            cur_key: cur_val for cur_key, cur_val in out_dict.items() if cur_key in keep_keywords
        }

        # Parse as string
        out_str = str(out_dict)

        # Remove full stops and others as otherwise the path may be invalid
        out_str = re.sub(r"[{}\\'.<>\[\]()\s]", "", out_str)

        # Alter the string
        if prefix: out_str = prefix + "_" + out_str
        if suffix: out_str = out_str + "_" + suffix

        return out_str

    @staticmethod
    def dict_to_str(in_dict: dict) -> str:
        """
        Parse the values of a dictionary as string
        :param in_dict: dictionary
        :return: dictionary with the same keys but the values as string
        """
        out_dict = {cur_key: str(cur_val) for cur_key, cur_val in in_dict.items()}

        return out_dict

    def get_model_path(
            self, base_path: Path,
            file_name: str = None, file_suffix: str = ".tf",
            sub_folder: str = "", sub_sub_folder: str = "",
    ) -> Path:
        """
        Get the path to save the NN models
        :param base_path: path to the project
        :param file_name: name of the model file (prefix is prepended)
        :param file_suffix: suffix of the file
        :param sub_folder: folder below model folder, e.g. for alarm/target
        :param sub_sub_folder: folder below subfolder, e.g. architecture details
        :return:
        """
        out_path = base_path

        if sub_folder:
            out_path /= sub_folder

        if sub_sub_folder:
            out_path /= sub_sub_folder

        # Create the path if it does not exist
        if not out_path.exists():
            out_path.mkdir(parents=True, exist_ok=False)

        if file_name:
            out_path /= f"{self.save_prefix}_{file_name}"
            out_path = out_path.with_suffix(file_suffix)

        return out_path

