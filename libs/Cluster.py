import joblib

import tensorflow as tf
import numpy as np

from libs.DataTypes import ExperimentData
from libs.DataHandler import DataLabels

from typing import Dict, List, Tuple
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt

from sklearn.cluster import MiniBatchKMeans
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin


class QuadrantCluster(TransformerMixin, ClusterMixin, BaseEstimator):
    def __init__(self):
        super(QuadrantCluster, self).__init__()


class SplitQuadrant:
    def __init__(
            self, target_ae: tf.keras.Model, data_source: ExperimentData,
    ):
        """
        Split the normal class into multiple clusters by splitting them in quadrants on the AAE code layer
        :param target_ae: an autoencoder trained on the data to split
        :param data_source: data describing the experiment
        """
        # Save the configuration
        self.data_source = deepcopy(data_source)

        # We only split if one label is available
        assert not isinstance(self.data_source.train_target, dict) or len(self.data_source.train_target) == 1, \
            "Label splitting is only applicable if the number of training labels is 1"
        # If this was a dictionary: save the label
        self.label_normal = "normal" if not isinstance(self.data_source.train_target, dict) \
            else list(self.data_source.train_target.keys())[0]

        # Save the code layer of the AE
        self.m_code = self._extract_code(target_ae)

    @staticmethod
    def _extract_code(target_ae: tf.keras.Model):
        return target_ae.m_enc

    def _get_idx_quadrant(self, xy_in: np.ndarray, r_norm: float = 3):
        # Use the x & y coordinate to cluster the data
        x_in = xy_in[:, 0]
        y_in = xy_in[:, 1]

        # First mark all data outside the normal circle as anomalous
        idx_anom = np.where(
            np.sqrt(np.square(x_in) + np.square(y_in)) > r_norm
        )

        # Then give the right gating decision based on the quadrant
        inside_r = np.sqrt(np.square(x_in) + np.square(y_in)) <= r_norm
        idx_1 = np.where(inside_r & (x_in > 0) & (y_in > 0))
        idx_2 = np.where(inside_r & (x_in < 0) & (y_in > 0))
        idx_3 = np.where(inside_r & (x_in < 0) & (y_in < 0))
        idx_4 = np.where(inside_r & (x_in > 0) & (y_in < 0))

        return idx_1, idx_2, idx_3, idx_4, idx_anom

    def quadrant_alarm(
            self, x_in: np.ndarray, y_in: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:

        # Work based on the code layer
        code_in = self.m_code.predict(x_in)

        y_pred = np.copy(y_in)
        y_gate = np.zeros((code_in.shape[0], 5))

        # Use the x & y coordinate to cluster the data
        idx_1, idx_2, idx_3, idx_4, idx_anom = self._get_idx_quadrant(code_in)

        # First mark all data outside the normal circle as anomalous
        y_pred[idx_anom, 0] = 1
        y_gate[idx_anom, -1] = 1

        # Then give the right gating decision based on the quadrant
        y_gate[idx_1, 0] = 1
        y_gate[idx_2, 1] = 1
        y_gate[idx_3, 2] = 1
        y_gate[idx_4, 3] = 1

        # # For debugging: Plot with color codes
        # c_in = np.zeros(y_in.shape)
        # c_in[idx_2] = 1
        # c_in[idx_3] = 2
        # c_in[idx_4] = 3
        # c_in[idx_anom] = 4
        # plt.scatter(code_in[:, 0], code_in[:, 1], c=c_in)
        # plt.show()

        return y_pred, y_gate

    def quadrant_mae(self, x_in) -> dict:
        # Work based on the code layer
        code_in = self.m_code.predict(x_in)
        x_quad = {}

        # Use the x & y coordinate to cluster the data
        idx_1, idx_2, idx_3, idx_4, idx_anom = self._get_idx_quadrant(code_in)

        # Add the four classes to the MAE dict
        for i_idx, cur_idx in enumerate([idx_1, idx_2, idx_3, idx_4]):
            x_quad[f"{self.label_normal}-{i_idx}"] = 2*[x_in[cur_idx]]

        # Equalise sizes
        x_quad = DataLabels.equalise_expert_data(x_quad)

        return x_quad

    def split_data(self, in_data: ExperimentData) -> ExperimentData:
        # Convert the different parts
        y_train_alarm, y_train_gate = self.quadrant_alarm(x_in=in_data.train_alarm[0], y_in=in_data.train_alarm[1])
        if isinstance(in_data.train_target, dict):
            x_train_mae = self.quadrant_mae(x_in=in_data.train_target[self.label_normal][0])
        else:
            x_train_mae = self.quadrant_mae(x_in=in_data.train_target[0])

        if isinstance(in_data.val_target, dict):
            x_val_mae = self.quadrant_mae(x_in=in_data.val_target[self.label_normal][0])
        else:
            x_val_mae = self.quadrant_mae(x_in=in_data.val_target[0])

        if isinstance(in_data.test_target, dict):
            x_test_mae = self.quadrant_mae(x_in=in_data.test_target[self.label_normal][0])
        else:
            x_test_mae = self.quadrant_mae(x_in=in_data.test_target[0])

        # We leave the val & test alarm data untouched as we need the original labels
        return ExperimentData(
            train_target=x_train_mae,
            train_alarm=(in_data.train_alarm[0], y_train_alarm),
            train_attention=y_train_gate,
            val_target=x_val_mae,
            val_alarm=in_data.val_alarm,
            val_attention=in_data.val_attention,
            test_target=x_test_mae,
            test_alarm=in_data.test_alarm,
            test_attention=in_data.test_attention,
            data_shape=in_data.data_shape,
            input_shape=in_data.input_shape
        )


class SplitKMeans(SplitQuadrant):
    def __init__(
            self, random_seed: int = None, cluster_path: Path = None,
            min_n_clusters: int = 2, max_n_clusters: int = 5, r_norm: float = 4.0,
            is_override: bool = False, **kwargs
    ):
        """
        Split the normal class into multiple clusters by applying KMeans on the AAE code layer
        :param target_ae: an autoencoder trained on the data to split
        :param data_source: data describing the experiment
        :param min_n_clusters: minimum number of clusters
        :param max_n_clusters: maximum number of clusters, if None use the very same as min_n_clusters
        :param r_norm: radius up to which the samples are considered normal, assume all points are normal if None
        :param random_seed: random seed
        :param cluster_path: path of the fitted clustering model, don't load or save if None
        :param use_random: assign to random clusters instead of kmeans
        """
        super(SplitKMeans, self).__init__(**kwargs)

        # Save the configuration
        self.min_n_clusters = min_n_clusters
        self.max_n_clusters = min_n_clusters if max_n_clusters is None else max_n_clusters
        self.r_norm = r_norm
        self.random_seed = random_seed
        self.n_cluster = None

        # Load or store the clustering model
        if (cluster_path is None or not cluster_path.exists()) or is_override:
            self._fit_cluster()
            if cluster_path is not None:
                joblib.dump(self.m_cluster, cluster_path)
        else:
            self.m_cluster = joblib.load(cluster_path)
            self.n_cluster = self.m_cluster.cluster_centers_.shape[0]

    @staticmethod
    def _extract_code(target_ae: tf.keras.Model):
        return target_ae.m_enc

    def _fit_cluster(self):
        print("Fitting KMeans to cluster the data")

        if isinstance(self.data_source.train_target, dict):
            x_code = self.m_code.predict(self.data_source.train_target[self.label_normal][0])
            x_code_val = self.m_code.predict(self.data_source.val_target[self.label_normal][0])
        else:
            x_code = self.m_code.predict(self.data_source.train_target[0])
            x_code_val = self.m_code.predict(self.data_source.val_target[0])

        # For CNNs there may be strange dimensions
        x_code = np.reshape(x_code, (x_code.shape[0], -1))
        x_code_val = np.reshape(x_code_val, (x_code_val.shape[0], -1))

        # Filter samples that are far away
        idx_norm, idx_anom = self._get_idx_norm(x_code)
        x_code = x_code[idx_norm]
        idx_norm_val, idx_anom_val = self._get_idx_norm(x_code_val)
        x_code_val = x_code_val[idx_norm_val]

        # Train the clustering model with this information
        m_clusters = []
        x_fit = []
        x_fit_val = []
        # Try a few cluster combinations
        n_clusters = list(range(self.min_n_clusters, self.max_n_clusters+1))
        for n_cluster in n_clusters:
            m_clusters.append(MiniBatchKMeans(n_clusters=n_cluster, random_state=self.random_seed))
            x_fit.append(m_clusters[-1].fit_predict(x_code))
            x_fit_val.append(m_clusters[-1].predict(x_code_val))

        # Look where the difference between the most abundant and the less abundant class is minimal
        class_count = [np.unique(cur_el, return_counts=True) for cur_el in x_fit]
        class_count_val = [np.unique(cur_el, return_counts=True) for cur_el in x_fit_val]
        class_spread = [np.max(cur_el[1]) - np.min(cur_el[1]) for cur_el in class_count]
        class_spread_val = [np.max(cur_el[1]) - np.min(cur_el[1]) for cur_el in class_count_val]
        class_var = [np.var(cur_el[1]) for cur_el in class_count]
        idx_min = np.argmin(np.array(class_spread))
        idx_min_val = np.argmin(np.array(class_spread_val))

        # Self the number of clusters and the respective model
        self.n_cluster = n_clusters[idx_min_val]
        self.m_cluster = m_clusters[idx_min_val]

    def _get_idx_norm(self, code_in: np.ndarray):

        # If no radius was set, assume that all point are normal
        if self.r_norm is None:
            return list(range(code_in.shape[0])), []

        # Get the radius of each entry
        r_in = np.sqrt(np.sum(np.square(code_in), axis=1))
        # Get the indices of points that lie within the ball
        idx_norm = np.where(r_in <= self.r_norm)
        idx_anom = np.where(r_in > self.r_norm)

        return idx_norm, idx_anom

    def quadrant_alarm(
            self, x_in: np.ndarray, y_in: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:

        # Work based on the code layer
        code_in = self.m_code.predict(x_in)
        # For CNNs there may be strange dimensions
        code_in = np.reshape(code_in, (code_in.shape[0], -1))

        y_pred = np.copy(y_in)
        y_gate = np.zeros((code_in.shape[0], self.n_cluster+1))

        # Filter based on the radius
        idx_norm, idx_anom = self._get_idx_norm(code_in)

        # Mark all data outside the normal circle as anomalous
        y_pred[idx_anom, 0] = 1
        y_gate[idx_anom, -1] = 1

        # Apply k-means
        y_class = self.m_cluster.predict(code_in)
        # Then give the right gating decision based on k-means
        y_gate[idx_norm, y_class[idx_norm]] = 1

        # # For debugging: Plot with color codes
        # c_in = np.zeros(y_in.shape)
        # c_in[idx_norm, 0] = y_class[idx_norm]
        # c_in[idx_anom, 0] = self.n_cluster+1
        # plt.scatter(code_in[:, 0], code_in[:, 1], c=c_in, cmap="tab20")
        # plt.show()

        return y_pred, y_gate

    def quadrant_mae(self, x_in) -> dict:
        # Work based on the code layer
        code_in = self.m_code.predict(x_in)
        # For CNNs there may be strange dimensions
        code_in = np.reshape(code_in, (code_in.shape[0], -1))
        x_quad = {}

        # Filter based on the radius
        idx_norm, idx_anom = self._get_idx_norm(code_in)
        # Apply k-means
        y_class = self.m_cluster.predict(code_in)

        # Split the data
        for cur_class in range(self.n_cluster):
            idx_class = np.where(y_class == cur_class)
            # Remove anomalous data
            idx_class = np.setdiff1d(idx_class, idx_anom)
            x_quad[f"{self.label_normal}-{cur_class}"] = 2*[x_in[idx_class]]

        # Equalise sizes
        x_quad = DataLabels.equalise_expert_data(x_quad)

        return x_quad


class SplitAAE(SplitQuadrant):
    def __init__(
            self, r_norm: float = None, **kwargs
    ):
        """
        Split the normal class into multiple clusters by using the clustering of an AAE
        :param r_norm: radius up to which the samples are considered normal, assume all points are normal if None
        """
        super(SplitAAE, self).__init__(**kwargs)

        # Save the configuration
        self.r_norm = r_norm

    @staticmethod
    def _extract_code(target_ae: tf.keras.Model):
        return target_ae.m_enc

    def _get_idx_norm(self, code_in: np.ndarray):

        # If no radius was set, assume that all point are normal
        if self.r_norm is None:
            return list(range(code_in.shape[0])), []

        # Get the radius of each entry
        r_in = np.sqrt(np.sum(np.square(code_in), axis=1))
        # Get the indices of points that lie within the ball
        idx_norm = np.where(r_in <= self.r_norm)
        idx_anom = np.where(r_in > self.r_norm)

        return idx_norm, idx_anom

    def quadrant_alarm(
            self, x_in: np.ndarray, y_in: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:

        # Extract the clustering decision
        all_in = self.m_code.predict(x_in)
        code_in = all_in[0]
        # class_in = tf.keras.layers.Activation("softmax")(all_in[1])
        class_in = all_in[1]
        # CNNs may result in weird dimensions
        code_in = np.reshape(code_in, (code_in.shape[0], code_in.shape[-1]))
        class_in = np.reshape(class_in, (class_in.shape[0], class_in.shape[-1]))
        n_class = class_in.shape[1]
        # The clusters are the maximum value in each row
        y_class = np.argmax(class_in, axis=1)

        # Determine the labels
        y_pred = np.copy(y_in)
        y_gate = np.eye(n_class+1)[y_class]

        # Filter based on the radius
        idx_norm, idx_anom = self._get_idx_norm(code_in)
        # Mark all data outside the normal circle as anomalous
        y_pred[idx_anom, 0] = 1
        gate_anom = np.zeros((1, n_class+1))
        gate_anom[0, -1] = 1
        y_gate[idx_anom, :] = gate_anom

        # # For debugging: Plot with color codes
        # y_debug = y_class.copy()
        # y_debug[idx_anom] = y_class.max() + 1
        # plt.scatter(code_in[:, 0], code_in[:, 1], c=y_debug, cmap="tab20")
        # plt.show()
        # # Show samples in the respective class
        # x_clust_0 = x_in[y_class == 0, :]
        # x_clust_1 = x_in[y_class == 1, :]
        # x_clust_2 = x_in[y_class == 2, :]
        # x_clust_3 = x_in[y_class == 3, :]
        # x_clust_4 = x_in[y_class == 4, :]

        return y_pred, y_gate

    def quadrant_mae(self, x_in) -> dict:
        # Extract the clustering decision
        code_in = self.m_code.predict(x_in)[0]
        class_in = self.m_code.predict(x_in)[1]
        # CNNs may result in weird dimensions
        code_in = np.reshape(code_in, (code_in.shape[0], code_in.shape[-1]))
        class_in = np.reshape(class_in, (class_in.shape[0], class_in.shape[-1]))
        # The clusters are the maximum value in each row
        y_class = np.argmax(class_in, axis=1)

        # Filter based on the radius
        idx_norm, idx_anom = self._get_idx_norm(code_in)

        x_quad = {}
        # Split the data
        for cur_class in range(class_in.shape[1]):
            idx_class = np.where(y_class == cur_class)
            # Remove anomalous data
            idx_class = np.setdiff1d(idx_class, idx_anom)
            x_quad[f"{self.label_normal}-{cur_class}"] = 2*[x_in[idx_class]]

        # Equalise sizes
        x_quad = DataLabels.equalise_expert_data(x_quad)

        return x_quad
