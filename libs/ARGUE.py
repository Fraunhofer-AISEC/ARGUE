import tensorflow as tf

from typing import Union, Dict, Tuple

from libs.architecture.target import MultiAutoencoder
from libs.network.network import add_dense


class ARGUE(tf.keras.Model):
    def __init__(
            self, layer_dims: tuple, m_target: MultiAutoencoder = None, name="ARGUE",
            hidden_activation: str = "relu", p_dropout: float = .1,
            full_act: bool = True
    ):
        """
        Anomaly Detection by Recombining Gated Unsupervised Experts
        :param layer_dims: dimensions of the (dense) alarm network
        :param m_target: pretrained multi-headed autoencoder
        :param name: name of the model
        :param hidden_activation: activation function of the alarm network
        :param p_dropout: dropout likelihood for the alarm network
        :param full_act: if True combine activations of the encoder and decoders, otherwise take the latter only
        """
        super(ARGUE, self).__init__(name=name)

        # Config
        self.layer_dims = layer_dims
        self.n_experts = None
        self.hidden_activation = hidden_activation
        self.p_dropout = p_dropout
        self.full_act = full_act

        # Network components
        self.m_mae: MultiAutoencoder = None
        self.m_gating: tf.keras.Model = None
        self.m_alarm: tf.keras.Model = None
        if m_target is not None: self.add_target(m_target)

        # Losses
        self.loss_gating = None
        self.loss_alarm = None

        # Optimiser
        self.opt_gating: tf.keras.optimizers.Optimizer = None
        self.opt_alarm: tf.keras.optimizers.Optimizer = None

    # == Helper functions ==
    def add_target(self, m_target: MultiAutoencoder):
        m_target.trainable = False
        self.m_mae = m_target

        # Extract the number of experts
        self.n_experts = len(self.m_mae.m_decs)

    # == Keras functions ==
    def compile(
            self,
            learning_rate=0.0001,
            loss_gating=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
            loss_alarm=tf.keras.losses.BinaryCrossentropy(from_logits=False),
            gate_factor=.5,
            **kwargs
    ):
        super(ARGUE, self).compile(**kwargs)

        self.loss_gating = loss_gating
        self.loss_alarm = loss_alarm

        # We'll use adam as default optimiser
        self.opt_gating = tf.keras.optimizers.Adam(gate_factor*learning_rate)
        self.opt_alarm = tf.keras.optimizers.Adam(learning_rate)

    def build(self, input_shape):

        # Alarm & gating use the same hidden dimensions, but other output sizes
        m_alarm = tf.keras.models.Sequential(name="Alarm")
        add_dense(m_alarm, layer_dims=self.layer_dims, activation=self.hidden_activation, p_dropout=self.p_dropout)
        m_alarm.add(tf.keras.layers.Dense(1, activation="sigmoid"))
        self.m_alarm = m_alarm

        m_gating = tf.keras.models.Sequential(name="Gating")
        add_dense(m_gating, layer_dims=self.layer_dims, activation=self.hidden_activation, p_dropout=self.p_dropout)
        m_gating.add(tf.keras.layers.Dense(self.n_experts + 1, activation="softmax"))
        self.m_gating = m_gating

    def train_step(self, data, add_noise: bool = False, use_teacher: bool = False, equal_weight: bool = False):

        # Extract the data
        x_train = data[0]
        y_train_alarm = data[1][0]
        y_train_gating = data[1][1]

        # Trivial anomalies
        x_noise = tf.random.normal(shape=tf.shape(x_train), mean=.5, stddev=1.0)
        y_noise_alarm = tf.keras.backend.ones_like(y_train_alarm)
        # Read: 1-hot vector with 1 at the position of n_experts (i.e. the last one)
        y_noise_gating = tf.one_hot(tf.fill(tf.shape(y_train_alarm), self.n_experts)[:, 0], self.n_experts + 1)\
            if not equal_weight else tf.fill(tf.shape(y_train_gating), 1/(self.n_experts+1))

        if add_noise:
            y_train_alarm = tf.cast(y_train_alarm, tf.float32) + tf.random.normal(tf.shape(y_train_alarm), stddev=.01)
            y_noise_alarm = tf.cast(y_noise_alarm, tf.float32) + tf.random.normal(tf.shape(y_noise_alarm), stddev=.01)
            y_noise_gating = tf.cast(y_noise_gating, tf.float32) + tf.random.normal(tf.shape(y_noise_gating),
                                                                                    stddev=.01)

        # == Alarm: use activations of each decoder ==
        with tf.GradientTape() as alarm_tape:
            # Combine them using the gating decision
            y_pred_alarm_train = self([x_train, y_train_gating], training=True) if use_teacher else self(x_train, training=True)
            y_pred_alarm_noise = self([x_noise, y_noise_gating], training=True) if use_teacher else self(x_noise, training=True)

            # Match the training labels
            loss_alarm = self.loss_alarm(y_true=y_train_alarm, y_pred=y_pred_alarm_train) \
                         + self.loss_alarm(y_true=y_noise_alarm, y_pred=y_pred_alarm_noise)

        # Backpropagate
        grad_alarm = alarm_tape.gradient(loss_alarm, self.m_alarm.trainable_weights)
        self.opt_alarm.apply_gradients(zip(grad_alarm, self.m_alarm.trainable_weights))

        # == Gating: take encoder activations as input ==
        t_enc_act_train = self.m_mae.m_enc_act(x_train, training=False)
        t_enc_act_noise = self.m_mae.m_enc_act(x_noise, training=False)

        with tf.GradientTape() as gating_tape:
            # The gating decision is either expert path or the "anomaly expert"
            y_pred_train_gating = self.m_gating(t_enc_act_train, training=True)
            y_pred_noise_gating = self.m_gating(t_enc_act_noise, training=True)

            loss_gating = self.loss_gating(y_true=y_train_gating, y_pred=y_pred_train_gating) \
                          + self.loss_gating(y_true=y_noise_gating, y_pred=y_pred_noise_gating)

        # Backpropagate
        grad_gating = gating_tape.gradient(loss_gating, self.m_gating.trainable_weights)
        self.opt_gating.apply_gradients(zip(grad_gating, self.m_gating.trainable_weights))

        return {
            "Gating": loss_gating,
            "Alarm": loss_alarm
        }

    def call(self, inputs, training=None, mask=None):

        if isinstance(inputs, list):
            is_teacher = True
            x_in = inputs[0]
            teacher_gate = tf.cast(inputs[1], tf.float32)
        else:
            is_teacher = False
            x_in = inputs

        # Get the gating decision
        t_enc_act = self.m_mae.m_enc_act(x_in, training=False)
        t_gating = self.m_gating(t_enc_act, training=training, mask=mask)

        # Get the alarm decisions
        t_dec_acts = self.m_mae.m_dec_act(x_in, training=False)
        # If this is only one tensor, convert to list
        if isinstance(t_dec_acts, tf.Tensor):
            t_dec_acts = [t_dec_acts]
        all_t_alarm = []
        for cur_t_dec_act in t_dec_acts:
            t_act_in = tf.concat([t_enc_act, cur_t_dec_act], axis=1) if self.full_act else cur_t_dec_act
            all_t_alarm.append(self.m_alarm(t_act_in, training=training, mask=mask))
        # Add the "anomaly expert"
        all_t_alarm.append(tf.keras.backend.ones_like(all_t_alarm[0]))
        # Convert to tensor
        all_t_alarm = tf.concat(all_t_alarm, axis=1)

        # Weight decisions
        y_pred = tf.reduce_sum(t_gating * all_t_alarm, axis=1) if not is_teacher \
            else tf.reduce_sum(teacher_gate * all_t_alarm, axis=1)

        return y_pred

    def get_config(self):
        config = {
            'layer_dims': self.layer_dims,
        }

        base_config = super(ARGUE, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
